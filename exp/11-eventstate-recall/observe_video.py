#!/usr/bin/env python3
"""Build a versioned event trace and LayerRecall plan with local Qwen3-VL."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import av
import torch
from jsonschema import ValidationError, validate
from PIL import Image, ImageDraw
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    from json_repair import repair_json
except ImportError:  # Optional: strict JSON plus model retry still works without this package.
    repair_json = None

from event_memory import EventStateMemory


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError(f"model did not return a JSON object: {text[:300]!r}")
        text = text[start:end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if repair_json is None:
            raise
        repaired = repair_json(text, return_objects=True)
        if not isinstance(repaired, dict):
            raise ValueError("repaired model output is not a JSON object")
        return repaired


def compact_validation_error(error: Exception) -> str:
    if isinstance(error, json.JSONDecodeError):
        return f"invalid JSON near line {error.lineno}, column {error.colno}: {error.msg}"
    if isinstance(error, ValidationError):
        location = ".".join(str(item) for item in error.absolute_path) or "root"
        return f"schema error at {location}: {error.message}"
    return str(error)[:500]


def normalize_event_types(payload: Dict[str, Any]) -> List[str]:
    """Repair harmless container-type drift without inventing semantic content."""
    repairs: List[str] = []
    for index, observation in enumerate(payload.get("chunk_observations", [])):
        for field in ("visible_entities", "state_delta"):
            value = observation.get(field)
            if isinstance(value, str):
                observation[field] = [value] if value else []
                repairs.append(f"chunk_observations.{index}.{field}: string_to_list")
    for field in ("related_event_ids",):
        value = payload.get(field)
        if isinstance(value, str):
            payload[field] = [value] if value else []
            repairs.append(f"{field}: string_to_list")
        elif isinstance(value, list) and any(not isinstance(item, str) for item in value):
            payload[field] = [str(item) for item in value]
            repairs.append(f"{field}: items_to_string")
    return repairs


def read_case(case_dir: Path) -> Tuple[List[int], List[str]]:
    durations = [int(x) for x in (case_dir / "shot_durations.txt").read_text().split()]
    captions = [
        json.loads((case_dir / f"{index}.json").read_text(encoding="utf-8"))["caption"]
        for index in range(len(durations))
    ]
    return durations, captions


def decode_all(video_path: Path) -> List[Image.Image]:
    container = av.open(str(video_path))
    frames = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
    container.close()
    return frames


def chunk_center_frame(decoded_frames: Sequence[Image.Image], chunk_id: int, total_chunks: int):
    index = round((int(chunk_id) + 0.5) * len(decoded_frames) / total_chunks)
    return decoded_frames[min(max(index, 0), len(decoded_frames) - 1)]


def event_montage(
    decoded_frames: Sequence[Image.Image],
    chunk_ids: Sequence[int],
    total_chunks: int,
    output_path: Path,
) -> Path:
    thumb_w, thumb_h = 336, 202
    cols = min(4, len(chunk_ids))
    rows = (len(chunk_ids) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 28)), "white")
    draw = ImageDraw.Draw(canvas)
    for position, chunk_id in enumerate(chunk_ids):
        image = chunk_center_frame(decoded_frames, chunk_id, total_chunks).resize((thumb_w, thumb_h))
        x, y = (position % cols) * thumb_w, (position // cols) * (thumb_h + 28)
        canvas.paste(image, (x, y + 28))
        draw.text((x + 8, y + 8), f"chunk {chunk_id}", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
    return output_path


def compile_layer_recall_plan(
    events: Sequence[Dict[str, Any]],
    decisions: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compile every typed event decision, including explicit empty/abstain ranges."""
    chunks_by_event = {event["event_id"]: event.get("chunk_ids", []) for event in events}
    entries = []
    for event, item in zip(events, decisions):
        decision = item["decision"]
        selected_events = decision.get("target_event_ids", []) if decision.get("action") == "RECALL" else []
        preferred = sorted({
            int(chunk_id)
            for event_id in selected_events
            for chunk_id in chunks_by_event.get(event_id, [])
        })
        chunks = [int(item) for item in event["chunk_ids"]]
        entries.append({
            "current_chunks": [min(chunks), max(chunks)],
            "preferred_chunks": preferred,
            "action": decision.get("action", "IGNORE"),
        })
    return {"layer_recall_agent_plan": entries}


class QwenEventObserver:
    def __init__(self, model_path: Path):
        self.processor = AutoProcessor.from_pretrained(str(model_path))
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model_path),
            dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="flash_attention_2",
        ).eval()

    def generate(self, messages: List[Dict[str, Any]], max_new_tokens: int = 1024) -> str:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, padding=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def generate_json(
        self,
        messages: List[Dict[str, Any]],
        schema: Dict[str, Any] | None = None,
        max_new_tokens: int = 1536,
        retries: int = 2,
        normalize_event: bool = False,
        expected_chunk_ids: Sequence[int] | None = None,
    ) -> Tuple[Dict[str, Any], List[str], List[str]]:
        """Generate validated JSON, retaining every raw attempt for auditability."""
        attempts: List[str] = []
        current_messages = messages
        last_error: Exception | None = None
        all_repairs: List[str] = []
        for _ in range(retries + 1):
            raw = self.generate(current_messages, max_new_tokens=max_new_tokens)
            attempts.append(raw)
            try:
                parsed = extract_json(raw)
                if normalize_event:
                    all_repairs.extend(normalize_event_types(parsed))
                if schema is not None:
                    validate(parsed, schema)
                if expected_chunk_ids is not None:
                    actual = [
                        int(item["chunk_id"])
                        for item in parsed.get("chunk_observations", [])
                    ]
                    expected = [int(item) for item in expected_chunk_ids]
                    if sorted(actual) != sorted(expected) or len(actual) != len(expected):
                        raise ValueError(
                            f"chunk_observations IDs must be exactly {expected}, got {actual}"
                        )
                return parsed, attempts, all_repairs
            except (ValueError, json.JSONDecodeError, ValidationError) as error:
                last_error = error
                repair_prompt = (
                    "Repair the candidate below into exactly one valid JSON object. Preserve only "
                    "claims already present; do not add visual facts. Output JSON only.\n"
                    f"Validation failure: {compact_validation_error(error)}\n"
                    f"Candidate:\n{raw}"
                )
                # Retain the original image on retries so missing visual observations can be
                # regenerated from evidence instead of guessed from a truncated candidate.
                current_messages = messages + [
                    {"role": "assistant", "content": [{"type": "text", "text": raw}]},
                    {"role": "user", "content": [{"type": "text", "text": repair_prompt}]},
                ]
        raise ValueError(
            f"JSON generation failed after {len(attempts)} attempts: "
            f"{compact_validation_error(last_error or ValueError('unknown error'))}"
        )

    def observe_event(
        self,
        montage_path: Path,
        event_id: str,
        chunk_ids: Sequence[int],
        caption: str,
        previous_events: Sequence[Dict[str, Any]],
        schema: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str], List[str]]:
        contract = {
            "event_id": event_id,
            "operation": "CREATE_EVENT|EXTEND_EVENT|UPDATE_STATE|INVALIDATE_STATE|LINK_REAPPEARANCE",
            "summary": "short factual summary",
            "confidence": 0.0,
            "related_event_ids": [],
            "chunk_observations": [{
                "chunk_id": int(chunk_ids[0]),
                "visible_entities": ["stable entity IDs"],
                "scene": "what is visibly present",
                "state_delta": [],
            }],
            "entity_states": [{
                "entity_id": "stable ID",
                "attributes": {"appearance": "visible facts only"},
                "valid": True,
                "supersedes": None,
            }],
        }
        prompt = (
            "You are the slow observer in an online long-video memory system. The image is a "
            "chronological montage labeled by chunk ID. Describe visible evidence, not merely "
            "the requested caption. Track stable entity identity across prior events, distinguish "
            "temporary absence from a true state update, and use LINK_REAPPEARANCE when the same "
            "entity returns. Return exactly one compact JSON object and no prose. Use exactly "
            "one chunk_observations entry for every supplied chunk ID. Keep each scene under "
            "18 words, each state_delta under 10 words, and do not repeat entity attributes in "
            "every chunk. Keep the whole answer under 900 tokens.\n"
            f"Current event ID: {event_id}\nChunk IDs: {list(chunk_ids)}\n"
            f"Generation caption (intent only; visual evidence wins): {caption}\n"
            f"Prior events: {json.dumps(previous_events, ensure_ascii=False)}\n"
            f"Required shape example: {json.dumps(contract, ensure_ascii=False)}"
        )
        messages = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{montage_path.resolve()}"},
            {"type": "text", "text": prompt},
        ]}]
        return self.generate_json(
            messages, schema=schema, max_new_tokens=1536, normalize_event=True,
            expected_chunk_ids=chunk_ids,
        )

    def plan_return(
        self,
        event_summaries: Sequence[Dict[str, Any]],
        current_event_id: str,
        current_caption: str,
    ) -> Tuple[Dict[str, Any], List[str], List[str]]:
        prompt = (
            "Act as a bounded-memory controller immediately BEFORE the current video event is "
            "generated. Event memory contains only completed historical events; the current caption "
            "is an instruction, not an observation of an already generated result. Select historical "
            "events whose valid state should guide generation. Do not choose exact neural chunks. "
            "RECALL when the caption says an earlier entity returns, reappears, is the same, or "
            "resumes after absence/occlusion; choose the historical event where it was visibly "
            "established. KEEP_CURRENT only when the immediately preceding observed event already "
            "contains all required entities. UPDATE only for an explicit persistent attribute/state "
            "change; a camera pan or temporary absence never invalidates an entity. "
            "invalid_event_ids is ONLY for historical events containing an explicitly superseded "
            "persistent state, never for temporary absence or an irrelevant camera view. IGNORE means no "
            "historical evidence is needed. target_event_ids must be drawn only from Event memory. "
            "Return JSON only with keys action (RECALL, KEEP_CURRENT, UPDATE, or IGNORE), "
            "target_event_ids, invalid_event_ids, target_entities, confidence, and rationale.\n"
            f"Event memory: {json.dumps(event_summaries, ensure_ascii=False)}\n"
            f"Current event: {current_event_id}\nCurrent caption: {current_caption}"
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self.generate_json(messages, max_new_tokens=384)


def build_trace(args):
    durations, captions = read_case(args.case_dir)
    total_chunks = sum(durations)
    frames = decode_all(args.video)
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    observer = QwenEventObserver(args.model)
    memory = EventStateMemory()
    events, raw_outputs, decisions = [], [], []
    chunk_start = 0
    started = time.perf_counter()
    for shot_index, (duration, caption) in enumerate(zip(durations, captions), start=1):
        chunk_ids = list(range(chunk_start, chunk_start + duration))
        event_id = f"shot_{shot_index}"
        # This decision is intentionally made before the current event is observed. It mirrors
        # generation-time use and prevents the agent from peeking at the result it is meant to guide.
        if events:
            decision, raw_plan, plan_repairs = observer.plan_return(events, event_id, caption)
        else:
            decision = {
                "action": "KEEP_CURRENT", "target_event_ids": [], "invalid_event_ids": [],
                "target_entities": [], "confidence": 1.0,
                "rationale": "No historical event exists.",
            }
            raw_plan = []
            plan_repairs = []
        decisions.append({"event_id": event_id, "decision": decision})
        raw_outputs.append({
            "event_id": event_id, "stage": "pre_generation_plan",
            "attempts": raw_plan, "structural_repairs": plan_repairs,
        })
        montage = event_montage(
            frames, chunk_ids, total_chunks, args.output_dir / "montages" / f"{event_id}.jpg"
        )
        event, raw, event_repairs = observer.observe_event(
            montage, event_id, chunk_ids, caption, events, schema
        )
        event["chunk_ids"] = chunk_ids
        events.append(event)
        raw_outputs.append({
            "event_id": event_id, "stage": "post_generation_observation",
            "attempts": raw, "structural_repairs": event_repairs,
        })
        entity_ids = [item["entity_id"] for item in event.get("entity_states", [])]
        entity_ids.extend(
            entity_id
            for observation in event.get("chunk_observations", [])
            for entity_id in observation.get("visible_entities", [])
        )
        memory.create_or_extend_event(
            event_id, event["summary"], chunk_ids, entity_ids, event["confidence"]
        )
        for state in event.get("entity_states", []):
            if state.get("valid", True):
                previous = memory.valid_state(state["entity_id"], chunk_ids[0])
                if previous is None or previous.attributes != state.get("attributes", {}):
                    memory.update_state(
                        state["entity_id"], state.get("attributes", {}), chunk_ids[0],
                        event["confidence"],
                    )
        if event.get("operation") == "LINK_REAPPEARANCE":
            for source_event in event.get("related_event_ids", []):
                if source_event in memory.events and source_event != event_id:
                    memory.link_reappearance(source_event, event_id)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "agent_trace.partial.json").write_text(
            json.dumps({
                "model": str(args.model), "video": str(args.video),
                "case_dir": str(args.case_dir), "events": events,
                "pre_generation_decisions": decisions,
                "event_state_memory": memory.to_dict(), "raw_outputs": raw_outputs,
                "elapsed_seconds": time.perf_counter() - started,
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        chunk_start += duration

    decision = decisions[-1]["decision"]
    plan = compile_layer_recall_plan(events, decisions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": str(args.model),
        "video": str(args.video),
        "case_dir": str(args.case_dir),
        "events": events,
        "pre_generation_decisions": decisions,
        "decision": decision,
        "event_state_memory": memory.to_dict(),
        "layer_recall_plan": plan,
        "elapsed_seconds": time.perf_counter() - started,
        "raw_outputs": raw_outputs,
    }
    (args.output_dir / "agent_trace.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "layer_recall_plan.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "events": len(events), "decision": decision, "plan": plan,
        "elapsed_seconds": payload["elapsed_seconds"],
    }, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=Path("/private/lc/download/Qwen3-VL-4B-Instruct"))
    parser.add_argument("--schema", type=Path, default=Path(__file__).parent / "schemas/event_observation.schema.json")
    parser.add_argument("--output-dir", required=True, type=Path)
    build_trace(parser.parse_args())


if __name__ == "__main__":
    main()
