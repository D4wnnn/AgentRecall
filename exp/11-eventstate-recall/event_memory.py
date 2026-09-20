"""Versioned event-state memory whose semantic nodes address LayerRecall chunk IDs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class StateVersion:
    version_id: str
    entity_id: str
    attributes: Dict[str, Any]
    valid_from_chunk: int
    valid_to_chunk: Optional[int] = None
    supersedes: Optional[str] = None
    confidence: float = 1.0

    def is_valid_at(self, chunk_id: int) -> bool:
        return self.valid_from_chunk <= chunk_id and (
            self.valid_to_chunk is None or chunk_id <= self.valid_to_chunk
        )


@dataclass
class EventNode:
    event_id: str
    summary: str
    chunk_ids: List[int] = field(default_factory=list)
    entity_ids: List[str] = field(default_factory=list)
    confidence: float = 1.0
    links: Dict[str, List[str]] = field(default_factory=dict)


class EventStateMemory:
    """Small serializable controller state; it never stores tensor K/V itself."""

    def __init__(self) -> None:
        self.events: Dict[str, EventNode] = {}
        self.versions: Dict[str, List[StateVersion]] = {}

    def create_or_extend_event(
        self,
        event_id: str,
        summary: str,
        chunk_ids: Iterable[int],
        entity_ids: Iterable[str] = (),
        confidence: float = 1.0,
    ) -> EventNode:
        node = self.events.get(event_id)
        if node is None:
            node = EventNode(event_id=event_id, summary=summary)
            self.events[event_id] = node
        elif summary:
            node.summary = summary
        node.chunk_ids = sorted(set(node.chunk_ids).union(int(x) for x in chunk_ids))
        node.entity_ids = sorted(set(node.entity_ids).union(str(x) for x in entity_ids))
        node.confidence = float(confidence)
        return node

    def update_state(
        self,
        entity_id: str,
        attributes: Dict[str, Any],
        at_chunk: int,
        confidence: float = 1.0,
    ) -> StateVersion:
        versions = self.versions.setdefault(entity_id, [])
        previous = versions[-1] if versions else None
        if previous is not None and previous.valid_to_chunk is None:
            previous.valid_to_chunk = int(at_chunk) - 1
        version = StateVersion(
            version_id=f"{entity_id}:v{len(versions) + 1}",
            entity_id=entity_id,
            attributes=dict(attributes),
            valid_from_chunk=int(at_chunk),
            supersedes=previous.version_id if previous else None,
            confidence=float(confidence),
        )
        versions.append(version)
        return version

    def valid_state(self, entity_id: str, at_chunk: int) -> Optional[StateVersion]:
        for version in reversed(self.versions.get(entity_id, [])):
            if version.is_valid_at(int(at_chunk)):
                return version
        return None

    def link_reappearance(self, source_event: str, target_event: str) -> None:
        source = self.events[source_event]
        source.links.setdefault("reappears_in", [])
        if target_event not in source.links["reappears_in"]:
            source.links["reappears_in"].append(target_event)

    def candidate_chunks(
        self,
        event_ids: Iterable[str],
        resident_chunk_ids: Optional[Iterable[int]] = None,
    ) -> List[int]:
        candidates = set()
        for event_id in event_ids:
            if event_id in self.events:
                candidates.update(self.events[event_id].chunk_ids)
        if resident_chunk_ids is not None:
            candidates.intersection_update(int(x) for x in resident_chunk_ids)
        return sorted(candidates)

    def layer_recall_plan(
        self,
        current_chunks: Tuple[int, int],
        event_ids: Iterable[str],
        resident_chunk_ids: Optional[Iterable[int]] = None,
    ) -> Dict[str, Any]:
        return {
            "layer_recall_agent_plan": [{
                "current_chunks": [int(current_chunks[0]), int(current_chunks[1])],
                "preferred_chunks": self.candidate_chunks(event_ids, resident_chunk_ids),
            }]
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "events": {key: asdict(value) for key, value in self.events.items()},
            "versions": {
                key: [asdict(version) for version in values]
                for key, values in self.versions.items()
            },
        }
