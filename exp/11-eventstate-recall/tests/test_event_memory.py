from event_memory import EventStateMemory


def test_state_update_closes_old_version_without_erasing_identity():
    memory = EventStateMemory()
    v1 = memory.update_state("person_a", {"coat": "red"}, at_chunk=1)
    v2 = memory.update_state("person_a", {"coat": "blue"}, at_chunk=12)
    assert v1.valid_to_chunk == 11
    assert v2.supersedes == v1.version_id
    assert memory.valid_state("person_a", 8).attributes["coat"] == "red"
    assert memory.valid_state("person_a", 20).attributes["coat"] == "blue"


def test_event_plan_intersects_physical_residency():
    memory = EventStateMemory()
    memory.create_or_extend_event("shot_1", "artist appears", range(0, 8), ["artist"])
    plan = memory.layer_recall_plan((16, 23), ["shot_1"], resident_chunk_ids=range(3, 16))
    assert plan["layer_recall_agent_plan"][0]["preferred_chunks"] == [3, 4, 5, 6, 7]


def test_event_extension_and_reappearance_link_are_deduplicated():
    memory = EventStateMemory()
    memory.create_or_extend_event("a", "first", [0, 1], ["person"])
    memory.create_or_extend_event("a", "first extended", [1, 2], ["person"])
    memory.create_or_extend_event("b", "return", [10], ["person"])
    memory.link_reappearance("a", "b")
    memory.link_reappearance("a", "b")
    assert memory.events["a"].chunk_ids == [0, 1, 2]
    assert memory.events["a"].links["reappears_in"] == ["b"]
