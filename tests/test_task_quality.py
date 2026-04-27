from gshp.task.generator import _quality_report


def test_quality_report_passes_well_formed_task():
    data = {
        "options": ["A", "B", "C"],
        "correct_option": "B",
        "facts": {
            "s1": "shared attractor support",
            "c1": "cluster evidence against A",
            "c2": "cluster evidence against C",
        },
        "shared_fact_ids": ["s1"],
        "cluster_fact_ids": {0: ["c1"], 1: ["c2"], 2: []},
        "bridge_agent_fact_ids": {},
        "fact_eliminates": {"s1": [], "c1": ["A"], "c2": ["C"]},
        "interaction_eliminates": {},
    }
    r = _quality_report(data)
    assert r["score"] >= 0.7
    assert r["remaining_with_all_facts"] == ["B"]


def test_quality_report_flags_bad_full_resolution():
    data = {
        "options": ["A", "B", "C"],
        "correct_option": "B",
        "facts": {"s1": "shared", "c1": "cluster"},
        "shared_fact_ids": ["s1"],
        "cluster_fact_ids": {0: ["c1"], 1: [], 2: []},
        "bridge_agent_fact_ids": {},
        "fact_eliminates": {"s1": [], "c1": ["A"]},
        "interaction_eliminates": {},
    }
    r = _quality_report(data)
    assert r["score"] < 0.7
    assert any("full information does not reduce" in issue for issue in r["issues"])
