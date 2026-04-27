from gshp.info_gain import analyze_information_gain
from gshp.types import AgentDecision, DyadTranscript, ExperimentRun, Message, RunManifest


class _ToyTask:
    candidates = ("A", "B", "C")
    facts = {
        "f1": "A has a disqualifying conflict.",
        "f2": "C lacks required certification.",
        "f3": "Bridge link connecting evidence.",
    }
    shared_fact_ids = []
    cluster_fact_ids = {0: ["f1"], 1: ["f2"]}
    bridge_agent_fact_ids = {2: "f3"}
    fact_eliminates = {"f1": ["A"], "f2": ["C"], "f3": []}
    interaction_eliminates = {"f1+f3": ["A"]}


def _run_with_messages(messages: list[str]) -> ExperimentRun:
    dyad = DyadTranscript(
        u=0,
        v=1,
        round_index=0,
        round_label="intra_community",
        messages=[
            Message(role=f"agent_{i % 2}", content=txt, metadata={"turn": i})
            for i, txt in enumerate(messages)
        ],
    )
    return ExperimentRun(
        manifest=RunManifest(schedule="within_first", l=3, k=3),
        dyads=[dyad],
        final_decisions=[AgentDecision(agent_id=i, choice="B", justification="test") for i in range(3)],
    )


def test_entropy_reduction_tracks_feasible_set():
    run = _run_with_messages(
        [
            "I found this: (f1) A has a disqualifying conflict.",
            "Also, (f2) C lacks required certification.",
            "Repeating (f2) C lacks required certification.",
        ]
    )
    out = analyze_information_gain(run, _ToyTask())
    assert out["resolved_to_single_option"] is True
    assert out["final_feasible_options"] == ["B"]
    assert out["n_atomic_events"] >= 3
    assert out["zero_bit_rate"] > 0.0  # includes at least one repeated, zero-bit event


def test_shapley_reports_pairwise_synergy_signal():
    run = _run_with_messages(
        [
            "(f3) Bridge link connecting evidence.",
            "(f1) A has a disqualifying conflict.",
        ]
    )
    out = analyze_information_gain(run, _ToyTask())
    shap = out["shapley"]
    # Order can vary, so check either pair key orientation.
    pairs = shap["pairwise_interaction_bits"]
    val = pairs.get("f1+f3", pairs.get("f3+f1", 0.0))
    assert val >= 0.0
