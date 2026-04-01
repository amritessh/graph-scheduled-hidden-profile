"""Aho–Corasick automaton + fact metrics consistency."""

from gshp.aho_corasick import AhoCorasickAutomaton
from gshp.metrics import fact_mention_rates


def test_automaton_basic_overlaps():
    ac = AhoCorasickAutomaton(["he", "she", "hers", "his"])
    # Classic AC example: "ushers" matches he, she, hers at overlapping endings.
    assert ac.matching_pattern_indices("ushers") == {0, 1, 2}


def test_fact_mention_rates_matches_naive_substrings():
    facts = {"a": "foo bar", "b": "uniqueX", "empty": ""}
    calls = [
        {
            "system": "FOO BAR mentioned",
            "messages": [{"role": "user", "content": "nothing"}],
            "response": "uniquex lower",
        }
    ]

    blob_parts: list[str] = []
    for c in calls:
        blob_parts.append(c.get("system") or "")
        for m in c.get("messages") or []:
            if isinstance(m, dict):
                blob_parts.append(m.get("content") or "")
        blob_parts.append(c.get("response") or "")
    blob = "\n".join(blob_parts).lower()

    naive = {}
    for fid in sorted(facts.keys()):
        needle = (facts[fid] or "").strip().lower()
        naive[fid] = bool(needle and needle in blob)

    got = fact_mention_rates(facts, calls)
    assert got["per_fact_mentioned"] == naive
    assert got["matcher"] == "aho_corasick"
