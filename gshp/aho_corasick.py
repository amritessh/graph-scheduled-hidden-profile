"""Multi-pattern substring search (Aho–Corasick) — pure Python, no extra deps."""

from __future__ import annotations

from collections import deque


class AhoCorasickAutomaton:
    """
    Build once from ``patterns`` (each treated as a literal UTF-8 / Unicode string),
    then scan ``text`` in O(len(text) + matches + alphabet) time.

    This is used for **cheap bulk disclosure checks** over many short fact strings;
    see ``docs/algorithms.md`` for limitations vs LLM judges or calibration.
    """

    __slots__ = ("_go", "_link", "_out", "patterns")

    def __init__(self, patterns: list[str]) -> None:
        self.patterns = list(patterns)
        # Trie: transitions list[dict[ch -> state]]
        go: list[dict[str, int]] = [{}]
        link: list[int] = [0]
        out: list[list[int]] = [[]]

        for pid, pat in enumerate(self.patterns):
            state = 0
            for ch in pat:
                nxt = go[state].get(ch)
                if nxt is None:
                    nxt = len(go)
                    go[state][ch] = nxt
                    go.append({})
                    link.append(0)
                    out.append([])
                state = nxt
            out[state].append(pid)

        q: deque[int] = deque()
        for _, nxt in go[0].items():
            link[nxt] = 0
            q.append(nxt)

        while q:
            v = q.popleft()
            for ch, u in go[v].items():
                q.append(u)
                j = link[v]
                while j != 0 and ch not in go[j]:
                    j = link[j]
                link[u] = go[j].get(ch, 0)
                # Output propagation: patterns ending at fail[u] also end here via suffix.
                if link[u] != 0:
                    out[u].extend(out[link[u]])

        self._go = go
        self._link = link
        self._out = out

    def matching_pattern_indices(self, text: str) -> set[int]:
        """Return indices ``i`` such that ``patterns[i]`` occurs as a substring of ``text``."""
        go, link, out = self._go, self._link, self._out
        state = 0
        found: set[int] = set()
        for ch in text:
            while state != 0 and ch not in go[state]:
                state = link[state]
            state = go[state].get(ch, 0)
            if out[state]:
                found.update(out[state])
        return found
