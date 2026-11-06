from dataclasses import dataclass
from collections.abc import Callable
from collections import deque
from typing import Dict, List


@dataclass
class DFA[STATE]:          # PEP 695 generic syntax: needs Python 3.12+
    S: set[str]  # alphabet
    K: set[STATE]  # set of states
    q0: STATE  # initial state
    d: dict[tuple[STATE, str], STATE]  # transition function
    F: set[STATE]  # final (accepting) states

    def accept(self, word: str) -> bool:
        current_state = self.q0

        for symbol in word:
            if (current_state, symbol) not in self.d:
                return False
            current_state = self.d[(current_state, symbol)]

        return current_state in self.F

    def remap_states[OTHER_STATE](self, f: Callable[[STATE], 'OTHER_STATE']) -> 'DFA[OTHER_STATE]':
        new_K = {f(state) for state in self.K}
        new_q0 = f(self.q0)
        new_F = {f(state) for state in self.F}
        new_d = {}

        for (state, symbol), next_state in self.d.items():
            new_state = f(state)
            new_next_state = f(next_state)
            new_d[(new_state, symbol)] = new_next_state

        return DFA(
            S=self.S,
            K=new_K,
            q0=new_q0,
            d=new_d,
            F=new_F
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _find_reachable_states(self) -> set[STATE]:
        """BFS from q0 — returns all reachable states."""
        reachable = set()
        queue = deque([self.q0])

        while queue:
            state = queue.popleft()
            if state not in reachable:
                reachable.add(state)
                for symbol in self.S:
                    if (state, symbol) in self.d:
                        next_state = self.d[(state, symbol)]
                        if next_state not in reachable:
                            queue.append(next_state)

        return reachable

    def complete(self) -> 'DFA[STATE]':
        """Adds a sink state if the DFA is partial."""
        sink = '__SINK__'
        new_d = dict(self.d)
        needs_sink = False
        states = set(self.K) | {self.q0}          # FIX: q0 counts even if K is empty

        for s in states:
            for c in self.S:
                if (s, c) not in new_d:
                    new_d[(s, c)] = sink
                    needs_sink = True

        if not needs_sink:
            return self

        new_K = states | {sink}
        for c in self.S:
            new_d[(sink, c)] = sink

        return DFA(
            S=self.S,
            K=new_K,
            q0=self.q0,
            d=new_d,
            F=self.F,
        )

    def canonical(self) -> 'DFA[int]':
        """Renames states BFS-style. Two MINIMAL DFAs for the same language become identical
        (this does not minimize: two equivalent non-minimal DFAs can still differ)."""
        # FIX: the empty language has one canonical form, whatever shape the input has
        if not (self._find_reachable_states() & self.F):
            return DFA(S=set(self.S), K={0}, q0=0, d={(0, c): 0 for c in self.S}, F=set())

        dfa = self.complete()

        seen = {dfa.q0: 0}
        queue = deque([dfa.q0])

        while queue:
            s = queue.popleft()
            for c in sorted(dfa.S):
                if (s, c) in dfa.d:
                    t = dfa.d[(s, c)]
                    if t not in seen:
                        seen[t] = len(seen)
                        queue.append(t)

        new_d = {}
        for (s, c), t in dfa.d.items():
            if s in seen and t in seen:
                new_d[(seen[s], c)] = seen[t]

        new_F = {seen[s] for s in dfa.F if s in seen}

        return DFA(
            S=set(dfa.S),
            K=set(range(len(seen))),
            q0=0,
            d=new_d,
            F=new_F,
        )

    def numbered(self) -> 'DFA[int]':
        """Same DFA with states renamed 0, 1, 2, ... (cheap to hash, unlike nested frozensets)."""
        ids = {s: i for i, s in enumerate(self.K)}
        return self.remap_states(lambda s: ids[s])

    def _quotient(self, block_of: Dict[STATE, int]) -> 'DFA[int]':
        """Merge every block of the final partition into one state.
        Assumes block ids are 0..m-1."""
        return DFA(
            S=self.S,
            K=set(block_of.values()),
            q0=block_of[self.q0],
            d={(block_of[s], c): block_of[t] for (s, c), t in self.d.items()},
            F={block_of[s] for s in self.F},
        )

    # ------------------------------------------------------------------
    # Hopcroft's algorithm
    # ------------------------------------------------------------------

    def _hopcroft_minimize(self, ops: dict | None = None, trace: list | None = None) -> 'DFA[int]':
        """Hopcroft (1971), O(k n log n). The worklist holds (block, symbol) splitters;
        the predecessors of a splitter are found through an inverse transition map, and
        after a split only the smaller half is queued. The smaller-half rule is only
        correct on a COMPLETE DFA, so the DFA is completed first."""
        ops = ops if ops is not None else {}
        for key in ('splitters_processed', 'splits', 'inverse_edges_scanned'):
            ops.setdefault(key, 0)

        dfa = self.complete()
        symbols = sorted(dfa.S)

        # inverse[(t, c)] = all states s with d(s, c) = t
        inverse: Dict[tuple, List] = {}
        for (s, c), t in dfa.d.items():
            inverse.setdefault((t, c), []).append(s)

        blocks: List[set] = [set(b) for b in (dfa.F & dfa.K, dfa.K - dfa.F) if b]
        block_of: Dict[STATE, int] = {}
        for i, block in enumerate(blocks):
            for s in block:
                block_of[s] = i

        if trace is not None:         # partition snapshots for the visualizer
            trace.append({'blocks': [sorted(b, key=str) for b in blocks]})

        worklist: deque = deque()     # deque, not set: same order on every run (reproducible ops)
        in_worklist: set = set()
        if len(blocks) == 2:
            smaller = 0 if len(blocks[0]) <= len(blocks[1]) else 1
            for c in symbols:
                worklist.append((smaller, c))
                in_worklist.add((smaller, c))

        while worklist:
            b, c = worklist.popleft()
            in_worklist.discard((b, c))
            ops['splitters_processed'] += 1

            # X = states that reach block b on symbol c
            X = [p for q in blocks[b] for p in inverse.get((q, c), ())]
            ops['inverse_edges_scanned'] += len(X)

            touched: Dict[int, set] = {}
            for p in X:
                touched.setdefault(block_of[p], set()).add(p)

            for y, inside in touched.items():
                Y = blocks[y]
                if len(inside) == len(Y):
                    continue                     # Y lies entirely inside X: no split
                Y -= inside                      # Y keeps the part outside X,
                new = len(blocks)                # the part inside X becomes a new block
                blocks.append(inside)
                for p in inside:
                    block_of[p] = new
                ops['splits'] += 1
                if trace is not None:
                    trace.append({'splitter': sorted(blocks[b], key=str), 'symbol': c,
                                  'split': [sorted(Y, key=str), sorted(inside, key=str)],
                                  'blocks': [sorted(bl, key=str) for bl in blocks]})

                for a in symbols:
                    if (y, a) in in_worklist:
                        target = new             # old block already queued: queue the new one too
                    else:
                        target = new if len(inside) <= len(Y) else y   # smaller half only
                    if (target, a) not in in_worklist:
                        worklist.append((target, a))
                        in_worklist.add((target, a))

        return dfa._quotient(block_of)

    # ------------------------------------------------------------------
    # Moore's algorithm
    # ------------------------------------------------------------------

    def _moore_minimize(self, ops: dict | None = None, trace: list | None = None) -> 'DFA[int]':
        """Moore (1956), O(k n^2). Each round, a state's signature is (its block, the blocks
        of its successors); states with equal signatures stay together. Stops when a round
        creates no new block. The DFA is completed first so that an explicit dead state and
        a missing transition are treated as the same thing."""
        ops = ops if ops is not None else {}
        for key in ('rounds', 'transitions_scanned', 'splits'):
            ops.setdefault(key, 0)

        dfa = self.complete()
        symbols = sorted(dfa.S)          # FIX: sort once, not once per state per round

        block_of = {s: (1 if s in dfa.F else 0) for s in dfa.K}
        num_blocks = len(set(block_of.values()))

        def snapshot(assignment):     # partition as a list of blocks, for the visualizer
            groups: Dict[int, list] = {}
            for s, b in assignment.items():
                groups.setdefault(b, []).append(s)
            return [sorted(g, key=str) for _, g in sorted(groups.items())]

        if trace is not None:
            trace.append({'blocks': snapshot(block_of)})

        while True:
            ops['rounds'] += 1
            sig_to_id: Dict[tuple, int] = {}
            new_block_of: Dict[STATE, int] = {}
            for s in dfa.K:
                sig = (block_of[s],) + tuple(block_of[dfa.d[(s, c)]] for c in symbols)
                new_block_of[s] = sig_to_id.setdefault(sig, len(sig_to_id))
            ops['transitions_scanned'] += len(dfa.K) * len(symbols)
            if trace is not None:
                trace.append({'blocks': snapshot(new_block_of)})

            if len(sig_to_id) == num_blocks:
                ops['blocks_final'] = num_blocks
                return dfa._quotient(new_block_of)
            ops['splits'] += len(sig_to_id) - num_blocks
            block_of, num_blocks = new_block_of, len(sig_to_id)

    # ------------------------------------------------------------------
    # Brzozowski's algorithm
    # ------------------------------------------------------------------

    def reverse(self):
        """Returns the reversed NFA (with epsilon transitions).
        Uses a lazy import to avoid a circular dependency with NFA.py."""
        from .NFA import NFA, EPSILON

        new_start = '__REV_START__'

        new_d: dict = {}
        for (s, c), t in self.d.items():
            new_d.setdefault((t, c), set()).add(s)

        # Every accepting state becomes a source of an epsilon-transition
        # back to the original start state (which becomes the only accepting state).
        new_d[(new_start, EPSILON)] = set(self.F)

        return NFA(
            S=set(self.S),
            K=set(self.K) | {new_start},
            q0=new_start,
            d=new_d,
            F={self.q0},
        )

    def trim(self) -> 'DFA[int]':
        """Removes unreachable states and states that cannot reach an accepting state.
        States are renumbered 0, 1, 2, ... in breadth-first order from q0. (Sorting them by
        str() instead is extremely slow for Brzozowski, whose states are nested frozensets.)"""
        reachable = self._find_reachable_states()

        # co-reachable: walk the transitions backwards from the accepting states
        predecessors: dict = {}
        for (s, c), t in self.d.items():
            predecessors.setdefault(t, []).append(s)
        co_reach = set(self.F)
        stack = list(self.F)
        while stack:
            for p in predecessors.get(stack.pop(), ()):
                if p not in co_reach:
                    co_reach.add(p)
                    stack.append(p)

        keep = reachable & co_reach

        if not keep:
            return DFA(S=set(self.S), K=set(), q0=self.q0, d={}, F=set())

        # every kept state lies on a path q0 -> ... -> final, so BFS through kept states finds all
        symbols = sorted(self.S)
        new_id = {self.q0: 0}
        queue = deque([self.q0])
        while queue:
            s = queue.popleft()
            for c in symbols:
                t = self.d.get((s, c))
                if t in keep and t not in new_id:
                    new_id[t] = len(new_id)
                    queue.append(t)

        new_d = {}
        for (s, c), t in self.d.items():
            if s in new_id and t in new_id:
                new_d[(new_id[s], c)] = new_id[t]

        return DFA(
            S=set(self.S),
            K=set(range(len(new_id))),
            q0=0,
            d=new_d,
            F={new_id[s] for s in self.F if s in new_id},
        )

    def _brzozowski_minimize(self, ops: dict | None = None,
                             max_states: int | None = None) -> 'DFA[STATE]':
        """Brzozowski minimization via reverse-determinize-reverse-determinize.
        Note: each reversal adds one extra start state, so the reversal sizes are n + 1.
        max_states aborts with StateLimitExceeded if a determinization grows too large."""
        ops = ops if ops is not None else {}
        ops.setdefault('intermediate_sizes', [])

        r1 = self.reverse()
        ops['intermediate_sizes'].append(len(r1.K))

        d1 = r1.subset_construction(max_states).numbered()
        ops['intermediate_sizes'].append(len(d1.K))

        r2 = d1.reverse()
        ops['intermediate_sizes'].append(len(r2.K))

        d2 = r2.subset_construction(max_states)
        ops['intermediate_sizes'].append(len(d2.K))

        return d2.trim()

    # ------------------------------------------------------------------
    # Unified entry point
    # ------------------------------------------------------------------

    def minimize(self, algorithm: str = 'hopcroft', ops: dict | None = None,
                 max_states: int | None = None) -> 'DFA':
        """Minimize using the specified algorithm: 'hopcroft', 'moore', or 'brzozowski'.
        Moore and Hopcroft return a COMPLETE minimal DFA (may contain a sink state);
        Brzozowski returns a TRIMMED one (no sink). Compare results with .canonical(),
        and report state counts in one form, e.g. len(result.trim().K)."""
        if not self.K:
            return DFA(S=self.S, K=set(), q0=self.q0, d={}, F=set())

        # Restrict to reachable states first
        reachable = self._find_reachable_states()
        if not reachable:
            return DFA(S=self.S, K=set(), q0=self.q0, d={}, F=set())

        reachable_dfa = DFA(
            S=self.S,
            K=reachable,
            q0=self.q0,
            d={(s, a): ns for (s, a), ns in self.d.items()
               if s in reachable and ns in reachable},
            F=self.F & reachable,
        ).numbered()          # same integer-labelled input for every algorithm (fair timing)

        if algorithm == 'hopcroft':
            return reachable_dfa._hopcroft_minimize(ops)
        elif algorithm == 'moore':
            return reachable_dfa._moore_minimize(ops)
        elif algorithm == 'brzozowski':
            return reachable_dfa._brzozowski_minimize(ops, max_states)
        else:
            raise ValueError(f"Unknown minimization algorithm: {algorithm}")
