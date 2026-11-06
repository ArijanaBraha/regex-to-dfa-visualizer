from .DFA import DFA

from dataclasses import dataclass
from collections.abc import Callable
from collections import deque

EPSILON = ''  # representation of epsilon transitions


class StateLimitExceeded(RuntimeError):
    """Raised when a subset construction creates more DFA states than allowed."""


@dataclass
class NFA[STATE]:  # generic NFA definition (PEP 695 syntax: Python 3.12+)
    S: set[str]  # alphabet
    K: set[STATE]  # set of states
    q0: STATE  # initial state
    d: dict[tuple[STATE, str], set[STATE]]  # transition function
    F: set[STATE]  # accepting states

    # compute all states reachable through epsilon transitions from a given state
    def epsilon_closure(self, state: STATE) -> set[STATE]:
        closure = set()
        stack = [state]

        while stack:
            current = stack.pop()
            if current not in closure:
                closure.add(current)

                if (current, EPSILON) in self.d:
                    for next_state in self.d[(current, EPSILON)]:
                        if next_state not in closure:
                            stack.append(next_state)

        return closure

    # convert the NFA into an equivalent DFA using the subset construction algorithm
    def subset_construction(self, max_states: int | None = None,
                            trace: list | None = None) -> DFA[frozenset[STATE]]:
        """Rabin & Scott subset construction with two refinements:

        1. Epsilon-closures are computed once per NFA state and cached.
        2. Each DFA state is labelled only by its IMPORTANT NFA states (Aho, Lam, Sethi &
           Ullman 2006, sec. 3.9.1): states that are accepting or have at least one
           non-epsilon transition. Other states never contribute a move or acceptance,
           so two subsets that differ only in such states behave identically and must
           be the same DFA state. Without this, the artificial start state added by
           DFA.reverse() makes Brzozowski's algorithm return a non-minimal DFA.
        """
        TRAP_STATE = frozenset()
        dfa_alphabet = self.S - {EPSILON}

        important = {s for s in self.K
                     if s in self.F or any((s, c) in self.d for c in dfa_alphabet)}

        closure_cache: dict = {}

        def closure_of(state):
            if state not in closure_cache:
                closure_cache[state] = frozenset(self.epsilon_closure(state)) & important
            return closure_cache[state]

        def close(states) -> frozenset:
            out = set()
            for s in states:
                out |= closure_of(s)
            return frozenset(out)

        initial_state = close([self.q0])

        dfa_states = {initial_state}
        unprocessed_states = deque([initial_state])
        dfa_transitions = {}
        dfa_accepting = set()

        while unprocessed_states:
            current = unprocessed_states.popleft()

            if any(s in self.F for s in current):
                dfa_accepting.add(current)

            for symbol in sorted(dfa_alphabet):      # fixed order: reproducible numbering
                next_states = set()
                for s in current:
                    next_states.update(self.d.get((s, symbol), ()))

                target = close(next_states)          # empty set = TRAP_STATE
                dfa_transitions[(current, symbol)] = target

                is_new = target not in dfa_states
                if is_new:
                    if max_states is not None and len(dfa_states) >= max_states:
                        raise StateLimitExceeded(f"more than {max_states} DFA states")
                    dfa_states.add(target)
                    unprocessed_states.append(target)
                if trace is not None:   # (subset, symbol, target, is_new) for the visualizer
                    trace.append((current, symbol, target, is_new))

        return DFA(
            S=dfa_alphabet,
            K=dfa_states,
            q0=initial_state,
            d=dfa_transitions,
            F=dfa_accepting
        )

    # reverse the NFA: flip every transition, new start state with epsilon-edges to the old finals
    def reverse(self) -> 'NFA':
        new_start = '__REV_START__'
        new_d: dict = {}
        for (s, c), targets in self.d.items():
            for t in targets:
                new_d.setdefault((t, c), set()).add(s)
        new_d.setdefault((new_start, EPSILON), set()).update(self.F)

        return NFA(
            S=set(self.S),
            K=set(self.K) | {new_start},
            q0=new_start,
            d=new_d,
            F={self.q0},
        )

    # Brzozowski's algorithm applied directly to the NFA (skips the first subset construction)
    def brzozowski_minimize(self, ops: dict | None = None,
                            max_states: int | None = None) -> DFA:
        """minimal DFA = det(rev(det(rev(N)))). The first determinization already yields a
        DFA, so this works for any NFA, not only for DFAs (Brzozowski 1962)."""
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

    # create a new NFA where states are transformed using function f
    def remap_states[OTHER_STATE](self, f: 'Callable[[STATE], OTHER_STATE]') -> 'NFA[OTHER_STATE]':
        new_K = {f(state) for state in self.K}
        new_q0 = f(self.q0)
        new_F = {f(state) for state in self.F}
        new_d = {}

        for (state, symbol), next_states in self.d.items():
            new_state = f(state)
            new_next_states = {f(next_state) for next_state in next_states}
            new_d[(new_state, symbol)] = new_next_states

        return NFA(
            S=self.S,
            K=new_K,
            q0=new_q0,
            d=new_d,
            F=new_F
        )
