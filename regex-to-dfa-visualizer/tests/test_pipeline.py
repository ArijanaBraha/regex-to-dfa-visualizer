"""
Correctness tests - run with:   python -m pytest -q

This is the evidence for 'the implementation is correct' in the thesis:
  * every stage accepts exactly the words that Python's own regex engine accepts,
  * all minimizers produce the same canonical minimal DFA,
  * known sizes from theory are reproduced exactly.
"""
import itertools
import os
import random
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from automata import DFA, RegexSyntaxError, parse, regex_to_nfa  # noqa: E402
from automata.regex import Concat, Epsilon, Star, Symbol, Union  # noqa: E402
from bench import nth_from_end, nth_from_start, random_regex  # noqa: E402

ALGORITHMS = ("moore", "hopcroft", "brzozowski")


def words(alphabet, max_len):
    for n in range(max_len + 1):
        for w in itertools.product(sorted(alphabet), repeat=n):
            yield "".join(w)


def all_minimal_dfas(regex, alphabet=None):
    nfa = regex_to_nfa(regex, alphabet)
    dfa = nfa.subset_construction()
    results = {a: dfa.minimize(a) for a in ALGORITHMS}
    results["brzozowski_nfa"] = nfa.brzozowski_minimize()
    return nfa, dfa, results


# ------------------------------------------------------------------ parser
def test_parser_precedence():
    # star binds tighter than concatenation, concatenation tighter than union
    assert parse("ab*|c") == Union(Concat(Symbol("a"), Star(Symbol("b"))), Symbol("c"))


def test_parser_sugar_and_escapes():
    assert parse("a+") == Concat(Symbol("a"), Star(Symbol("a")))
    assert parse("a?") == Union(Symbol("a"), Epsilon())
    assert parse(r"\*") == Symbol("*")
    assert parse("") == Epsilon() and parse("()") == Epsilon() and parse("ε") == Epsilon()


@pytest.mark.parametrize("bad", ["(a", "a)", "*a", "a|*", "\\", "(*)"])
def test_syntax_errors(bad):
    with pytest.raises(RegexSyntaxError):
        parse(bad)


# ------------------------------------------------------------------ whole pipeline
@pytest.mark.parametrize("seed", range(300))
def test_random_regex_against_python_re(seed):
    rng = random.Random(seed)
    alphabet = "abc"[: rng.randint(1, 3)]
    rx = random_regex(rng, rng.randint(1, 12), alphabet)
    nfa, dfa, results = all_minimal_dfas(rx, set(alphabet))

    canon = {name: m.canonical() for name, m in results.items()}
    assert len({repr(sorted(c.d.items())) + repr(sorted(c.F)) for c in canon.values()}) == 1, rx

    for w in words(alphabet, 6):
        expected = re.fullmatch(rx, w) is not None
        assert dfa.accept(w) == expected, (rx, w)
        for name, m in results.items():
            assert m.accept(w) == expected, (name, rx, w)


def test_textbook_example():
    # (a|b)*abb: the minimal DFA has 4 states (Aho, Lam, Sethi & Ullman 2006, ch. 3)
    _, _, results = all_minimal_dfas("(a|b)*abb")
    assert all(len(m.trim().K) == 4 for m in results.values())


@pytest.mark.parametrize("n", range(1, 8))
def test_family_sizes_match_theory(n):
    _, _, r1 = all_minimal_dfas(nth_from_end(n), {"a", "b"})
    assert all(len(m.trim().K) == 2 ** n for m in r1.values())
    _, _, r2 = all_minimal_dfas(nth_from_start(n), {"a", "b"})
    assert all(len(m.trim().K) == n + 1 for m in r2.values())    # n+2 with the sink state


def test_brzozowski_blowup_is_recorded():
    nfa = regex_to_nfa(nth_from_start(8), {"a", "b"})
    ops = {}
    nfa.subset_construction().minimize("brzozowski", ops)
    assert ops["intermediate_sizes"][1] == 2 ** 8                 # first determinization


def test_extra_alphabet_symbols():
    # 'c' is in the alphabet but not in the regex: it must lead to rejection
    _, _, results = all_minimal_dfas("a*", {"a", "c"})
    for m in results.values():
        assert m.accept("aa") and not m.accept("ac")


def test_dead_state_and_missing_transition_are_merged():
    dfa = DFA(S={"a", "b", "c"}, K={0, 1, 2, 3}, q0=0,
              d={(0, "a"): 1, (0, "b"): 2, (0, "c"): 3,
                 (1, "a"): 1, (1, "b"): 1, (1, "c"): 1}, F={3})
    for a in ALGORITHMS:
        assert len(dfa.minimize(a).trim().K) == 2


def test_empty_language_canonical_form():
    empty_trimmed = DFA(S={"a"}, K=set(), q0=0, d={}, F=set())
    empty_full = DFA(S={"a"}, K={0}, q0=0, d={(0, "a"): 0}, F=set())
    assert empty_trimmed.canonical() == empty_full.canonical()
