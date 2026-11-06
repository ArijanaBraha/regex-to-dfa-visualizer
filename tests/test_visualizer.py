"""Tests for the visualizer backend (app.py): the data it sends to the page must agree with the engine."""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app  # noqa: E402
from automata import RegexSyntaxError  # noqa: E402
from bench import random_regex  # noqa: E402


@pytest.mark.parametrize("seed", range(40))
def test_visualizer_data_is_consistent(seed):
    rng = random.Random(seed)
    alphabet = "abc"[: rng.randint(1, 3)]
    rx = random_regex(rng, rng.randint(1, 10), alphabet)
    data = app.run(rx, alphabet)

    sizes = {c['states'] for c in data['comparison']}
    assert len(sizes) == 1 and data['all_equal'], rx
    minimal_size = sizes.pop()
    # the partitions animated for Moore and Hopcroft end in the minimal automaton
    for algo in ('moore', 'hopcroft'):
        assert data['algorithms'][algo]['minimal']['n_states'] == minimal_size, (algo, rx)
        last = data['algorithms'][algo]['steps'][-1]['blocks']
        assert sorted(s for b in last for s in b) == sorted(s['id'] for s in data['dfa']['states'])
    # the subset-construction steps rebuild exactly the DFA
    discovered = {data['dfa']['start']} | {s['to'] for s in data['subset_steps'] if s['new']}
    assert discovered == {s['id'] for s in data['dfa']['states']}
    assert len(data['subset_steps']) == data['dfa']['n_states'] * len(data['alphabet'])
    # Brzozowski's last stage is the minimal (complete) DFA
    assert data['algorithms']['brzozowski']['stages'][-1]['automaton']['n_states'] == minimal_size, rx


def test_syntax_error_is_reported():
    with pytest.raises(RegexSyntaxError):
        app.run('(a|b', None)
