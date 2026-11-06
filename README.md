# Regex to Minimal DFA Visualizer

This repository contains the source code and experimental benchmark for the paper:

**"Minimizing Automata Obtained from Regular Expressions: An Experimental Comparison of the Algorithms of Moore, Hopcroft and Brzozowski with State Complexity Analysis"**

It converts regular expressions through the standard pipeline and measures how much time, work and intermediate state the three classic DFA minimization algorithms need to reach the (unique) minimal DFA.

## Setup

Requires Python 3.12+. Graphviz (https://graphviz.org) is optional, for pictures.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

## Setup

Requires **Python 3.12+**. Graphviz (https://graphviz.org) is optional, for pictures.

    python -m venv .venv
    source .venv/bin/activate          # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

## Usage

    python -m pytest -q                              # 1. correctness tests (must all pass)
    python main.py "(a|b)*abb" --draw drawings       # 2. one regex, every stage, pictures
    python bench.py                                  # 3. experiment  -> results.csv
    python plots.py                                  # 4. figures     -> figures/

For the final thesis run use more and larger inputs, e.g.

    python bench.py --random 400 --max-leaves 150 --repeats 9 --out results_final.csv
    python plots.py results_final.csv

## Files

| File | Purpose |
|---|---|
| `automata/regex.py` | parser (regex → syntax tree) and Thompson's construction (→ ε-NFA) |
| `automata/NFA.py` | ε-NFA, subset construction (→ DFA), reversal, Brzozowski from an NFA |
| `automata/DFA.py` | DFA, Moore, Hopcroft, Brzozowski, trimming, completion, canonical form |
| `main.py` | runs the pipeline on one regex; prints sizes, timings, operation counts; draws stages |
| `bench.py` | generates inputs, measures all algorithms, writes `results.csv` |
| `plots.py` | turns `results.csv` into the thesis figures and tables |
| `tests/test_pipeline.py` | correctness tests against Python's `re` and known theoretical sizes |

## Regex syntax

`ab` concatenation · `a|b` union · `a*` star · `a+` one or more · `a?` optional ·
`( )` grouping · `ε` or `()` empty word · `\*` a literal special character.

## Measurement conventions

* Subset construction labels DFA states by *important* NFA states only
  (Aho, Lam, Sethi & Ullman 2006, §3.9.1) and always includes the trap state.
* Before minimizing, every algorithm receives the same reachable, integer-labelled DFA.
* Reported DFA sizes are **complete** (trap state included); `min_states_trim` excludes it.
* Brzozowski blow-up = largest determinized intermediate / final DFA of Brzozowski.
* Timings: median of `--repeats` runs with the garbage collector disabled.
* Random inputs use a fixed seed (`--seed`), so every run of `bench.py` uses the same regexes.
