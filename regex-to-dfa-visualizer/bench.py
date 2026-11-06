"""
bench.py - the experiment. Runs every minimizer on every input regex and writes
results.csv with one row per (regex, algorithm).

    python bench.py                          # 100 random regexes + the two families
    python bench.py --random 400 --max-leaves 150 --repeats 9 --out results_final.csv

Timing protocol (Python docs on `timeit`; Georges et al. 2007): each measurement is repeated,
the garbage collector is disabled during a run, and the median and minimum are kept.
Operation counts come from a separate run so they never slow down the timed runs.
"""
import argparse
import csv
import gc
import random
import statistics
import time

from automata import StateLimitExceeded, regex_to_nfa


# ------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------

def random_regex(rng: random.Random, leaves: int, alphabet: str,
                 p_star: float = 0.25, p_union: float = 0.4) -> str:
    """Random regex containing exactly `leaves` symbols (our size parameter)."""
    if leaves == 1:
        s = rng.choice(alphabet)
    else:
        left = rng.randint(1, leaves - 1)
        l = random_regex(rng, left, alphabet, p_star, p_union)
        r = random_regex(rng, leaves - left, alphabet, p_star, p_union)
        s = f"({l}|{r})" if rng.random() < p_union else l + r
    if rng.random() < p_star:
        s = f"({s})*"
    return s


def nth_from_end(n: int) -> str:
    """'the n-th symbol from the END is a': the minimal DFA has 2^n states."""
    return "(a|b)*a" + "(a|b)" * (n - 1)


def nth_from_start(n: int) -> str:
    """'the n-th symbol from the START is a': minimal DFA has n+2 states, but the reversed
    language needs 2^n, so Brzozowski's first determinization blows up."""
    return "(a|b)" * (n - 1) + "a(a|b)*"


def make_inputs(args):
    rng = random.Random(args.seed)          # fixed seed: the experiment is reproducible
    for _ in range(args.random):
        leaves = rng.randint(3, args.max_leaves)
        alphabet = "abcdefgh"[:rng.choice([2, 3, 4])]
        yield "random", random_regex(rng, leaves, alphabet), alphabet
    for n in range(1, args.max_n + 1):
        yield "nth_from_end", nth_from_end(n), "ab"
        yield "nth_from_start", nth_from_start(n), "ab"


# ------------------------------------------------------------------
# Measurement
# ------------------------------------------------------------------

def time_ns(fn, repeats: int) -> tuple[float, int]:
    samples = []
    for _ in range(repeats):
        gc.collect()
        gc.disable()
        t0 = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t0)
        gc.enable()
    return statistics.median(samples), min(samples)


def measure(instance: int, family: str, regex: str, alphabet: str, args) -> list[dict]:
    nfa = regex_to_nfa(regex, set(alphabet))
    try:
        dfa = nfa.subset_construction(args.max_states)
    except StateLimitExceeded:
        print(f"   skipped: subset construction exceeded {args.max_states} states")
        return []
    subset_ns, _ = time_ns(lambda: nfa.subset_construction(), args.repeats)
    reference = dfa.minimize("hopcroft").canonical()

    base = dict(instance=instance, family=family, regex=regex, regex_length=len(regex),
                alphabet_size=len(alphabet),
                nfa_states=len(nfa.K), nfa_transitions=sum(map(len, nfa.d.values())),
                dfa_states=len(dfa.K), dfa_transitions=len(dfa.d), subset_ns=subset_ns)

    runs = {
        "moore":          lambda ops=None: dfa.minimize("moore", ops),
        "hopcroft":       lambda ops=None: dfa.minimize("hopcroft", ops),
        "brzozowski":     lambda ops=None: dfa.minimize("brzozowski", ops, args.max_states),
        "brzozowski_nfa": lambda ops=None: nfa.brzozowski_minimize(ops, args.max_states),
    }
    rows = []
    for name, run in runs.items():
        row = dict(base, algorithm=name, status="ok")
        try:
            ops = {}
            minimal = run(ops)                           # instrumented run + correctness check
            canon = minimal.canonical()
            row["correct"] = canon == reference
            row.update(min_states=len(canon.K), min_transitions=len(canon.d),
                       min_states_trim=len(minimal.trim().K))
            sizes = ops.pop("intermediate_sizes", None)
            if sizes:                                    # Brzozowski: [rev1, det1, rev2, det2]
                row.update(rev1_states=sizes[0], det1_states=sizes[1],
                           rev2_states=sizes[2], det2_states=sizes[3],
                           peak_intermediate_states=max(sizes[1], sizes[3]))
            row.update({f"{name}_{k}": v for k, v in ops.items()})
            row["time_ns"], row["time_min_ns"] = time_ns(run, args.repeats)
        except StateLimitExceeded:
            row["status"] = "state_limit"
        rows.append(row)
    return rows


FIELDS = ["instance", "family", "regex", "regex_length", "alphabet_size",
          "nfa_states", "nfa_transitions", "dfa_states", "dfa_transitions", "subset_ns",
          "algorithm", "status", "correct", "min_states", "min_transitions", "min_states_trim",
          "time_ns", "time_min_ns",
          "moore_rounds", "moore_transitions_scanned", "moore_splits",
          "hopcroft_splitters_processed", "hopcroft_inverse_edges_scanned", "hopcroft_splits",
          "rev1_states", "det1_states", "rev2_states", "det2_states", "peak_intermediate_states"]


def main():
    p = argparse.ArgumentParser(description="benchmark the three minimization algorithms")
    p.add_argument("--random", type=int, default=100, help="number of random regexes")
    p.add_argument("--max-leaves", type=int, default=60, help="largest random regex (symbols)")
    p.add_argument("--max-n", type=int, default=14, help="largest n for the two families")
    p.add_argument("--repeats", type=int, default=7, help="timed runs per measurement")
    p.add_argument("--max-states", type=int, default=100_000, help="abort determinization above this")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default="results.csv")
    args = p.parse_args()

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for i, (family, regex, alphabet) in enumerate(make_inputs(args)):
            print(f"[{i:3d}] {family:15s} {regex[:60]}")
            for row in measure(i, family, regex, alphabet, args):
                writer.writerow(row)
            fh.flush()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
