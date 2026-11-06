"""
main.py - run the whole pipeline on ONE regular expression and show every stage.

    python main.py "(a|b)*abb"
    python main.py "(a|b)*abb" --draw drawings        # also write Graphviz pictures
    python main.py "a(b|c)*" --alphabet abc --test abcb acc b

With --draw, every stage is written as a .dot file; if Graphviz is installed
(https://graphviz.org, command `dot`), PNG pictures are rendered as well.
"""
import argparse
import os
import shutil
import subprocess
import time

from automata import ALGORITHMS, DFA, NFA, EPSILON, RegexSyntaxError, parse, thompson
from automata.regex import to_string


# ------------------------------------------------------------------
# Graphviz export
# ------------------------------------------------------------------

def _label(state) -> str:
    if isinstance(state, frozenset):
        if not state:
            return "∅"
        try:
            members = sorted(state)              # numeric order: {4,6,8,10}
        except TypeError:
            members = sorted(state, key=str)     # mixed types: fall back to text order
        return "{" + ",".join(map(str, members)) + "}"
    return str(state)


def to_dot(aut: NFA | DFA, title: str) -> str:
    """Graphviz DOT text for an NFA or DFA. Parallel edges are merged into one labelled edge."""
    ids = {s: i for i, s in enumerate(sorted(aut.K, key=lambda s: (len(_label(s)), _label(s))))}
    lines = ["digraph A {", "  rankdir=LR;", f'  label="{title}"; labelloc=t;',
             '  node [shape=circle, fontsize=11];', '  __start [shape=point];']
    for s, i in ids.items():
        shape = "doublecircle" if s in aut.F else "circle"
        lines.append(f'  q{i} [label="{_label(s)}", shape={shape}];')
    lines.append(f"  __start -> q{ids[aut.q0]};")
    edges: dict[tuple, list] = {}
    for (s, c), target in aut.d.items():
        for t in (target if isinstance(aut, NFA) else [target]):
            if s in ids and t in ids:
                edges.setdefault((ids[s], ids[t]), []).append("ε" if c == EPSILON else c)
    for (i, j), symbols in sorted(edges.items()):
        lines.append(f'  q{i} -> q{j} [label="{",".join(sorted(symbols))}"];')
    lines.append("}")
    return "\n".join(lines)


def draw(aut, name: str, title: str, folder: str) -> None:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{name}.dot")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(to_dot(aut, title))
    if shutil.which("dot"):
        subprocess.run(["dot", "-Tpng", path, "-o", path[:-4] + ".png"], check=True)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="regex -> NFA -> DFA -> minimal DFA")
    p.add_argument("regex")
    p.add_argument("--alphabet", help="alphabet as a string, e.g. abc (default: symbols in the regex)")
    p.add_argument("--draw", metavar="FOLDER", help="write Graphviz drawings of every stage")
    p.add_argument("--test", nargs="*", default=[], metavar="WORD", help="words to test for membership")
    args = p.parse_args()

    try:
        tree = parse(args.regex)
    except RegexSyntaxError as e:
        raise SystemExit(f"syntax error: {e}")

    nfa = thompson(tree, set(args.alphabet) if args.alphabet else None)
    dfa = nfa.subset_construction()

    print(f"regex          {args.regex}")
    print(f"parsed as      {to_string(tree)}")
    print(f"alphabet       {{{', '.join(sorted(nfa.S))}}}")
    print(f"NFA (Thompson) {len(nfa.K):6d} states {sum(map(len, nfa.d.values())):6d} transitions")
    print(f"DFA (subsets)  {len(dfa.K):6d} states {len(dfa.d):6d} transitions")
    print()
    print(f"{'algorithm':16s} {'states':>7s} {'trimmed':>8s} {'transitions':>12s} {'time [ms]':>10s}  operations")

    results = {}
    runs = [(a, lambda a=a, ops=None: dfa.minimize(a, ops)) for a in ALGORITHMS]
    runs.append(("brzozowski_nfa", lambda ops=None: nfa.brzozowski_minimize(ops)))
    for name, run in runs:
        ops = {}
        t0 = time.perf_counter()
        minimal = run(ops=ops)
        ms = (time.perf_counter() - t0) * 1000
        results[name] = minimal
        canon = minimal.canonical()
        print(f"{name:16s} {len(canon.K):7d} {len(minimal.trim().K):8d} {len(canon.d):12d} "
              f"{ms:10.3f}  {ops}")

    agree = len({repr(sorted(m.canonical().d.items())) + repr(sorted(m.canonical().F))
                 for m in results.values()}) == 1
    print(f"\nall algorithms produce the same minimal DFA: {agree}")

    for w in args.test:
        print(f"accepts {w!r:12s} -> {results['hopcroft'].accept(w)}")

    if args.draw:
        minimal = results["hopcroft"]
        draw(nfa, "1_nfa", f"Thompson NFA for {args.regex}", args.draw)
        draw(dfa, "2_dfa", "DFA after subset construction", args.draw)
        draw(minimal.trim(), "3_minimal_dfa", "Minimal DFA (trimmed)", args.draw)
        made = "PNG + DOT" if shutil.which("dot") else "DOT (install Graphviz to get PNG)"
        print(f"\ndrawings written to {args.draw}/ as {made}")


if __name__ == "__main__":
    main()
