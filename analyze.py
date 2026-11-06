"""
analyze.py - the numbers behind the figures, for writing the results chapter.
 
    python analyze.py results_final.csv > analysis.txt
 
Prints, per research question, the statistics the summary table in plots.py hides:
sample sizes, win rates, Moore's number of rounds, Brzozowski's blow-up, correlations,
and what distinguishes the inputs on which Brzozowski-on-NFA beats subset+Hopcroft.
"""
import sys
 
import numpy as np
import pandas as pd
 
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)
 
df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results.csv")
print("=" * 70, "\n0. DATA")
print("rows per status:", df.status.value_counts().to_dict())
ok = df[df.status == "ok"].copy()
print("all results verified correct:", bool(ok.correct.all()))
ok["size_class"] = pd.cut(ok.dfa_states, [0, 10, 100, 1000, np.inf],
                          labels=["<10", "10-99", "100-999", ">=1000"], right=False)
inst = ok.drop_duplicates("instance")
print("\ninputs per size class and family:")
print(pd.crosstab(inst.size_class, inst.family, margins=True))
 
# ------------------------------------------------------------------ RQ1
print("=" * 70, "\nRQ1. WHICH ALGORITHM IS FASTEST?")
wide = ok.pivot_table(index="instance", columns="algorithm", values="time_ns")
wide = wide.join(inst.set_index("instance")[["size_class", "family", "dfa_states"]])
wide["hopcroft_faster_than_moore"] = wide.hopcroft < wide.moore
wide["moore_over_hopcroft"] = wide.moore / wide.hopcroft
wide["brz_over_hopcroft"] = wide.brzozowski / wide.hopcroft
print(wide.groupby("size_class", observed=True).agg(
    n=("moore", "size"),
    hopcroft_wins_vs_moore=("hopcroft_faster_than_moore", "mean"),
    median_moore_over_hopcroft=("moore_over_hopcroft", "median"),
    median_brz_over_hopcroft=("brz_over_hopcroft", "median")).round(2))
 
moore = ok[ok.algorithm == "moore"]
print("\nMoore: number of refinement rounds (worst case would be about n):")
print(moore.groupby("size_class", observed=True).agg(
    median_states=("dfa_states", "median"), median_rounds=("moore_rounds", "median"),
    max_rounds=("moore_rounds", "max")))
 
# ------------------------------------------------------------------ RQ2
print("=" * 70, "\nRQ2. HOW LARGE DO BRZOZOWSKI'S INTERMEDIATE AUTOMATA GET?")
brz = ok[ok.algorithm.isin(["brzozowski", "brzozowski_nfa"])].copy()
brz["blowup"] = brz.peak_intermediate_states / brz.det2_states
brz["det1_over_input"] = brz.det1_states / brz.dfa_states
print(brz.groupby(["algorithm", "family"]).blowup.describe(
    percentiles=[.5, .9]).round(2)[["count", "50%", "90%", "max"]])
print("\nlargest blow-ups:")
print(brz.nlargest(5, "blowup")[["algorithm", "family", "dfa_states", "det1_states",
                                 "det2_states", "blowup", "regex"]].to_string(index=False))
 
# ------------------------------------------------------------------ RQ3
print("=" * 70, "\nRQ3. WHAT MAKES EACH ALGORITHM SLOW? (Spearman rank correlation with time)")
features = ["regex_length", "alphabet_size", "nfa_states", "dfa_states", "dfa_transitions"]
rows = {}
for a in ["moore", "hopcroft", "brzozowski", "brzozowski_nfa"]:
    sub = ok[ok.algorithm == a]
    cols = features + (["peak_intermediate_states"] if a.startswith("brz") else [])
    rows[a] = sub[cols + ["time_ns"]].corr(method="spearman")["time_ns"].drop("time_ns")
print(pd.DataFrame(rows).round(2))
 
# ------------------------------------------------------------------ RQ4
print("=" * 70, "\nRQ4. WHEN IS BRZOZOWSKI-ON-NFA CHEAPER THAN SUBSET + HOPCROFT?")
pipe = wide.join(inst.set_index("instance")[["subset_ns", "nfa_states", "regex_length"]])
pipe = pipe.join(ok[ok.algorithm == "brzozowski_nfa"].set_index("instance")[["det1_states"]])
pipe["nfa_brz_wins"] = pipe.brzozowski_nfa < pipe.subset_ns + pipe.hopcroft
pipe["reverse_dfa_over_dfa"] = pipe.det1_states / pipe.dfa_states
print("win rate per size class:")
print(pipe.groupby("size_class", observed=True).nfa_brz_wins.agg(["size", "mean"]).round(2))
print("\nmedian characteristics, wins vs. losses:")
print(pipe.groupby("nfa_brz_wins")[["regex_length", "nfa_states", "dfa_states", "det1_states",
                                    "reverse_dfa_over_dfa"]].median().round(2))
 