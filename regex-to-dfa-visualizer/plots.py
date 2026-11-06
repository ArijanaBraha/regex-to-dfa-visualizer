"""
plots.py - turns results.csv into the four thesis figures (PDF for LaTeX, PNG for preview)
plus summary tables.
 
    python plots.py                      # reads results.csv, writes into figures/
    python plots.py results_final.csv    # another results file
 
Counting conventions (state them in the methodology chapter):
  * DFA sizes before and after minimization are COMPLETE DFAs (trap/sink state included).
  * Brzozowski blow-up = largest determinized intermediate / Brzozowski's final DFA,
    both straight out of the subset construction (trap state included in both).
"""
import sys
import os
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
ALGOS = ["moore", "hopcroft", "brzozowski", "brzozowski_nfa"]
LABEL = {"moore": "Moore", "hopcroft": "Hopcroft", "brzozowski": "Brzozowski (from DFA)",
         "brzozowski_nfa": "Brzozowski (from NFA)"}
STYLE = {"moore": ("tab:blue", "o"), "hopcroft": ("tab:green", "s"),
         "brzozowski": ("tab:red", "^"), "brzozowski_nfa": ("tab:orange", "v")}
BINS = [0, 10, 100, 1000, np.inf]
BIN_LABELS = ["< 10", "10-99", "100-999", "≥ 1000"]
 
os.makedirs("figures", exist_ok=True)
df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results.csv")
ok = df[df.status == "ok"].copy()
ok["time_ms"] = ok.time_ns / 1e6
assert ok.correct.all(), "some minimizer produced a wrong result - check before plotting!"
 
 
def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/{name}.{ext}", dpi=200)
    plt.close(fig)
 
 
# ---- Figure 1: time vs DFA size (RQ1) -------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for a in ALGOS:
    sub = ok[ok.algorithm == a]
    color, marker = STYLE[a]
    ax.scatter(sub.dfa_states, sub.time_ms, s=12, alpha=0.35, color=color, marker=marker)
    # median per logarithmic size bucket, drawn as a line
    rnd = sub[sub.family == "random"]          # lines: random regexes only (families would
    buckets = np.floor(np.log2(rnd.dfa_states))  # dominate some size buckets and bend the line)
    med = rnd.groupby(buckets).agg(x=("dfa_states", "median"), y=("time_ms", "median"),
                                   n=("time_ms", "size"))
    med = med[med.n >= 5]       # skip buckets with too few inputs for a stable median
    ax.plot(med.x, med.y, color=color, marker=marker, label=LABEL[a])
ax.set(xscale="log", yscale="log", xlabel="states of the DFA entering minimization",
       ylabel="running time [ms]",
       title="Running time vs. DFA size (lines: medians over random regexes)")
ax.legend(fontsize=8)
ax.grid(True, which="major", alpha=0.3)
save(fig, "fig1_time_vs_size")
 
# ---- Figure 2: state and transition reduction ------------------------------------------
one = ok[ok.algorithm == "hopcroft"]           # all algorithms give the same minimal DFA
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, before, after, what in [(axes[0], "dfa_states", "min_states", "states"),
                                (axes[1], "dfa_transitions", "min_transitions", "transitions")]:
    for fam, marker in [("random", "o"), ("nth_from_end", "s"), ("nth_from_start", "^")]:
        s = one[one.family == fam]
        ax.scatter(s[before], s[after], s=14, marker=marker, alpha=0.6, label=fam)
    lim = [1, one[before].max() * 1.5]
    ax.plot(lim, lim, "k--", lw=0.8, label="no reduction")
    ax.set(xscale="log", yscale="log", xlabel=f"{what} before minimization",
           ylabel=f"{what} after minimization", title=f"{what.capitalize()} reduction")
    ax.grid(True, alpha=0.3)
axes[0].legend(fontsize=8)
save(fig, "fig2_reduction")
 
# ---- Figure 3: Brzozowski blow-up (RQ2) -------------------------------------------------
brz = ok[ok.algorithm.isin(["brzozowski", "brzozowski_nfa"])].copy()
# both counted the same way (straight from the subset construction)
brz["blowup"] = brz.peak_intermediate_states / brz.det2_states
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for fam, marker in [("nth_from_end", "s"), ("nth_from_start", "^")]:
    s = brz[(brz.family == fam) & (brz.algorithm == "brzozowski")].copy()
    n = np.arange(1, len(s) + 1)              # families are generated for n = 1, 2, 3, ...
    axes[0].plot(n, s.blowup.values, marker=marker, label=fam)
axes[0].set(yscale="log", xlabel="n", ylabel="peak intermediate / final states",
            title="Blow-up on the two families")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)
# Dividing by the final size inflates the ratio when the language is trivial (final DFA of
# 1 state), so for random regexes compare the first determinization with the forward DFA.
rnd = brz[brz.family == "random"]
data = [rnd[rnd.algorithm == a].det1_states / rnd[rnd.algorithm == a].dfa_states
        for a in ["brzozowski", "brzozowski_nfa"]]
axes[1].boxplot(data, tick_labels=["from DFA", "from NFA"])
axes[1].axhline(1, color="k", ls="--", lw=0.8)
axes[1].set(yscale="log", ylabel="reverse determinization / forward DFA (states)",
            title="Reverse vs. forward determinization (random)")
axes[1].grid(True, alpha=0.3)
save(fig, "fig3_brzozowski_blowup")
 
# ---- Figure 4: fastest algorithm per size class (RQ1/RQ3) ------------------------------
ok["size_class"] = pd.cut(ok.dfa_states, BINS, labels=BIN_LABELS, right=False)
fastest = ok.loc[ok.groupby("instance").time_ns.idxmin()]
counts = (fastest.groupby(["size_class", "algorithm"], observed=False).size()
          .unstack(fill_value=0).reindex(columns=ALGOS, fill_value=0))
share = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
fig, ax = plt.subplots(figsize=(7.5, 4.6))
bottom = np.zeros(len(share))
for a in ALGOS:
    vals = share[a].fillna(0).values
    ax.bar(BIN_LABELS, vals, bottom=bottom, color=STYLE[a][0], label=LABEL[a])
    bottom += vals
for i, total in enumerate(counts.sum(axis=1)):
    ax.text(i, 101, f"n={total}", ha="center", fontsize=8)
ax.set(ylim=(0, 110), ylabel="% of inputs where fastest", xlabel="DFA size class (states)",
       title="Fastest minimizer per size class")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4)
save(fig, "fig4_winner_per_size")
 
# ---- Summary table for the results chapter ---------------------------------------------
table = (ok.groupby(["size_class", "algorithm"], observed=True).time_ms.median()
         .unstack().reindex(columns=ALGOS).round(3))
table.to_csv("figures/table_median_time_ms.csv")
with open("figures/table_median_time_ms.tex", "w", encoding="utf-8") as fh:
    fh.write(table.to_latex(na_rep="--"))
print(table)
print("figures written to figures/")
 
# ---- RQ4: whole-pipeline comparison --------------------------------------------------
# Brzozowski on the NFA replaces BOTH subset construction and minimization,
# so compare it with (subset construction + Hopcroft), not with Hopcroft alone.
wide = ok.pivot_table(index="instance", columns="algorithm", values="time_ns")
subset = ok.groupby("instance").subset_ns.first()
pipeline = pd.DataFrame({"subset_plus_hopcroft": subset + wide["hopcroft"],
                         "brzozowski_on_nfa": wide["brzozowski_nfa"]}).dropna()
cheaper = (pipeline.brzozowski_on_nfa < pipeline.subset_plus_hopcroft).mean() * 100
speedup = (pipeline.subset_plus_hopcroft / pipeline.brzozowski_on_nfa).median()
print(f"RQ4: Brzozowski-on-NFA is cheaper than subset+Hopcroft on {cheaper:.1f}% of inputs "
      f"(median ratio subset+Hopcroft / Brzozowski-on-NFA = {speedup:.2f})")
pipeline.to_csv("figures/table_rq4_pipeline_ns.csv")
 
# ---- Generator check: trivial languages ---------------------------------------------
r = ok[(ok.algorithm == "hopcroft") & (ok.family == "random")]
print(f"random regexes whose minimal DFA has 1 state (language = all words or none): "
      f"{(r.min_states == 1).mean() * 100:.1f}% of {len(r)}")
 