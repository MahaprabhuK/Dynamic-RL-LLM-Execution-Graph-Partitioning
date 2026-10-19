"""
Checks every number in docs/report/final_report.html against its sources.

1. Strict table checks: tables generated from data are compared cell by cell with the value the
   source gives (v6 sweep, v7 latency, v7 tests, best fixed arm, segment oracle).
2. Every other number in the report text is extracted with its section. It must either
   - match a value from that section's source files (CSV cells, numbers in text/yaml summaries,
     aggregates computed here), after rounding to the decimals printed, or
   - be excluded by a named structural rule (identifiers, dates, headings, code, references).
   Percentages printed for fractions (0.123 -> 12.3%) and absolute values are accepted as matches.

Writes experiments/check_report_numbers.txt. Exit code 1 if any number is unmatched or any table cell
differs.

    python experiments/check_report_numbers.py [report.html]   (output file is written only for the default report)
"""

import html as htmllib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "docs" / "report" / "final_report.html"
OUT = ROOT / "experiments" / "check_report_numbers.txt"
DP = "data/processed/"
EX = "experiments/"

V7 = [EX + f for f in ["e4_summary.txt", "e4_supplement.txt", "e4_latency.csv", "e4_tests.csv",
                       "e4_tests_sensitivity.csv", "e4_latency_sensitivity.csv", "e4_segment_oracle.csv",
                       "e4_best_fixed.csv", "e4_by_date.csv", "e4_plan.yaml", "e3_dqn_v7.txt", "e3_rule.yaml",
                       "e2_frozen.yaml", "check_main_final.txt", "agent_progress.md", "e4_bandit_prior.yaml"]] + \
     [DP + "skipped_runs_v7.csv", "src/scheduler/adaptive.py"]
V5 = [DP + f for f in ["adaptive_report_v5r.txt", "adaptive_summary_v5r.csv", "adaptive_paired_diffs_v5r.csv",
                       "adaptive_paired_diffs_sensitivity_v5r.csv", "adaptive_summary_sensitivity_v5r.csv",
                       "adaptive_exclusions_v5r.csv", "adaptive_report_v5.txt", "adaptive_summary_v5.csv"]]
V4 = [DP + f for f in ["adaptive_report_v4.txt", "adaptive_summary_v4.csv", "adaptive_paired_diffs_v4.csv"]]
V3 = [DP + f for f in ["fixed_strategy_summary_v3.csv", "fixed_strategy_paired_diffs_v3.csv",
                       "fixed_strategy_summary_pilot.csv", "fixed_strategy_paired_diffs_pilot.csv",
                       "adaptive_summary_v3.csv", "adaptive_paired_diffs_v3.csv", "rl_offline_evaluation_v3.csv",
                       "rl_training_history_v3.csv"]]
V12 = [DP + f for f in ["fixed_strategy_summary.csv", "fixed_strategy_summary_v2.csv", "rl_policy_summary_v2.csv"]]
V6 = [DP + f for f in ["rl_offline_evaluation_v6.csv", "rl_training_history_v6.csv", "training_data_v6.csv"]]
CONFIG = ["configs/config.yaml", "src/scheduler/adaptive.py", "src/llm/partition.py"]
REFS = [EX + "reference_check.md"]

# h2 section number -> sources. "*" (abstract, contents) uses the v7 sources.
SECTIONS = {
    "Abstract": V7, "1": V7 + V5 + V4 + V3 + CONFIG + [DP + "load_sweep_v4.csv"], "2": CONFIG, "3": [],
    "4": CONFIG + V4 + V5 + V3, "5": V12, "6": V3, "7": V3, "8": V3, "9": V4, "10": [DP + "load_sweep_v4.csv"] + V5,
    "11": V5, "12": V5 + V4 + V3, "13": V5 + CONFIG, "14": V5 + V4 + V3 + CONFIG,
    "15": V6 + [DP + "load_sweep_v6.csv", EX + "agent_progress.md"] + V5 + CONFIG, "16": V7,
    "17": REFS + CONFIG + [EX + "machine_v7.json"], "18": [], "19": [],
}


def derived():
    """Values the report computes from sources (ratios, differences, means of raw columns), per section."""
    out = {}
    r5 = pd.read_csv(ROOT / DP / "adaptive_runs_v5.csv")
    m = r5.pivot_table(index="scenario", columns="condition", values="latency_seconds", aggfunc="mean")
    best = m[["balanced", "latency_optimized", "static"]].min(axis=1)
    b = m["bandit_adaptive"]
    w = r5.pivot_table(index="context_id", columns="condition", values="latency_seconds")
    reg = w["bandit_adaptive"] - w[["balanced", "latency_optimized", "static"]].min(axis=1)
    ids = sorted(w.index)
    out["11"] = (list((b / best - 1) * 100) + list((b / m["rl_adaptive"] - 1) * 100) +
                 list((b / m["rule_adaptive"] - 1) * 100) +
                 [reg.loc[ids[:len(ids) // 2]].mean(), reg.loc[ids[len(ids) // 2:]].mean()] +
                 [w.loc[c, k] for c in (14, 36) for k in ("rl_adaptive", "rl_once")])
    e6 = pd.read_csv(ROOT / DP / "rl_offline_evaluation_v6.csv")
    out["15"] = [e6[c].mean() for c in e6.columns if c.endswith("latency") or c == "regret"] + \
                [100 * (e6.rl_latency.mean() / e6.oracle_latency.mean() - 1)]
    lat = pd.read_csv(ROOT / EX / "e4_latency.csv").pivot_table(index="scenario", columns="condition", values="mean")
    diff = (lat["bandit_adaptive"] - lat["rule_lookup"]).abs()
    out["16"] = [2 ** 10, diff.max(), 100 * diff.max() / lat.loc[diff.idxmax(), "rule_lookup"]]
    return out

NUM = re.compile(r"(?<![\w.])([-+−]?\d+(?:\.\d+)?)(?![\w])")
MONTHS = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"


# ---------------------------------------------------------------- sources -> value pools
def numbers_in_text(text):
    out = []
    for m in NUM.finditer(text.replace("−", "-")):
        try:
            out.append(float(m.group(1)))
        except ValueError:
            pass
    return out


def aggregates(path):
    """Aggregates the report quotes from raw CSVs (means per design cell)."""
    vals = []
    if path.endswith("load_sweep_v6.csv") or path.endswith("load_sweep_v4.csv"):
        d = pd.read_csv(ROOT / path)
        col = "thread_budget" if "thread_budget" in d.columns else "strategy"
        for v in ("decode_sec_per_token", "latency_seconds", "external_cpu_during"):
            if v in d.columns:
                vals += d.pivot_table(index="load_workers", columns=col, values=v, aggfunc="mean").values.ravel().tolist()
                vals += d.groupby("load_workers")[v].agg(["mean", "min", "max"]).values.ravel().tolist()
        vals.append(len(d))
    return vals


def pool_for(paths, extra=()):
    vals = list(extra)
    for p in paths:
        f = ROOT / p
        if not f.exists():
            raise SystemExit(f"source missing: {p}")
        if p.endswith(".csv"):
            d = pd.read_csv(f)
            vals += [float(x) for x in d.select_dtypes("number").values.ravel() if pd.notna(x)]
            vals.append(float(len(d)))
            for c in d.select_dtypes(exclude="number").columns:  # numbers inside text cells
                for s in d[c].dropna().astype(str):
                    vals += numbers_in_text(s)
            vals += aggregates(p)
        else:
            vals += numbers_in_text(f.read_text(encoding="utf-8", errors="replace"))
    a = np.array(vals, dtype=float)
    a = a[np.isfinite(a)]
    return np.unique(np.concatenate([np.abs(a), np.abs(a) * 100]))


def matches(x_str, pool):
    x = abs(float(x_str.replace("−", "-")))
    d = len(x_str.split(".")[1]) if "." in x_str else 0
    if d == 0:  # integers: exact, or a printed rounding of a decimal value
        return bool(np.any(np.round(pool) == x))
    return bool(np.any(np.abs(np.round(pool, d) - x) < 0.5 * 10 ** (-d) + 1e-9))


# ---------------------------------------------------------------- report parsing
def strip_blocks(h):
    h = re.sub(r"<pre>.*?</pre>", " ", h, flags=re.S)
    h = re.sub(r"<code>.*?</code>", " ", h, flags=re.S)
    h = re.sub(r"<style>.*?</style>", " ", h, flags=re.S)
    h = re.sub(r"<!--.*?-->", " ", h, flags=re.S)
    h = re.sub(r"<img[^>]*>", " ", h)
    return h


def sections(h):
    body = h.split("<body>", 1)[1]
    parts = re.split(r"(<h2>.*?</h2>)", body)
    out, name = [], "Front"
    for p in parts:
        m = re.match(r"<h2>(?:(\d+)\.\s*)?(.*?)</h2>", p)
        if m:
            name = m.group(1) or m.group(2)
            continue
        out.append((name, p))
    return out


def text_of(fragment):
    t = re.sub(r"<h3>.*?</h3>|<h4>.*?</h4>", " ", fragment, flags=re.S)  # headings: section numbers
    t = re.sub(r"<(td|th|li|p|div|tr)[^>]*>", " | ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    return htmllib.unescape(t).replace("−", "-")


STRUCTURAL = [
    ("date", re.compile(rf"(\b{MONTHS}\s+\d{{1,2}}\b|\b\d{{1,2}}(-\d{{1,2}})?\s+{MONTHS}\b|\b(19[5-9]\d|20[0-3]\d)\b|\d{{1,2}}:\d{{2}})")),
    ("version/label", re.compile(r"\b(v\d+r?|P\d(\.\d)?|Gate \d|Phase \d|Figure \d|stage \d|Stage \d|[Ss]ection[s]? \d+(\.\d+)?(-\d+)?|sections? \d+(-\d+)?)\b")),
    ("commit/id", re.compile(r"\b[0-9a-f]{7,40}\b|arXiv:\d{4}\.\d{4,5}|doi:\S+|\[\d\]")),
    ("arm/model name", re.compile(r"\bthreads_\d+|\b\d+(\.\d+)?B\b|Llama ?\d|RTX \d+|\bi5-\d+\w*|fp32|bfloat16|float32")),
    ("range in name", re.compile(r"\bprompts? \d+-\d+|\bIDs? \d+-\d+|\blayers \d+-\d+|\b\d+-\d+ \(two stages\)")),
    ("superseded value in correction note", re.compile(r"earlier versions of this report[^|]*?where the CSV mean")),
]


def structural_spans(text):
    spans = []
    for name, rx in STRUCTURAL:
        for m in rx.finditer(text):
            spans.append((m.start(), m.end(), name))
    return spans


# ---------------------------------------------------------------- strict table checks
def strict_tables(h):
    problems, n = [], 0

    def cells_after(marker, nrows):
        i = h.index(marker)
        tbl = h[i:h.index("</table>", i)]
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", tbl, flags=re.S)[1:nrows + 1]
        return [[htmllib.unescape(re.sub(r"<[^>]+>", "", c)).strip()
                 for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, flags=re.S)] for r in rows]

    # v6 sweep
    d = pd.read_csv(ROOT / DP / "load_sweep_v6.csv")
    piv = d.pivot_table(index="load_workers", columns="thread_budget", values="decode_sec_per_token", aggfunc="mean")
    for r, (w, row) in zip(cells_after("BEGIN v6_sweep_table", len(piv)), piv.iterrows()):
        for c, b in zip(r[1:1 + len(piv.columns)], piv.columns):
            n += 1
            if f"{row[b]:.4f}" != c:
                problems.append(f"v6 sweep {w} workers / {b} threads: report {c}, CSV {row[b]:.4f}")
    # v7 mean latency
    lat = pd.read_csv(ROOT / EX / "e4_latency.csv").pivot_table(index="scenario", columns="condition", values="mean")
    cols = ["threads_2", "balanced", "latency_optimized", "static", "rule_v5", "rule_lookup", "cores_heuristic",
            "rl_adaptive", "bandit_adaptive"]
    for r in cells_after("<h3>16.3 Mean latency", 5):
        for c, cond in zip(r[1:], cols):
            n += 1
            if f"{lat.loc[r[0], cond]:.2f}" != c:
                problems.append(f"v7 latency {r[0]} / {cond}: report {c}, CSV {lat.loc[r[0], cond]:.2f}")
    # v7 tests
    tst = pd.read_csv(ROOT / EX / "e4_tests.csv")
    rows = cells_after("<h3>16.4 Pre-registered tests", 20)
    for i, (r, t) in enumerate(zip(rows, tst.itertuples())):
        r = r[1:] if i % 5 == 0 else r  # first row of each block has the rowspan cell
        lo, hi = re.findall(r"-?\d+\.\d+", r[3])
        holm = r[4]
        checks = [(r[1], f"{t.mean_diff_s:+.2f}"), (r[2], f"{t.mean_diff_percent:+.1f}"),
                  (lo, f"{t.ci95_low_s:.2f}"), (hi, f"{t.ci95_high_s:.2f}")]
        for rep, src in checks:
            n += 1
            if rep.replace("+", "") != src.replace("+", ""):
                problems.append(f"v7 test {t.a} vs {t.b} {t.scenario}: report {rep}, CSV {src}")
        n += 1
        ok = (holm == "<0.0001" and t.p_t_holm < 0.0001) or (holm != "<0.0001" and
                                                             abs(float(holm) - t.p_t_holm) < 0.5 * 10 ** -(len(holm.split(".")[1])))
        if not ok:
            problems.append(f"v7 test {t.a} vs {t.b} {t.scenario}: Holm p report {holm}, CSV {t.p_t_holm:.4g}")
        n += 1
        w = r[5]
        if abs(float(w) - t.p_wilcoxon) >= 0.5 * 10 ** -(len(w.split(".")[1])) + 1e-9:
            problems.append(f"v7 test {t.a} vs {t.b} {t.scenario}: Wilcoxon report {w}, CSV {t.p_wilcoxon:.4g}")
    # best fixed
    bf = pd.read_csv(ROOT / EX / "e4_best_fixed.csv")
    for r, t in zip(cells_after("<h3>16.2 Headline", 5), bf.itertuples()):
        exp = [t.scenario, t.best_fixed, f"{t.mean_rule_lookup_s:.2f}", f"{t.mean_best_fixed_s:.2f}",
               f"{t.mean_diff_s:.2f}", f"{t.mean_diff_percent:.1f}", f"{t.contexts_rule_lookup_faster} / {t.n}"]
        for rep, src in zip(r, exp):
            n += 1
            if rep != src:
                problems.append(f"best fixed {t.scenario}: report {rep}, CSV {src}")
    # segment oracle
    orc = pd.read_csv(ROOT / EX / "e4_segment_oracle.csv").set_index("scenario")
    ocols = ["mean_segment_oracle_s", "mean_best_single_fixed_s"] + [f"{p}_regret_vs_oracle_s" for p in
                                                                      ["rule_lookup", "bandit_adaptive", "cores_heuristic", "rl_adaptive", "rule_v5"]]
    for r in cells_after("<h3>16.6 Segment-oracle", 5):
        for c, col in zip(r[1:], ocols):
            n += 1
            if f"{orc.loc[r[0], col]:.2f}" != c:
                problems.append(f"oracle {r[0]} / {col}: report {c}, CSV {orc.loc[r[0], col]:.2f}")
    return n, problems


def main():
    report = Path(sys.argv[1]) if len(sys.argv) > 1 else REPORT
    h = report.read_text(encoding="utf-8")
    n_cells, table_problems = strict_tables(h)
    cleaned = strip_blocks(h)
    pools, results, unmatched = {}, [], []
    struct_counts = {}
    der = derived()
    for name, frag in sections(cleaned):
        if name in ("Front", "Contents"):
            srcs = []
        else:
            srcs = SECTIONS.get(name, V7)
        key = (name, tuple(srcs))
        if key not in pools:
            pools[key] = pool_for(srcs, der.get(name, ())) if srcs else np.array([])
        text = text_of(frag)
        spans = structural_spans(text)
        n_ok = n_struct = 0
        for m in NUM.finditer(text):
            s = m.group(1)
            hit = next((nm for a, b, nm in spans if a <= m.start() < b), None)
            if hit or name in ("Front", "Contents", "3", "18", "19"):
                why = hit or {"Front": "front matter", "Contents": "contents", "3": "timeline (dates/commits)",
                              "18": "reference list (reference_check.md)", "19": "appendix (paths/commands)"}[name]
                struct_counts[why] = struct_counts.get(why, 0) + 1
                n_struct += 1
            elif len(pools[key]) and matches(s, pools[key]):
                n_ok += 1
            else:
                ctx = text[max(0, m.start() - 70):m.end() + 40].replace("\n", " ")
                unmatched.append((name, s, re.sub(r"\s+", " ", ctx)))
        results.append((name, n_ok, n_struct, srcs))

    lines = [f"Report number check: {report}", ""]
    lines.append(f"1. Strict table checks: {n_cells} cells, {len(table_problems)} differ")
    lines += ["   " + p for p in table_problems]
    lines.append("")
    lines.append("2. Numbers in text, by section (matched to source / structural):")
    tot_ok = tot_struct = 0
    for name, ok, st, srcs in results:
        tot_ok += ok
        tot_struct += st
        if ok or st:
            lines.append(f"   {name:>9}: {ok:4d} matched, {st:4d} structural; sources: {', '.join(srcs) or '-'}")
    lines.append(f"   total: {tot_ok} matched, {tot_struct} structural, {len(unmatched)} unmatched")
    lines.append("   structural rules used: " + ", ".join(f"{k} {v}" for k, v in sorted(struct_counts.items())))
    lines.append("")
    lines.append(f"3. Unmatched numbers: {len(unmatched)}")
    lines += [f"   [{s}] {x}   ...{c}..." for s, x, c in unmatched]
    ok = not table_problems and not unmatched
    lines += ["", f"RESULT: {'PASS' if ok else 'FAIL'}"]
    text = "\n".join(lines) + "\n"
    if report == REPORT:
        OUT.write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
