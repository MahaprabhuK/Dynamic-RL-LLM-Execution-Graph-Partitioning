"""
Checks every number in docs/dashboard/index.html against its source.

The page keeps its numbers in two places, and both are recomputed here from the committed sources,
independently of experiments/build_dashboard.py (nothing is imported from it):

1. Facts: numbers printed in the text as <span class="n" data-k="key">. Each key is recomputed and must
   print the same at the decimals shown; each span must show its fact's value.
2. DATA arrays: every value the charts and tables draw (sweep means, scenario steps, timelines and arm
   bands, best fixed arm, DQN folds, tests, sensitivity tests, oracle regret). Each must match its
   source within half a unit of the stored precision.
3. Every other digit in the visible text, attributes and chart labels must match a named rule
   (identifiers, file names, and three constants whose evidence is checked in the repo).
   Numbers inside the page's own script/style are layout constants; data numbers live only in DATA.

--selftest plants a wrong fact, a wrong span, a wrong data value and a stray number, and confirms the
check fails on each.

Writes experiments/check_dashboard_numbers.txt. Exit code 1 on any mismatch.

    python experiments/check_dashboard_numbers.py [--selftest]
"""

import copy
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "docs" / "dashboard" / "index.html"
OUT = ROOT / "experiments" / "check_dashboard_numbers.txt"
DP, EX = "data/processed/", "experiments/"


def csv(path):
    return pd.read_csv(ROOT / path)


def ymload(path):
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- expected values from the sources
def expected():
    plan, cfg = ymload(EX + "e4_plan.yaml"), ymload("configs/config.yaml")
    mach = json.loads((ROOT / EX / "machine_v7.json").read_text(encoding="utf-8"))
    sweep, runs = csv(DP + "load_sweep_v7.csv"), csv(DP + "v7runs_v7.csv")
    tl = csv(DP + "v7timelines_v7.csv")
    bf, dqn = csv(EX + "e4_best_fixed.csv"), csv(EX + "e3_dqn_v7.csv")
    tests, sens, orc = csv(EX + "e4_tests.csv"), csv(EX + "e4_tests_sensitivity.csv"), csv(EX + "e4_segment_oracle.csv")
    scen = list(plan["scenarios"].keys())
    F, D = {}, {}

    # sweep
    arms = next(c for c in plan["conditions"] if c["name"] == "rl_adaptive")["arms"]
    g = sweep.groupby(["load_workers", "arm"]).decode_sec_per_token.mean()
    ws = sorted(sweep.load_workers.unique())
    D["sweep.workers"] = [int(w) for w in ws]
    D["sweep.arms"] = arms
    for a in arms:
        D[f"sweep.mean.{a}"] = [g[(w, a)] for w in ws]
    top = max(ws)
    at = {a: g[(top, a)] for a in arms}
    best = min(at, key=at.get)
    F.update({"sweep.n_runs": len(sweep), "sweep.cliff_workers": top, "sweep.cliff_t10": at["threads_10"],
              "sweep.cliff_best_arm": best, "sweep.cliff_best": at[best], "sweep.cliff_ratio": at["threads_10"] / at[best]})
    D["sweep.cliff"] = {"workers": int(top), "best_arm": best}

    # model, plan
    lr = cfg["strategies"]["static"]["layer_ranges"]
    F["cfg.n_layers"] = lr[-1][1] - lr[0][0] + 1
    F["cfg.model"] = cfg["model"]["name"].rsplit("/", 1)[1]
    F["plan.interval"] = plan["decision_interval"]
    ident = [grp.token_ids.nunique() == 1 for _, grp in runs.groupby(["prompt_id", "scenario"])]
    F["main.identical_pct"] = 100.0 * sum(ident) / len(ident)
    F.update({"plan.n_prompts": len(plan["prompts"]), "plan.n_scenarios": len(scen),
              "plan.n_conditions": len(plan["conditions"]), "plan.total_runs": plan["total_runs"],
              "plan.max_new_tokens": plan["max_new_tokens"], "plan.n_tests": len(plan["primary_analysis"]["tests"]),
              "main.n_runs": len(runs)})
    D["scenarios"] = {s: [list(map(int, p)) for p in plan["scenarios"][s]] for s in scen}
    D["max_new_tokens"] = plan["max_new_tokens"]
    D["scenario_order"] = scen

    # timelines: context whose rule_lookup latency is closest to the scenario median (ties: lower latency)
    for s in ["ramp", "spike"]:
        r = runs[(runs.scenario == s) & (runs.condition == "rule_lookup")]
        med = float(np.median(r.latency_seconds))
        ctx = min(r.itertuples(), key=lambda x: (abs(x.latency_seconds - med), x.latency_seconds))
        part = tl[tl.context_id == ctx.context_id]
        D[f"timeline.{s}.context_id"] = int(ctx.context_id)
        D[f"timeline.{s}.prompt_id"] = int(ctx.prompt_id)
        F[f"tl.{s}.prompt"] = int(ctx.prompt_id)
        for c in ["static", "balanced", "rl_adaptive", "rule_lookup"]:
            seq = part[part.condition == c].sort_values("token_step")
            cum = np.cumsum(seq.token_seconds.to_numpy())
            D[f"timeline.{s}.cum.{c}"] = list(cum)
            D[f"timeline.{s}.steps"] = [int(t) for t in seq.token_step]
            F[f"tl.{s}.{c}"] = cum[-1]
        armseq = part[part.condition == "rule_lookup"].sort_values("token_step").arm.tolist()
        bands, i = [], 0
        while i < len(armseq):
            j = i
            while j + 1 < len(armseq) and armseq[j + 1] == armseq[i]:
                j += 1
            bands.append([i, j, armseq[i]])
            i = j + 1
        D[f"timeline.{s}.bands"] = bands

    # best fixed
    bfi = bf.set_index("scenario")
    D["best_fixed"] = [{"scenario": s, "arm": bfi.loc[s, "best_fixed"], "faster_pct": -bfi.loc[s, "mean_diff_percent"],
                        "contexts_faster": int(bfi.loc[s, "contexts_rule_lookup_faster"]), "n": int(bfi.loc[s, "n"]),
                        "rule_s": bfi.loc[s, "mean_rule_lookup_s"], "fixed_s": bfi.loc[s, "mean_best_fixed_s"]} for s in scen]
    gains = {s: -bfi.loc[s, "mean_diff_percent"] for s in scen}
    bs = max(gains, key=gains.get)
    F.update({"bf.min_pct": min(gains.values()), "bf.max_pct": gains[bs], "bf.best_scen": bs,
              "bf.best_arm": bfi.loc[bs, "best_fixed"], "bf.best_faster": int(bfi.loc[bs, "contexts_rule_lookup_faster"]),
              "bf.n": int(bfi.loc[bs, "n"])})

    # DQN folds
    pol = ["rule_lookup_fold", "dqn_v7", "cores_heuristic"]
    D["dqn.folds"] = sorted(int(x) for x in dqn.held_out_prompt.unique())
    for p in pol:
        D[f"dqn.pct.{p}"] = [float(dqn[(dqn.policy == p) & (dqn.held_out_prompt == f)].percent_best_arm.iloc[0]) for f in D["dqn.folds"]]
    F.update({"dqn.min": min(D["dqn.pct.dqn_v7"]), "dqn.max": max(D["dqn.pct.dqn_v7"]),
              "dqn.rule_min": min(D["dqn.pct.rule_lookup_fold"]), "dqn.rule_max": max(D["dqn.pct.rule_lookup_fold"])})

    # rl vs rule
    sigc = "significant_holm_0.05"
    rr = {t.scenario: t for t in tests[(tests.a == "rl_adaptive") & (tests.b == "rule_lookup")].rename(columns={sigc: "sig"}).itertuples()}
    D["rl_vs_rule"] = [{"scenario": s, "slower_pct": rr[s].mean_diff_percent, "holm": bool(rr[s].sig), "p_holm": rr[s].p_t_holm} for s in scen]
    pct = [rr[s].mean_diff_percent for s in scen]
    F.update({"rl.n": len(rr), "rl.n_slower": sum(p > 0 for p in pct), "rl.min": min(pct), "rl.max": max(pct),
              "rl.n_sig": sum(bool(rr[s].sig) for s in scen)})

    # tests: re-orient to "seconds rule_lookup saves"
    def orient(df):
        out = []
        for _, x in df.iterrows():
            if x.a == "rule_lookup":
                out.append({"other": x.b, "scenario": x.scenario, "n": int(x.n), "saved_s": -x.mean_diff_s,
                            "lo": -x.ci95_high_s, "hi": -x.ci95_low_s, "p_holm": x.p_t_holm, "holm": bool(x[sigc])})
            else:
                out.append({"other": x.a, "scenario": x.scenario, "n": int(x.n), "saved_s": x.mean_diff_s,
                            "lo": x.ci95_low_s, "hi": x.ci95_high_s, "p_holm": x.p_t_holm, "holm": bool(x[sigc])})
        return out
    D["tests"], D["tests_sens"] = orient(tests), orient(sens)
    saved = [t["saved_s"] for t in D["tests"]]
    sig = [t["holm"] for t in D["tests"]]
    sdir = {(t["other"], t["scenario"]): np.sign(t["saved_s"]) for t in D["tests_sens"]}
    F.update({"tests.n": len(tests), "tests.n_sig": sum(sig), "tests.n_sig_favour": sum(s and v > 0 for s, v in zip(sig, saved)),
              "sens.n_sig": int(sens[sigc].sum()),
              "sens.same_dir": sum(sdir[(t["other"], t["scenario"])] == np.sign(t["saved_s"]) for t in D["tests"])})

    # oracle
    oi = orc.set_index("scenario")
    opol = ["rule_lookup", "bandit_adaptive", "cores_heuristic", "rl_adaptive"]
    D["oracle.policies"] = opol
    for p in opol:
        D[f"oracle.regret.{p}"] = [oi.loc[s, f"{p}_regret_vs_oracle_s"] for s in scen]
    D["oracle.oracle_s"] = [oi.loc[s, "mean_segment_oracle_s"] for s in scen]
    rl_reg = dict(zip(scen, D["oracle.regret.rl_adaptive"]))
    ru_reg = dict(zip(scen, D["oracle.regret.rule_lookup"]))
    F.update({"or.rule_max": max(ru_reg.values()), "or.rl_max": max(rl_reg.values()),
              "or.rl_max_scen": max(rl_reg, key=rl_reg.get), "or.neg_scen": min(ru_reg, key=ru_reg.get),
              "or.neg_val": min(ru_reg.values())})

    # limits
    flag = runs.paging_flag.astype(str).str.lower() == "true"
    dates = pd.to_datetime(runs.wall_clock_start).dt.date.astype(str)
    F.update({"main.paging_n": int(flag.sum()), "main.runs_first": int((dates == "2026-10-13").sum()),
              "main.runs_later": int((dates > "2026-10-13").sum()),
              "machine.physical": mach["physical_cores"], "machine.logical": mach["logical_cores"]})
    return F, D


# ---------------------------------------------------------------- page parsing
class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = 0          # inside script/style
        self.span = []          # stack: True if a fact span
        self.cur_key = None
        self.spans, self.text, self.attrs, self.data_json = [], [], [], None
        self._in_data = False
        self._buf = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("script", "style"):
            self.skip += 1
            self._in_data = a.get("id") == "dash-data"
            self._buf = []
        for k in ("aria-label", "title", "alt", "placeholder"):
            if a.get(k) and not self.skip:
                self.attrs.append(a[k])
        if tag == "meta" and a.get("name") == "description":
            self.attrs.append(a.get("content", ""))
        if tag == "span":
            is_fact = "n" in (a.get("class") or "").split() and "data-k" in a
            self.span.append(is_fact)
            if is_fact:
                self.cur_key, self._fact = a["data-k"], []

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            if self._in_data:
                self.data_json = "".join(self._buf)
            self.skip -= 1
            self._in_data = False
        if tag == "span" and self.span:
            if self.span.pop():
                self.spans.append((self.cur_key, "".join(self._fact)))
                self.cur_key = None

    def handle_data(self, data):
        if self.skip:
            if self._in_data:
                self._buf.append(data)
            return
        if self.cur_key is not None:
            self._fact.append(data)
        else:
            self.text.append(data)


def parse(html_text):
    p = Page()
    p.feed(html_text)
    return p


# ---------------------------------------------------------------- comparisons
def printed_matches(expected_value, printed):
    if isinstance(expected_value, str):
        return expected_value == printed
    m = re.fullmatch(r"-?\d+(?:\.(\d+))?", printed)
    if not m:
        return False
    dp = len(m.group(1) or "")
    return f"{float(expected_value):.{dp}f}" == printed


def close(a, b, path, errs, tol):
    if isinstance(b, dict):
        if not isinstance(a, dict) or set(a) != set(b):
            errs.append(f"{path}: keys {sorted(a) if isinstance(a, dict) else type(a).__name__} != {sorted(b)}")
            return
        for k in b:
            close(a[k], b[k], f"{path}.{k}", errs, tol)
    elif isinstance(b, (list, tuple)):
        if not isinstance(a, list) or len(a) != len(b):
            errs.append(f"{path}: length {len(a) if isinstance(a, list) else '-'} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            close(x, y, f"{path}[{i}]", errs, tol)
    elif isinstance(b, (bool, np.bool_)) or isinstance(b, str):
        if a != b:
            errs.append(f"{path}: {a!r} != {b!r}")
    else:
        b = float(b)
        t = tol(path, b)
        if not isinstance(a, (int, float)) or isinstance(a, bool) or abs(a - b) > t:
            errs.append(f"{path}: page {a!r} vs source {b:.6g} (tolerance {t:.2g})")


def tolerance(path, value):
    # stored precisions in DATA (half a unit) ; p-values are stored to 4 significant digits
    if "p_holm" in path:
        return abs(value) * 5e-4 + 1e-15
    for key, dp in [("sweep.mean", 5), ("timeline", 4), ("faster_pct", 2), ("slower_pct", 2), ("rule_s", 3),
                    ("fixed_s", 3), ("dqn.pct", 1), ("saved_s", 4), (".lo", 4), (".hi", 4), ("oracle", 3)]:
        if key in path:
            return 0.5 * 10 ** -dp + 1e-9
    return 1e-9


def page_data_view(data):
    """Flatten the page DATA into the keys used by expected()."""
    v = {"sweep.workers": data["sweep"]["workers"], "sweep.arms": data["sweep"]["arms"], "sweep.cliff": data["sweep"]["cliff"],
         "scenarios": data["scenarios"], "max_new_tokens": data["max_new_tokens"], "scenario_order": data["scenario_order"],
         "best_fixed": data["best_fixed"], "dqn.folds": data["dqn"]["folds"], "rl_vs_rule": data["rl_vs_rule"],
         "tests": data["tests"], "tests_sens": data["tests_sens"], "oracle.policies": data["oracle"]["policies"],
         "oracle.oracle_s": data["oracle"]["oracle_s"]}
    for a, vals in data["sweep"]["mean"].items():
        v[f"sweep.mean.{a}"] = vals
    for s, t in data["timeline"].items():
        v[f"timeline.{s}.context_id"], v[f"timeline.{s}.prompt_id"] = t["context_id"], t["prompt_id"]
        v[f"timeline.{s}.steps"], v[f"timeline.{s}.bands"] = t["steps"], t["bands"]
        for c, cum in t["cum"].items():
            v[f"timeline.{s}.cum.{c}"] = cum
    for p, vals in data["dqn"]["pct"].items():
        v[f"dqn.pct.{p}"] = vals
    for p, vals in data["oracle"]["regret"].items():
        v[f"oracle.regret.{p}"] = vals
    return v


# Digits allowed outside facts and DATA, each with its reason. Evidence: (file, text that must be in it).
ALLOWED = [
    (r"\bthreads_\d+\b", "arm identifier", None),
    (r"\b[\w./-]*_v\d+\b(?:\.\w+)?|\b[\w/.-]+\.(?:csv|yaml|json|py|html)\b", "file or version identifier", None),
    (r"\bRTX 3050\b", "GPU model name (next steps)", None),
    (r"\b17 Oct\b", "date of the power-setting change", ("docs/report/final_report.html", "On 17 Oct the laptop")),
    (r"\b95% CI\b", "confidence level of the pre-registered CI", (EX + "e4_plan.yaml", "95% bootstrap CI")),
]


def stray_numbers(strings, errs, notes):
    seen = {}
    for s in strings:
        rest = s
        for pat, why, ev in ALLOWED:
            for m in re.finditer(pat, rest):
                seen.setdefault(why, set()).add(m.group(0))
            rest = re.sub(pat, " ", rest)
        for m in re.finditer(r"\d[\d.,]*", rest):
            ctx = rest[max(0, m.start() - 40): m.end() + 40].replace("\n", " ").strip()
            errs.append(f"unmatched number {m.group(0)!r} in: ...{ctx}...")
    for pat, why, ev in ALLOWED:
        if ev and why in seen:
            f, needle = ev
            if needle not in (ROOT / f).read_text(encoding="utf-8"):
                errs.append(f"allowed constant {sorted(seen[why])} ({why}) has no evidence: {needle!r} not in {f}")
            else:
                notes.append(f"allowed {sorted(seen[why])}: {why} (evidence: {f})")
        elif why in seen:
            notes.append(f"allowed {sorted(seen[why])}: {why}")


def data_strings(data):
    out = []
    def walk(x, path):
        if isinstance(x, dict):
            for k, v in x.items():
                if path == "" and k in ("facts", "sources"):
                    continue
                walk(v, f"{path}.{k}")
        elif isinstance(x, list):
            for v in x:
                walk(v, path)
        elif isinstance(x, str):
            out.append(x)
    walk(data, "")
    return out


def check(html_text, F, D):
    errs, notes = [], []
    p = parse(html_text)
    data = json.loads(p.data_json)
    facts = data["facts"]

    # 1. facts
    missing = sorted(set(F) - set(facts))
    extra = sorted(set(facts) - set(F))
    if missing:
        errs.append(f"facts missing from page: {missing}")
    if extra:
        errs.append(f"facts on page with no recomputation: {extra}")
    for k in sorted(set(F) & set(facts)):
        if not printed_matches(F[k], facts[k]):
            src = F[k] if isinstance(F[k], str) else f"{float(F[k]):.6g}"
            errs.append(f"fact {k}: page {facts[k]!r} vs source {src!r}")
    for k, shown in p.spans:
        if k not in facts:
            errs.append(f"span data-k={k!r} has no fact")
        elif shown != facts[k]:
            errs.append(f"span {k}: shows {shown!r}, fact is {facts[k]!r}")
    used = {k for k, _ in p.spans}
    unused = sorted(set(facts) - used)

    # 2. DATA arrays
    view = page_data_view(data)
    for k in sorted(set(D) | set(view)):
        if k not in view or k not in D:
            errs.append(f"data {k}: {'missing on page' if k not in view else 'no recomputation'}")
            continue
        close(view[k], D[k], k, errs, tolerance)
    if data["sources"]:
        for name, path in data["sources"].items():
            if not (ROOT / path).exists():
                errs.append(f"source {name} -> {path} does not exist")

    # 3. stray numbers in text, attributes and chart labels
    stray_numbers(p.text + p.attrs + data_strings(data), errs, notes)
    n_values = sum(len(v) if isinstance(v, list) else 1 for v in view.values())
    summary = [f"facts: {len(facts)} checked, {len(p.spans)} spans in text ({len(used)} keys used; unused: {unused or 'none'})",
               f"data values: {n_values} top-level entries/arrays elements compared across {len(view)} keys"]
    return errs, notes + summary


def selftest(html_text, F, D):
    """Each planted error must make the check fail, with an error naming that error."""
    p = parse(html_text)
    data = json.loads(p.data_json)
    cases = []
    k = "bf.max_pct"
    d1 = copy.deepcopy(data); d1["facts"][k] = "9.9"
    cases.append(("wrong fact value", replace_data(html_text, d1), "fact bf.max_pct"))
    shown = data["facts"]["tests.n_sig"]
    cases.append(("span differs from fact", html_text.replace(f'data-k="tests.n_sig">{shown}<', 'data-k="tests.n_sig">16<', 1),
                  "span tests.n_sig"))
    d2 = copy.deepcopy(data); d2["timeline"]["ramp"]["cum"]["rule_lookup"][100] += 0.01
    cases.append(("timeline value off by 0.01 s", replace_data(html_text, d2), "timeline.ramp.cum.rule_lookup[100]"))
    d3 = copy.deepcopy(data); d3["tests"][3]["hi"] += 0.001
    cases.append(("CI bound off by 0.001 s", replace_data(html_text, d3), "tests[3].hi"))
    cases.append(("stray number in text", html_text.replace("<h2>Summary</h2>", "<h2>Summary of 12 runs</h2>", 1), "Summary of 12"))
    d4 = copy.deepcopy(data); d4["meta"]["forest"]["title"] += " (n = 12)"
    cases.append(("stray number in a chart title", replace_data(html_text, d4), "(n = 12)"))
    lines, ok = [], True
    for name, mutated, needle in cases:
        e, _ = check(mutated, F, D)
        e = [x for x in e if needle in x] or e
        caught = any(needle in x for x in e)
        ok &= caught
        lines.append(f"  {'caught' if caught else 'MISSED'}: {name} ({len(e)} error(s); first: {e[0] if e else '-'})")
    return ok, lines


def replace_data(html_text, data):
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return re.sub(r'(<script id="dash-data" type="application/json">).*?(</script>)',
                  lambda m: m.group(1) + payload + m.group(2), html_text, count=1, flags=re.S)


def main():
    html_text = PAGE.read_text(encoding="utf-8")
    F, D = expected()
    errs, notes = check(html_text, F, D)
    lines = [f"Dashboard number check: {PAGE.relative_to(ROOT).as_posix()} ({PAGE.stat().st_size / 1e6:.2f} MB)"]
    lines += notes
    if "--selftest" in sys.argv:
        ok, st = selftest(html_text, F, D)
        lines.append("self-test (planted errors must be caught): " + ("PASS" if ok else "FAIL"))
        lines += st
        if not ok:
            errs.append("self-test missed a planted error")
    lines.append(f"errors: {len(errs)}")
    lines += ["  " + e for e in errs[:200]]
    lines.append("RESULT: " + ("PASS" if not errs else "FAIL"))
    text = "\n".join(lines)
    print(text)
    OUT.write_text(text + "\n", encoding="utf-8")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
