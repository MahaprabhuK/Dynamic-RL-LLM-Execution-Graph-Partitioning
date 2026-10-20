"""
Builds docs/dashboard/index.html, the presentation / results dashboard.

Reads only committed sources (CSV / YAML / JSON listed in SRC) and writes one self-contained page.
Plotly.js is embedded inline, so the page works offline with no server.

Every number on the page comes from the sources:
- numbers in the text are "facts": printed as <span class="n" data-k="key">, values in DATA.facts;
- chart and table values are arrays in the embedded DATA block; the page draws everything from it.
experiments/check_dashboard_numbers.py recomputes both from the sources independently.

    python experiments/build_dashboard.py
"""

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "dashboard" / "index.html"
REPO_URL = "https://github.com/MahaprabhuK/Dynamic-RL-LLM-Execution-Graph-Partitioning"

SRC = {
    "sweep": "data/processed/load_sweep_v7.csv",
    "runs": "data/processed/v7runs_v7.csv",
    "timelines": "data/processed/v7timelines_v7.csv",
    "plan": "experiments/e4_plan.yaml",
    "config": "configs/config.yaml",
    "machine": "experiments/machine_v7.json",
    "best_fixed": "experiments/e4_best_fixed.csv",
    "dqn": "experiments/e3_dqn_v7.csv",
    "tests": "experiments/e4_tests.csv",
    "sens": "experiments/e4_tests_sensitivity.csv",
    "oracle": "experiments/e4_segment_oracle.csv",
}

TIMELINE_CONDITIONS = ["static", "balanced", "rl_adaptive", "rule_lookup"]
DQN_POLICIES = ["rule_lookup_fold", "dqn_v7", "cores_heuristic"]
ORACLE_POLICIES = ["rule_lookup", "bandit_adaptive", "cores_heuristic", "rl_adaptive"]

facts = {}


def fact(key, value, dp=None):
    """Store a printed value; dp=None prints str(value) (ints, names)."""
    facts[key] = str(value) if dp is None else f"{value:.{dp}f}"
    return facts[key]


def n(key):
    return f'<span class="n" data-k="{key}">{html.escape(facts[key])}</span>'


def rnd(values, dp):
    return [round(float(v), dp) for v in values]


def read_csv(name):
    return pd.read_csv(ROOT / SRC[name])


def load():
    plan = yaml.safe_load((ROOT / SRC["plan"]).read_text(encoding="utf-8"))
    config = yaml.safe_load((ROOT / SRC["config"]).read_text(encoding="utf-8"))
    machine = json.loads((ROOT / SRC["machine"]).read_text(encoding="utf-8"))
    return plan, config, machine


def build_data():
    plan, config, machine = load()
    data = {"sources": SRC, "meta": {}}
    scenarios = list(plan["scenarios"])
    data["scenario_order"] = scenarios

    # Panel 2: load sweep, mean decode s/token per (workers, arm)
    sweep = read_csv("sweep")
    arms = plan["conditions"][[c["name"] for c in plan["conditions"]].index("rl_adaptive")]["arms"]
    mean = sweep.pivot_table(index="load_workers", columns="arm", values="decode_sec_per_token", aggfunc="mean")
    workers = [int(w) for w in mean.index]
    data["sweep"] = {"workers": workers, "arms": arms, "mean": {a: rnd(mean[a], 5) for a in arms}}
    cliff_w = max(workers)
    row = mean.loc[cliff_w]
    best_arm = row.idxmin()
    fact("sweep.n_runs", len(sweep))
    fact("sweep.cliff_workers", cliff_w)
    fact("sweep.cliff_t10", row["threads_10"], 2)
    fact("sweep.cliff_best_arm", best_arm)
    fact("sweep.cliff_best", row[best_arm], 2)
    fact("sweep.cliff_ratio", row["threads_10"] / row[best_arm], 0)
    data["sweep"]["cliff"] = {"workers": cliff_w, "best_arm": best_arm}

    # Panel 3: model, interval, identical output
    runs = read_csv("runs")
    layer_ranges = config["strategies"]["static"]["layer_ranges"]
    fact("cfg.n_layers", layer_ranges[-1][1] - layer_ranges[0][0] + 1)
    fact("cfg.model", config["model"]["name"].split("/")[-1])
    fact("plan.interval", plan["decision_interval"])
    same = runs.groupby(["prompt_id", "scenario"])["token_ids"].nunique() == 1
    fact("main.identical_pct", 100 * same.mean(), 0)
    data["arms"] = arms

    # Panel 4: plan and protocol
    conds = [c["name"] for c in plan["conditions"]]
    fact("plan.n_prompts", len(plan["prompts"]))
    fact("plan.n_scenarios", len(scenarios))
    fact("plan.n_conditions", len(conds))
    fact("plan.total_runs", plan["total_runs"])
    fact("plan.max_new_tokens", plan["max_new_tokens"])
    fact("plan.n_tests", len(plan["primary_analysis"]["tests"]))
    fact("main.n_runs", len(runs))
    data["scenarios"] = {s: [[int(t), int(w)] for t, w in plan["scenarios"][s]] for s in scenarios}
    data["max_new_tokens"] = int(plan["max_new_tokens"])

    # Panel 5: one context per scenario (rule_lookup latency closest to that scenario's median)
    tl = read_csv("timelines")
    data["timeline"] = {}
    for s in ["ramp", "spike"]:
        rl = runs[(runs.scenario == s) & (runs.condition == "rule_lookup")].copy()
        rl["dist"] = (rl.latency_seconds - rl.latency_seconds.median()).abs()
        pick = rl.sort_values(["dist", "latency_seconds"]).iloc[0]
        ctx = int(pick.context_id)
        part = tl[tl.context_id == ctx].sort_values("token_step")
        entry = {"context_id": ctx, "prompt_id": int(pick.prompt_id), "cum": {}, "bands": []}
        for c in TIMELINE_CONDITIONS:
            ts = part[part.condition == c]
            entry["steps"] = [int(x) for x in ts.token_step]
            entry["cum"][c] = rnd(ts.token_seconds.cumsum(), 4)
        arm_seq = list(part[part.condition == "rule_lookup"].arm)
        start = 0
        for i in range(1, len(arm_seq) + 1):
            if i == len(arm_seq) or arm_seq[i] != arm_seq[start]:
                entry["bands"].append([start, i - 1, arm_seq[start]])
                start = i
        data["timeline"][s] = entry
        fact(f"tl.{s}.prompt", entry["prompt_id"])
        for c in TIMELINE_CONDITIONS:
            fact(f"tl.{s}.{c}", entry["cum"][c][-1], 1)

    # Panel 6: rule_lookup vs best fixed arm per scenario
    bf = read_csv("best_fixed").set_index("scenario").loc[scenarios].reset_index()
    data["best_fixed"] = [{"scenario": r.scenario, "arm": r.best_fixed, "faster_pct": round(-r.mean_diff_percent, 2),
                           "contexts_faster": int(r.contexts_rule_lookup_faster), "n": int(r.n),
                           "rule_s": round(r.mean_rule_lookup_s, 3), "fixed_s": round(r.mean_best_fixed_s, 3)}
                          for r in bf.itertuples()]
    gain = -bf.mean_diff_percent
    top = bf.loc[gain.idxmax()]
    fact("bf.min_pct", gain.min(), 1)
    fact("bf.max_pct", gain.max(), 1)
    fact("bf.best_scen", top.scenario)
    fact("bf.best_arm", top.best_fixed)
    fact("bf.best_faster", int(top.contexts_rule_lookup_faster))
    fact("bf.n", int(top.n))

    # Panel 7: DQN offline (held-out prompts) and online slowdown vs rule_lookup
    dqn = read_csv("dqn")
    dqn = dqn[dqn.policy.isin(DQN_POLICIES)]
    data["dqn"] = {"folds": sorted(int(f) for f in dqn.held_out_prompt.unique()),
                   "pct": {p: rnd(dqn[dqn.policy == p].sort_values("held_out_prompt").percent_best_arm, 1)
                           for p in DQN_POLICIES}}
    d, r = dqn[dqn.policy == "dqn_v7"].percent_best_arm, dqn[dqn.policy == "rule_lookup_fold"].percent_best_arm
    fact("dqn.min", d.min(), 1)
    fact("dqn.max", d.max(), 1)
    fact("dqn.rule_min", r.min(), 1)
    fact("dqn.rule_max", r.max(), 1)
    tests = read_csv("tests")
    rlr = tests[(tests.a == "rl_adaptive") & (tests.b == "rule_lookup")].set_index("scenario").loc[scenarios]
    data["rl_vs_rule"] = [{"scenario": s, "slower_pct": round(x.mean_diff_percent, 2),
                           "holm": bool(x["significant_holm_0.05"]), "p_holm": float(f"{x.p_t_holm:.4g}")}
                          for s, x in rlr.iterrows()]
    fact("rl.n", len(rlr))
    fact("rl.n_slower", int((rlr.mean_diff_percent > 0).sum()))
    fact("rl.min", rlr.mean_diff_percent.min(), 1)
    fact("rl.max", rlr.mean_diff_percent.max(), 1)
    fact("rl.n_sig", int(rlr["significant_holm_0.05"].sum()))

    # Panel 8: forest plot, oriented as seconds rule_lookup saves (other - rule_lookup)
    sens = read_csv("sens")

    def oriented(df):
        out = []
        for x in df.rename(columns={"significant_holm_0.05": "sig"}).itertuples():
            flip = x.a == "rule_lookup"
            out.append({"other": x.b if flip else x.a, "scenario": x.scenario, "n": int(x.n),
                        "saved_s": round(-x.mean_diff_s if flip else x.mean_diff_s, 4),
                        "lo": round(-x.ci95_high_s if flip else x.ci95_low_s, 4),
                        "hi": round(-x.ci95_low_s if flip else x.ci95_high_s, 4),
                        "p_holm": float(f"{x.p_t_holm:.4g}"), "holm": bool(x.sig)})
        return out

    data["tests"], data["tests_sens"] = oriented(tests), oriented(sens)
    sig = tests["significant_holm_0.05"]
    favour = np.where(tests.a == "rule_lookup", tests.mean_diff_s < 0, tests.mean_diff_s > 0)
    keyed = sens.set_index(["a", "b", "scenario"]).mean_diff_s
    same_dir = sum(np.sign(keyed[(t.a, t.b, t.scenario)]) == np.sign(t.mean_diff_s) for t in tests.itertuples())
    fact("tests.n", len(tests))
    fact("tests.n_sig", int(sig.sum()))
    fact("tests.n_sig_favour", int((sig & favour).sum()))
    fact("sens.n_sig", int(sens["significant_holm_0.05"].sum()))
    fact("sens.same_dir", int(same_dir))

    # Panel 9: regret vs the per-segment oracle
    oracle = read_csv("oracle").set_index("scenario").loc[scenarios]
    data["oracle"] = {"policies": ORACLE_POLICIES,
                      "regret": {p: rnd(oracle[f"{p}_regret_vs_oracle_s"], 3) for p in ORACLE_POLICIES},
                      "oracle_s": rnd(oracle.mean_segment_oracle_s, 3)}
    rule_reg = oracle["rule_lookup_regret_vs_oracle_s"]
    fact("or.rule_max", rule_reg.max(), 2)
    fact("or.rl_max", oracle["rl_adaptive_regret_vs_oracle_s"].max(), 1)
    fact("or.rl_max_scen", oracle["rl_adaptive_regret_vs_oracle_s"].idxmax())
    fact("or.neg_scen", rule_reg.idxmin())
    fact("or.neg_val", rule_reg.min(), 2)

    # Panel 10: limits
    fact("main.paging_n", int(runs.paging_flag.astype(bool).sum()))
    day = runs.wall_clock_start.str[:10]
    fact("main.runs_first", int((day == "2026-10-13").sum()))
    fact("main.runs_later", int((day > "2026-10-13").sum()))
    fact("machine.physical", machine["physical_cores"])
    fact("machine.logical", machine["logical_cores"])

    # Chart titles, axis labels and sources (scanned by the checker like the page text)
    meta = data["meta"]
    meta["sweep"] = {"title": "Decode speed vs background load, per thread setting",
                     "x": "Background busy-loop workers", "y": "Mean decode time (s/token, log scale)",
                     "src": SRC["sweep"]}
    for s in scenarios:
        meta[f"scen_{s}"] = {"title": s.replace("_", " "), "x": "Token", "y": "Workers", "src": SRC["plan"]}
    meta["timeline"] = {"title": "Cumulative generation time inside one request",
                        "x": "Token", "y": "Cumulative time (s)", "src": SRC["timelines"]}
    meta["best_fixed"] = {"title": "rule_lookup vs the best fixed arm in each scenario",
                          "x": "Faster than best fixed arm (%)", "y": "", "src": SRC["best_fixed"]}
    meta["dqn"] = {"title": "Offline: contexts where the policy picked the best arm",
                   "x": "Held-out prompt", "y": "Best arm picked (%)", "src": SRC["dqn"]}
    meta["rl_slow"] = {"title": "Online: rl_adaptive slower than rule_lookup",
                       "x": "Slower than rule_lookup (%)", "y": "", "src": SRC["tests"]}
    meta["forest"] = {"title": "Pre-registered paired tests: seconds rule_lookup saves per request",
                      "x": "Seconds saved by rule_lookup (mean, 95% CI)", "y": "", "src": SRC["tests"]}
    meta["oracle"] = {"title": "Regret vs the per-segment oracle (s per request)",
                      "x": "Scenario", "y": "Policy", "src": SRC["oracle"]}
    data["facts"] = facts
    return data


def panels():
    """Panel HTML. Text numbers only through n(key)."""
    f = n
    P = []
    P.append(("Title", """
<div class="hero">
  <p class="kicker">VIT</p>
  <h1>Dynamic RL-LLM Execution Graph Partitioning</h1>
  <p class="subtitle">Load-adaptive thread scheduling for LLM inference on a CPU</p>
  <p class="team">Mahaprabhu K · Priyanka M · Karthikeyan S</p>
  <p class="guide">Guide: Berlin M A</p>
  <p class="hint present-only">→ or space to begin · P explore mode · N notes · F fullscreen</p>
  <p class="hint explore-only">Press P (or the button, top right) to present. Each section below is one slide.</p>
</div>""", """We run a small language model on a laptop CPU while other programs compete for it.
The question: can the model pick its own thread count, mid-answer, and stay fast?
Keys: arrows or space to move, number keys to jump, N for these notes."""))

    P.append(("The problem", f"""
<h2>Too many threads on a busy CPU makes the model collapse.</h2>
<p class="lede">We timed every thread setting while busy-loop workers took over the CPU ({f('sweep.n_runs')} runs).
With {f('sweep.cliff_workers')} background workers, <code>threads_10</code> needs <b class="callout-red">{f('sweep.cliff_t10')} s per token</b>:
{f('sweep.cliff_ratio')}× slower than <code>{f('sweep.cliff_best_arm')}</code> ({f('sweep.cliff_best')} s).</p>
<div class="visual"><div class="chart" id="ch-sweep"></div></div>""",
              f"""The setting that is fastest on an idle machine is the one that falls off the cliff first.
Each point is the mean over prompts and repeats; the y axis is logarithmic.
So no single fixed thread count is right all the time; that is the whole motivation."""))

    P.append(("Our idea", f"""
<h2>Re-pick the thread count while the answer is being written.</h2>
<p class="lede">The model runs layer by layer through our own decode loop, so the thread budget can change between tokens.
The output stays token-identical: same token IDs across all conditions in {f('main.identical_pct')}% of contexts.</p>
<div class="visual col">
  <div class="flow" role="img" aria-label="Decode loop: prompt, layers, read CPU load, policy picks threads, continue">
    <div class="box">Prompt</div><div class="arrow">→</div>
    <div class="box"><span class="bh">{f('cfg.n_layers')} layers</span><small>one or two stages, each with its own thread budget</small></div><div class="arrow">→</div>
    <div class="box"><span class="bh">Every {f('plan.interval')} tokens</span><small>read CPU used by other processes</small></div><div class="arrow">→</div>
    <div class="box accent">Policy picks a thread count</div><div class="arrow">→</div>
    <div class="box">Continue generating<small>↺ back to the layers</small></div>
  </div>
  <table class="policies">
    <thead><tr><th>Policy</th><th>How it picks</th></tr></thead>
    <tbody>
    <tr class="ours"><td><code>rule_lookup</code> <span>lookup table</span></td><td>Best arm for each CPU-load band, measured in the load sweep on this laptop.</td></tr>
    <tr><td><code>cores_heuristic</code> <span>spare cores</span></td><td>Threads = physical cores minus the cores other processes are using. Not fitted to data.</td></tr>
    <tr><td><code>rl_adaptive</code> <span>DQN</span></td><td>Deep Q-network trained offline on the sweep; sees CPU load, free RAM and prompt length.</td></tr>
    <tr><td><code>bandit_adaptive</code> <span>online bandit</span></td><td>Starts from the sweep's estimates and keeps learning from each segment's measured speed.</td></tr>
    </tbody>
  </table>
</div>""",
              f"""Model: {f('cfg.model')}. An arm is a thread count, or "balanced": two layer stages with their own budgets.
Switching arms between tokens never changes the generated tokens.
Four policies decide; the fixed arms are the baselines."""))

    P.append(("How we tested", f"""
<h2>Five load patterns, every policy on the same prompts.</h2>
<ul class="lede protocol">
  <li>{f('plan.n_prompts')} prompts × {f('plan.n_scenarios')} scenarios × {f('plan.n_conditions')} conditions = {f('plan.total_runs')} runs of {f('plan.max_new_tokens')} tokens ({f('main.n_runs')} completed).</li>
  <li>Paired and randomised: each prompt-and-load context runs every condition, in a seeded random order.</li>
  <li>Plan fixed before running: {f('plan.n_tests')} pre-registered tests, Holm correction.</li>
</ul>
<div class="visual"><div class="multiples" id="multiples"></div></div>""",
              f"""Each small chart is one scenario: how many busy-loop workers run while the {f('plan.max_new_tokens')} tokens are generated.
Spike, relief and ramp change the load in the middle of a request; that is where adapting can help.
The plan and the tests were written down and committed before the main run."""))

    P.append(("Inside one request", f"""
<h2>Inside one request: the lookup table changes gear when load arrives.</h2>
<div class="tl-controls">
  <div class="seg" role="tablist">
    <button role="tab" data-tl="ramp" aria-selected="true">Ramp</button>
    <button role="tab" data-tl="spike" aria-selected="false">Spike</button>
  </div>
  <button class="play" id="tl-play">▶ Play</button>
  <span class="readout" id="tl-readout"></span>
</div>
<p class="lede tl-cap" data-tl="ramp">Ramp scenario, prompt {f('tl.ramp.prompt')} (median rule_lookup latency of its scenario):
<span class="ours-ink">rule_lookup {f('tl.ramp.rule_lookup')} s</span>, balanced {f('tl.ramp.balanced')} s, rl_adaptive {f('tl.ramp.rl_adaptive')} s, static {f('tl.ramp.static')} s.</p>
<p class="lede tl-cap" data-tl="spike" hidden>Spike scenario, prompt {f('tl.spike.prompt')} (median rule_lookup latency of its scenario):
<span class="ours-ink">rule_lookup {f('tl.spike.rule_lookup')} s</span>, balanced {f('tl.spike.balanced')} s, rl_adaptive {f('tl.spike.rl_adaptive')} s, static {f('tl.spike.static')} s.</p>
<div class="visual"><div class="chart" id="ch-timeline"></div></div>""",
              """Shaded bands show which arm rule_lookup was running; dashed lines are the moments the background load changed.
Press Play: the replay runs the real per-token times, sped up, so the lines race each other.
static uses all physical cores and stalls once the load arrives; rule_lookup drops to fewer threads."""))

    P.append(("Main result", f"""
<h2>Measuring the machine beats every fixed setting.</h2>
<p class="lede">Against the best fixed arm in each scenario (picked afterwards, with hindsight), rule_lookup was {f('bf.min_pct')}% to {f('bf.max_pct')}% faster on average.</p>
<div class="visual split">
  <div class="bignum"><span class="n-big">{f('bf.max_pct')}%</span>
    <span>faster under a <b>{f('bf.best_scen')}</b> than <code>{f('bf.best_arm')}</code>, the best fixed arm there,
    and faster in {f('bf.best_faster')}/{f('bf.n')} contexts.</span></div>
  <div class="chart" id="ch-best_fixed"></div>
</div>""",
              f"""This is the hard comparison: the best fixed arm is chosen after seeing the results, which no real system could do.
The gain is largest when load changes mid-request (spike, relief) and close to zero on an idle machine.
Labels show which fixed arm was best and in how many of the {f('bf.n')} contexts rule_lookup was faster."""))

    P.append(("RL vs measured", f"""
<h2>The DQN did not generalise.</h2>
<p class="lede">On held-out prompts the DQN picked the best arm in {f('dqn.min')}–{f('dqn.max')}% of contexts, the lookup table in {f('dqn.rule_min')}–{f('dqn.rule_max')}%.
Online, rl_adaptive was slower than rule_lookup in {f('rl.n_slower')} of {f('rl.n')} scenarios ({f('rl.min')}% to {f('rl.max')}%), {f('rl.n_sig')} of them Holm-significant.</p>
<div class="visual pair"><div class="chart" id="ch-dqn"></div><div class="chart" id="ch-rl_slow"></div></div>
<p class="foot">This is not evidence that learning-based scheduling loses in general.</p>""",
              """The DQN and the lookup table were built from the same load sweep.
Offline check is leave-one-prompt-out: train on two prompts, test on the third.
With this little data, a small measured table was easier to get right than a learned policy."""))

    P.append(("Is it real?", f"""
<h2>Is it real? {f('tests.n_sig')} of {f('tests.n')} pre-registered tests say yes.</h2>
<p class="lede">Paired t-tests, Holm-corrected over all {f('tests.n')}; all {f('tests.n_sig_favour')} significant results favour rule_lookup.
With paging-flagged runs removed, {f('sens.n_sig')} of {f('tests.n')} stay significant and {f('sens.same_dir')} of {f('tests.n')} effects keep their direction.</p>
<div class="visual"><div class="chart" id="ch-forest"></div></div>""",
              """Filled dots are Holm-significant; open dots are not. Right of zero means rule_lookup was faster.
The sensitivity check drops runs where Windows paged memory during generation, so n shrinks and some tests lose power.
The direction of every effect survives that check."""))

    P.append(("How close to perfect?", f"""
<h2>How close to perfect? rule_lookup stays within {f('or.rule_max')} s of an oracle.</h2>
<p class="lede">The oracle picks, with hindsight, the fastest of the four fixed arms for every {f('plan.interval')}-token segment. Regret = policy time − oracle time per request.
Negative values are possible: rule_lookup can use thread counts the oracle's four fixed arms don't include (it does, in <code>{f('or.neg_scen')}</code>: {f('or.neg_val')} s).</p>
<div class="visual"><div class="chart" id="ch-oracle"></div></div>""",
              f"""Darker cells are further from perfect; the colour scale is logarithmic so small regrets stay visible.
The DQN's worst case is {f('or.rl_max')} s per request, in {f('or.rl_max_scen')}.
The bandit comes close to rule_lookup because it starts from the same measurements."""))

    P.append(("Limits and next steps", f"""
<h2>What this does not show, and what comes next.</h2>
<div class="visual split2">
  <div><h3>Limits</h3><ul class="lede-list">
    <li>One laptop ({f('machine.physical')} cores, {f('machine.logical')} logical CPUs) and one model ({f('cfg.model')}).</li>
    <li>Synthetic busy-loop load, not real co-located programs.</li>
    <li>n = {f('plan.n_prompts')} contexts per scenario.</li>
    <li>{f('main.paging_n')} of {f('main.n_runs')} main-run runs were paging-flagged (kept in the primary analysis; checked in the sensitivity analysis).</li>
    <li>Power settings changed on 17 Oct: {f('main.runs_first')} runs before, {f('main.runs_later')} after.</li>
    <li>"Partitioning" here means layer stages with their own thread budgets, run one after another on one CPU.</li>
  </ul></div>
  <div><h3>Next</h3><ul class="lede-list">
    <li>Split the model across CPU and GPU (RTX 3050).</li>
    <li>Real co-located workloads instead of busy loops.</li>
    <li>More prompts and more machines.</li>
  </ul></div>
</div>""",
              """These are the honest boundaries: one machine, one model, artificial load.
The power-setting change is documented and the by-session comparison shows fixed arms at about the same speed.
The obvious next step is a real CPU+GPU split, which is what "partitioning" usually means."""))

    P.append(("Summary", f"""
<h2>Summary</h2>
<ol class="summary">
  <li><b>Built:</b> a decode loop for {f('cfg.model')} that re-picks its CPU thread count every {f('plan.interval')} tokens without changing the output.</li>
  <li><b>Result:</b> a lookup table measured on this laptop beat the best fixed setting in every scenario ({f('bf.min_pct')}% to {f('bf.max_pct')}%) and won {f('tests.n_sig_favour')} of {f('tests.n')} pre-registered tests.</li>
  <li><b>Learned:</b> measure first. The offline DQN was {f('rl.min')}% to {f('rl.max')}% slower online than the simple measured table.</li>
</ol>
<p class="repo"><a href="{REPO_URL}">{REPO_URL.replace('https://', '')}</a></p>""",
              """Three things: we built it, it works on this machine, and the simple measured approach beat the learned one here.
Everything on these slides is generated from the committed CSVs and checked by a script.
Questions."""))
    return P


def page(data):
    sections = []
    for i, (label, body, notes) in enumerate(panels(), start=1):
        sections.append(f'<section class="panel" id="p{i}" data-label="{html.escape(label)}">\n{body}\n'
                        f'<aside class="notes"><b>Notes</b><p>{notes.strip().replace(chr(10), "<br>")}</p></aside>\n'
                        f'</section>')
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return (TEMPLATE.replace("{{PLOTLY}}", get_plotlyjs())
            .replace("{{DATA}}", payload)
            .replace("{{PANELS}}", "\n".join(sections)))


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Load-Adaptive LLM Threads</title>
<meta name="description" content="Results dashboard and presentation: load-adaptive thread scheduling for LLM inference on a CPU. Generated by experiments/build_dashboard.py.">
<style>
:root{
  --sans:"Inter","Segoe UI",system-ui,-apple-system,Roboto,"Helvetica Neue",Arial,sans-serif;
  --mono:"JetBrains Mono","Cascadia Mono","Cascadia Code",Consolas,"SFMono-Regular",Menlo,monospace;
  --bg:#f7f7f5; --surface:#ffffff; --ink:#16191d; --ink2:#454d57; --muted:#6c747e;
  --grid:#e5e7ea; --line:#cfd4da; --accent:#1d64c8; --accent-soft:rgba(29,100,200,.10);
  --red:#cc2f2a; --g1:#5f6873; --g2:#8f98a2; --g3:#b9c0c7; --heat0:#f2f3f5; --heat1:#3b4350;
  color-scheme:light;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#0f1215; --surface:#171b20; --ink:#eef1f4; --ink2:#bac1c9; --muted:#8c959f;
    --grid:#272d34; --line:#39414a; --accent:#5ea3f7; --accent-soft:rgba(94,163,247,.14);
    --red:#ff6a61; --g1:#aab2bb; --g2:#7f8892; --g3:#56606a; --heat0:#20252b; --heat1:#c9d0d8;
    color-scheme:dark;
  }
}
:root[data-theme="dark"]{
  --bg:#0f1215; --surface:#171b20; --ink:#eef1f4; --ink2:#bac1c9; --muted:#8c959f;
  --grid:#272d34; --line:#39414a; --accent:#5ea3f7; --accent-soft:rgba(94,163,247,.14);
  --red:#ff6a61; --g1:#aab2bb; --g2:#7f8892; --g3:#56606a; --heat0:#20252b; --heat1:#c9d0d8;
  color-scheme:dark;
}
*{box-sizing:border-box}
html,body{margin:0}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.45;-webkit-font-smoothing:antialiased}
code{font-family:var(--mono);font-size:.9em;background:var(--accent-soft);padding:.05em .3em;border-radius:4px;color:var(--ink)}
.n,.n-big,.readout{font-family:var(--mono);font-variant-numeric:tabular-nums}
a{color:var(--accent)}
h1,h2,h3{margin:0;font-weight:650;letter-spacing:-.01em;text-wrap:balance}
.callout-red{color:var(--red)}
.ours-ink{color:var(--accent);font-weight:600}
[hidden]{display:none!important}

/* toolbar */
.toolbar{position:fixed;top:10px;right:12px;z-index:30;display:flex;gap:6px}
.toolbar button,.seg button,.play{font:500 13px var(--sans);color:var(--ink2);background:var(--surface);border:1px solid var(--line);border-radius:7px;padding:6px 10px;cursor:pointer}
.toolbar button:hover,.seg button:hover,.play:hover{color:var(--ink);border-color:var(--muted)}
.toolbar button[aria-pressed="true"]{color:var(--accent);border-color:var(--accent)}
button:focus-visible,a:focus-visible,summary:focus-visible,th:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
kbd{font-family:var(--mono);font-size:.8em;opacity:.7;margin-left:4px}

/* explore mode */
.tabs{position:sticky;top:0;z-index:20;background:color-mix(in srgb,var(--bg) 92%,transparent);backdrop-filter:blur(6px);border-bottom:1px solid var(--grid);display:flex;gap:2px;overflow-x:auto;padding:8px 16px 0 max(16px,calc(50vw - 640px));margin-right:370px;scrollbar-width:thin}
.tabs a{white-space:nowrap;text-decoration:none;color:var(--muted);font-size:14px;padding:8px 10px;border-bottom:2px solid transparent}
.tabs a .tn{font-family:var(--mono);margin-right:6px;opacity:.7}
.tabs a.on{color:var(--ink);border-bottom-color:var(--accent)}
body.explore .panel{padding:56px max(16px,calc(50vw - 640px)) 48px;border-bottom:1px solid var(--grid);scroll-margin-top:44px}
body.explore h1{font-size:clamp(30px,4.4vw,52px)}
body.explore h2{font-size:clamp(22px,2.6vw,30px);margin-bottom:10px}
body.explore .lede,body.explore .lede-list{font-size:17px;color:var(--ink2);max-width:72ch;margin:0 0 18px}
body.explore .chart{height:440px;min-width:0}
body.explore #ch-forest{height:640px}
body.explore #ch-oracle{height:360px}
body.explore .multiples .chart{height:210px}
body.explore .present-only{display:none}
.panel .chart-wrap{min-width:0}
.visual.col > .multiples{flex:1;min-height:0}
body.present .chart-wrap{height:100%}
body.present .chart-wrap > .chart{flex:1 1 0;min-height:0}
.src{font-size:12px;color:var(--muted);margin:4px 0 0;font-family:var(--mono)}
details.data{margin-top:8px;font-size:14px}
details.data summary{cursor:pointer;color:var(--muted)}
.tbl-wrap{overflow-x:auto;max-height:360px;overflow-y:auto;border:1px solid var(--grid);border-radius:8px;margin-top:6px}
table.sortable{border-collapse:collapse;width:100%;font-size:13px}
table.sortable th,table.sortable td{padding:5px 10px;text-align:right;border-bottom:1px solid var(--grid);white-space:nowrap}
table.sortable th:first-child,table.sortable td:first-child{text-align:left}
table.sortable th{position:sticky;top:0;background:var(--surface);cursor:pointer;user-select:none;color:var(--ink2);font-weight:600}
table.sortable th[aria-sort="ascending"]::after{content:" ▲";font-size:.7em}
table.sortable th[aria-sort="descending"]::after{content:" ▼";font-size:.7em}
table.sortable td{font-family:var(--mono)}

/* shared panel content */
.hero{display:flex;flex-direction:column;justify-content:center;gap:10px;min-height:60vh}
.kicker{color:var(--accent);font-weight:600;letter-spacing:.06em;text-transform:uppercase;font-size:14px;margin:0}
.subtitle{font-size:clamp(18px,2vw,28px);color:var(--ink2);margin:0 0 18px}
.team{font-size:clamp(17px,1.6vw,24px);margin:0}
.guide{color:var(--ink2);margin:0}
.hint{color:var(--muted);font-size:15px;margin-top:28px}
.visual{display:flex;gap:24px;min-width:0}
.visual > .chart{flex:1}
.visual.col{flex-direction:column}
.visual.split{display:grid;grid-template-columns:minmax(220px,1fr) 2.4fr;align-items:center}
.visual.pair{display:grid;grid-template-columns:1fr 1fr}
.visual.split2{display:grid;grid-template-columns:1.5fr 1fr;align-items:start}
.multiples{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;width:100%}
.bignum{display:flex;flex-direction:column;gap:8px;color:var(--ink2)}
.n-big{font-size:clamp(56px,7vw,120px);font-weight:700;color:var(--accent);line-height:1}
.flow{display:flex;align-items:stretch;gap:8px;flex-wrap:wrap}
.flow .box .bh{display:block}
.flow .box{flex:1 1 140px;border:1.5px solid var(--line);border-radius:10px;padding:12px 14px;background:var(--surface);font-weight:600;display:flex;flex-direction:column;justify-content:center;gap:4px}
.flow .box small{font-weight:400;color:var(--muted);font-size:.78em;line-height:1.3}
.flow .box.accent{border-color:var(--accent);color:var(--accent);background:var(--accent-soft)}
.flow .arrow{align-self:center;color:var(--muted);font-size:1.4em}
table.policies{border-collapse:collapse;width:100%;margin-top:8px}
table.policies th,table.policies td{text-align:left;padding:6px 12px;border-bottom:1px solid var(--grid);vertical-align:top}
table.policies th{color:var(--muted);font-weight:600;font-size:.85em}
table.policies td span{color:var(--muted);font-size:.85em;white-space:nowrap}
table.policies td:first-child{white-space:nowrap}
table.policies tr.ours td:first-child{box-shadow:inset 3px 0 0 var(--accent)}
.protocol{padding-left:1.1em}
.protocol li{margin:2px 0}
.tl-controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:6px 0 10px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.seg button{border:0;border-radius:0}
.seg button[aria-selected="true"]{background:var(--accent);color:#fff}
.play{color:var(--accent);border-color:var(--accent);font-weight:600}
.readout{color:var(--muted);font-size:13px;flex-basis:100%;min-height:1.5em}
.foot{color:var(--muted);font-style:italic;margin:8px 0 0}
.lede-list{padding-left:1.1em}
.lede-list li{margin:4px 0}
h3{font-size:1.1em;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px}
.summary{font-size:clamp(18px,2vw,30px);line-height:1.4;padding-left:1.2em;max-width:60ch}
.summary li{margin:0 0 .6em}
.repo{font-family:var(--mono);font-size:clamp(14px,1.3vw,20px);overflow-wrap:anywhere}
.notes{display:none;border-left:3px solid var(--accent);background:var(--surface);padding:10px 14px;margin-top:16px;color:var(--ink2)}
.notes p{margin:4px 0 0}
body.show-notes .notes{display:block}
.fade-in{animation:fadein .7s ease-out both}
@keyframes fadein{from{opacity:0}to{opacity:1}}

/* present mode */
body.present{overflow:hidden}
body.present .tabs,body.present .explore-only,body.present details.data{display:none}
body.present .panel{display:none}
body.present .panel.active{display:flex;flex-direction:column;height:100vh;height:100dvh;padding:44px clamp(28px,4vw,72px) 46px;gap:6px}
body.present h1{font-size:clamp(48px,5.4vw,96px);line-height:1.08}
body.present h2{font-size:clamp(40px,3.1vw,60px);line-height:1.12;margin-bottom:4px}
body.present .lede,body.present .lede-list,body.present .policies,body.present .flow,body.present .bignum > span:not(.n-big){font-size:clamp(22px,1.45vw,28px)}
body.present .flow .box{padding:10px 14px}
.n-big .n{font-size:inherit}
body.present .lede{color:var(--ink2);margin:0 0 6px;max-width:80ch}
body.present .visual{flex:1;min-height:0}
body.present .visual > .chart,body.present .visual .chart{min-height:0;height:100%}
body.present .visual.split,body.present .visual.pair{grid-template-rows:minmax(0,1fr)}
body.present .multiples{grid-template-rows:minmax(0,1fr);height:100%}
body.present .src{font-size:14px}
body.present .hero{min-height:0;flex:1}
body.present .hint{font-size:22px}
body.present .team{font-size:clamp(24px,2vw,34px)}
body.present .guide{font-size:22px}
body.present .subtitle{font-size:clamp(24px,2.2vw,38px)}
body.present .summary{font-size:clamp(24px,2.1vw,36px)}
body.present .foot{font-size:18px}
body.present .tl-controls button{font-size:18px;padding:8px 14px}
body.present .readout{font-size:18px}
body.present.show-notes .panel.active .notes{position:fixed;left:0;right:0;bottom:0;z-index:25;margin:0;font-size:20px;padding:14px clamp(28px,4vw,72px) 30px;border-left:0;border-top:3px solid var(--accent);box-shadow:0 -8px 24px rgba(0,0,0,.12)}
body.present .toolbar{opacity:.35;transition:opacity .2s}
body.present .toolbar:hover,body.present .toolbar:focus-within{opacity:1}
#progress{display:none}
body.present #progress{display:block;position:fixed;left:0;right:0;bottom:0;height:6px;background:var(--grid);z-index:26}
#progress .bar{height:100%;background:var(--accent);transition:width .3s}
#progress .label{position:fixed;right:14px;bottom:12px;font:14px var(--mono);color:var(--muted)}

@media (max-width: 900px){
  .visual.split,.visual.pair,.visual.split2{grid-template-columns:1fr}
  .multiples{grid-template-columns:repeat(auto-fill,minmax(150px,1fr))}
  .flow .arrow{transform:rotate(90deg);flex-basis:100%;text-align:center}
  .tabs{margin-right:0;padding-top:52px}
}
@media (prefers-reduced-motion: reduce){
  .fade-in{animation:none}
  #progress .bar{transition:none}
}
</style>
</head>
<body class="explore">
<div class="toolbar" role="toolbar" aria-label="Dashboard controls">
  <button id="btn-mode" title="Toggle present / explore (P)">Present<kbd>P</kbd></button>
  <button id="btn-notes" aria-pressed="false" title="Speaker notes (N)">Notes<kbd>N</kbd></button>
  <button id="btn-theme" title="Theme: auto / light / dark">Theme: auto</button>
  <button id="btn-fs" title="Fullscreen (F)">Fullscreen<kbd>F</kbd></button>
</div>
<nav class="tabs" id="tabs" aria-label="Panels"></nav>
<main id="panels">
{{PANELS}}
</main>
<div id="progress" aria-hidden="true"><div class="bar"></div><span class="label"></span></div>
<script>{{PLOTLY}}</script>
<script id="dash-data" type="application/json">{{DATA}}</script>
<script>
(function(){
"use strict";
const DATA = JSON.parse(document.getElementById('dash-data').textContent);
const PANELS = Array.from(document.querySelectorAll('.panel'));
const N = PANELS.length;
const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const state = {mode:'explore', idx:0, theme:'auto', notes:false, tl:'ramp', rendered:new Set(), animated:new Set(), playing:null};
const store = {
  get(k){ try { return window.localStorage.getItem('dash.'+k); } catch(e){ return null; } },
  set(k,v){ try { window.localStorage.setItem('dash.'+k, v); } catch(e){} }
};
const lab = s => String(s).replace(/_/g,' ');
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const present = () => state.mode === 'present';
function sizes(){ return present() ? {base:18, tick:16, title:21, hover:16, legend:17} : {base:13, tick:12, title:15, hover:13, legend:13}; }
function merge(a,b){ for (const k in b){ if (b[k] && typeof b[k]==='object' && !Array.isArray(b[k]) && a[k] && typeof a[k]==='object') merge(a[k],b[k]); else a[k]=b[k]; } return a; }
const GREYS = () => [css('--g1'), css('--g2'), css('--g3')];

function baseLayout(id, extra){
  const s = sizes(), m = DATA.meta[id];
  const axis = t => ({title:{text:t, font:{size:s.base, color:css('--ink2')}, standoff:8}, gridcolor:css('--grid'), linecolor:css('--line'),
                      zeroline:false, tickfont:{size:s.tick, color:css('--ink2'), family:css('--mono')}, automargin:true});
  return merge({
    title:{text:m.title, font:{size:s.title, color:css('--ink'), family:css('--sans')}, x:0, xanchor:'left', xref:'paper', y:.98, yanchor:'top'},
    font:{family:css('--sans'), size:s.base, color:css('--ink2')},
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    margin:{l:10, r:16, t:s.title*2.4, b:10},
    xaxis:axis(m.x), yaxis:axis(m.y),
    hovermode: present() ? false : 'closest',
    hoverlabel:{font:{family:css('--mono'), size:s.hover}, bgcolor:css('--surface'), bordercolor:css('--line')},
    legend:{font:{size:s.legend, color:css('--ink2')}, orientation:'h', x:0, y:-0.16, yanchor:'top'},
    showlegend:false
  }, extra || {});
}
function fmt(v, dp){ return Number(v).toFixed(dp); }

/* ---------------- charts ---------------- */
const CHARTS = {};

CHARTS.sweep = {panel:'p2', anim:'reveal', build(){
  const d = DATA.sweep, red = css('--red'), g = GREYS();
  const style = {threads_2:[g[0],'solid'], threads_3:[g[1],'dot'], balanced:[g[0],'dash'], threads_4:[g[1],'dashdot'], threads_6:[g[2],'longdash']};
  const data = d.arms.map(a => {
    const cliff = a === 'threads_10', st = style[a] || [g[1],'solid'];
    return {x:d.workers, y:d.mean[a], name:a, type:'scatter', mode:'lines+markers',
      line:{color: cliff ? red : st[0], width: cliff ? 3.5 : 2, dash: cliff ? 'solid' : st[1]},
      marker:{size: cliff ? 9 : 7, color: cliff ? red : st[0]},
      hovertemplate:`<b>${a}</b><br>%{x} workers<br>%{y:.4f} s/token<extra></extra>`};
  });
  const c = d.cliff, y10 = d.mean.threads_10[d.workers.indexOf(c.workers)];
  const layout = baseLayout('sweep', {yaxis:{type:'log', tickmode:'array', tickvals:[0.05,0.1,0.2,0.5,1,2,5], ticktext:['0.05','0.1','0.2','0.5','1','2','5']}, showlegend:true,
    annotations:[{x:c.workers, y:Math.log10(y10), text:'<b>the cliff</b>', showarrow:true, arrowcolor:red, arrowwidth:2, ax:-90, ay:10,
                  font:{color:red, size:sizes().base+2}, bgcolor:css('--bg')}]});
  const rows = d.workers.map((w,i) => [w].concat(d.arms.map(a => d.mean[a][i])));
  return {data, layout, table:{cols:['workers'].concat(d.arms), rows, dp:[0].concat(d.arms.map(()=>5))}};
}};

DATA.scenario_order.forEach(s => {
  CHARTS['scen_'+s] = {panel:'p4', anim:'reveal', small:true, build(){
    const pts = DATA.scenarios[s], T = DATA.max_new_tokens;
    const x = pts.map(p=>p[0]).concat([T]), y = pts.map(p=>p[1]).concat([pts[pts.length-1][1]]);
    const maxW = Math.max(...Object.values(DATA.scenarios).flat().map(p=>p[1]));
    const s2 = sizes();
    const data = [{x, y, type:'scatter', mode:'lines', line:{shape:'hv', color:css('--g1'), width:2.5}, fill:'tozeroy', fillcolor:css('--accent-soft'),
                   hovertemplate:'from token %{x}: %{y} workers<extra></extra>'}];
    const layout = baseLayout('scen_'+s, {xaxis:{range:[0,T], tickmode:'array', tickvals:[0,T/2,T]}, yaxis:{range:[0,maxW*1.15], tickmode:'array', tickvals:[0,maxW/2,maxW]},
      margin:{l:6, r:8, t:s2.title*2, b:6}});
    return {data, layout, table:{cols:['from token','workers'], rows:pts.map(p=>[p[0],p[1]]), dp:[0,0]}};
  }};
});

const TL_STYLE = () => ({rule_lookup:[css('--accent'),'solid',3.5], balanced:[css('--g1'),'dash',2.2], rl_adaptive:[css('--g2'),'dot',2.4], static:[css('--g1'),'solid',1.6]});
function timelineSpec(slices){
  const d = DATA.timeline[state.tl], st = TL_STYLE(), s = sizes();
  const conds = Object.keys(d.cum);
  const data = conds.map(c => {
    const k = slices ? slices[c] : d.steps.length;
    return {x:d.steps.slice(0,k), y:d.cum[c].slice(0,k), name:c, type:'scatter', mode:'lines',
      line:{color:st[c][0], dash:st[c][1], width:st[c][2]},
      hovertemplate:`<b>${c}</b><br>token %{x}<br>%{y:.2f} s<extra></extra>`};
  });
  const maxY = Math.max(...conds.map(c => d.cum[c][d.cum[c].length-1]));
  const armOp = {threads_2:.30, threads_3:.24, balanced:.18, threads_4:.13, threads_6:.08, threads_10:.04};
  const shapes = d.bands.map(b => ({type:'rect', xref:'x', yref:'paper', x0:b[0]-.5, x1:b[1]+.5, y0:0, y1:1, layer:'below', line:{width:0},
                                    fillcolor:css('--accent'), opacity:armOp[b[2]] ?? .1}));
  const ann = d.bands.filter(b => b[1]-b[0] >= 14).map(b => ({x:(b[0]+b[1])/2, y:1, xref:'x', yref:'paper', yanchor:'top', showarrow:false,
                                    text:b[2], font:{size:s.tick, color:css('--accent'), family:css('--mono')}}));
  DATA.scenarios[state.tl].slice(1).forEach(p => {
    shapes.push({type:'line', xref:'x', yref:'paper', x0:p[0], x1:p[0], y0:0, y1:1, line:{color:css('--muted'), width:1.5, dash:'dash'}});
    ann.push({x:p[0], y:.86, xref:'x', yref:'paper', xanchor:'left', showarrow:false, text:` load → ${p[1]} workers`, font:{size:s.tick, color:css('--ink2')}, bgcolor:css('--bg')});
  });
  const layout = baseLayout('timeline', {shapes, annotations:ann, showlegend:true,
    legend:{orientation:'v', x:.01, xanchor:'left', y:.8, yanchor:'top', bgcolor:css('--bg'), bordercolor:css('--grid'), borderwidth:1},
    xaxis:{range:[-2, DATA.max_new_tokens + 2], autorange:false}, yaxis:{range:[0, maxY*1.04], autorange:false}});
  const rows = d.steps.map((t,i) => [t].concat(conds.map(c => d.cum[c][i])));
  return {data, layout, table:{cols:['token'].concat(conds.map(c=>c+' (s)')), rows, dp:[0].concat(conds.map(()=>3))}};
}
CHARTS.timeline = {panel:'p5', anim:'reveal', build(){ return timelineSpec(null); }};

CHARTS.best_fixed = {panel:'p6', anim:'grow', build(){
  const rows = DATA.best_fixed.slice().reverse(), s = sizes();
  const maxX = Math.max(...rows.map(r=>r.faster_pct));
  const data = [{type:'bar', orientation:'h', y:rows.map(r=>lab(r.scenario)), x:rows.map(r=>r.faster_pct),
    marker:{color:css('--accent')}, width:.6,
    text:rows.map(r=>`vs ${r.arm} · faster in ${r.contexts_faster}/${r.n}`), textposition:'outside', cliponaxis:false,
    textfont:{size:s.tick, color:css('--ink2'), family:css('--mono')},
    customdata:rows.map(r=>[r.rule_s, r.fixed_s, r.arm]),
    hovertemplate:'<b>%{y}</b><br>%{x:.2f}% faster than %{customdata[2]}<br>rule_lookup %{customdata[0]:.2f} s vs %{customdata[1]:.2f} s<extra></extra>'}];
  const layout = baseLayout('best_fixed', {xaxis:{range:[0, maxX*2.1], ticksuffix:'%'}, bargap:.35});
  return {data, layout, table:{cols:['scenario','best fixed arm','faster (%)','contexts faster','n','rule_lookup (s)','best fixed (s)'],
    rows:DATA.best_fixed.map(r=>[lab(r.scenario), r.arm, r.faster_pct, r.contexts_faster, r.n, r.rule_s, r.fixed_s]), dp:[null,null,2,0,0,3,3]}};
}};

CHARTS.dqn = {panel:'p7', anim:'grow', build(){
  const d = DATA.dqn, g = GREYS();
  const color = {rule_lookup_fold:css('--accent'), dqn_v7:g[0], cores_heuristic:g[2]};
  const name = {rule_lookup_fold:'rule_lookup (fold)', dqn_v7:'DQN', cores_heuristic:'cores_heuristic'};
  const xs = d.folds.map(f => 'prompt ' + f);
  const data = Object.keys(d.pct).map(p => ({type:'bar', x:xs, y:d.pct[p], name:name[p], marker:{color:color[p]},
    hovertemplate:`<b>${name[p]}</b><br>held-out %{x}<br>%{y:.1f}% best arm<extra></extra>`}));
  const layout = baseLayout('dqn', {barmode:'group', bargap:.25, bargroupgap:.08, showlegend:true, yaxis:{range:[0,100], ticksuffix:'%'}});
  return {data, layout, table:{cols:['held-out prompt'].concat(Object.keys(d.pct).map(p=>name[p]+' (%)')),
    rows:d.folds.map((f,i)=>[f].concat(Object.keys(d.pct).map(p=>d.pct[p][i]))), dp:[0,1,1,1]}};
}};

CHARTS.rl_slow = {panel:'p7', anim:'grow', build(){
  const rows = DATA.rl_vs_rule.slice().reverse(), s = sizes();
  const maxX = Math.max(...rows.map(r=>r.slower_pct));
  const data = [{type:'bar', orientation:'h', y:rows.map(r=>lab(r.scenario)), x:rows.map(r=>r.slower_pct), width:.6,
    marker:{color:rows.map(r=> r.holm ? css('--g1') : css('--g3'))},
    text:rows.map(r=> r.holm ? 'Holm-significant' : 'not significant'), textposition:'outside', cliponaxis:false,
    textfont:{size:s.tick, color:css('--ink2')},
    customdata:rows.map(r=>r.p_holm),
    hovertemplate:'<b>%{y}</b><br>%{x:.2f}% slower<br>Holm p = %{customdata:.4f}<extra></extra>'}];
  const layout = baseLayout('rl_slow', {xaxis:{range:[0, maxX*1.75], ticksuffix:'%'}});
  return {data, layout, table:{cols:['scenario','slower (%)','Holm p','significant'],
    rows:DATA.rl_vs_rule.map(r=>[lab(r.scenario), r.slower_pct, r.p_holm, r.holm ? 'yes':'no']), dp:[null,2,4,null]}};
}};

CHARTS.forest = {panel:'p8', anim:'reveal', build(){
  const order = ['balanced','threads_2','cores_heuristic','rl_adaptive'];
  const rows = DATA.tests.slice().sort((a,b) => order.indexOf(a.other)-order.indexOf(b.other) || DATA.scenario_order.indexOf(a.scenario)-DATA.scenario_order.indexOf(b.scenario));
  const label = r => `vs ${r.other} · ${lab(r.scenario)}`;
  const cats = rows.map(label).reverse();
  const acc = css('--accent'), s = sizes();
  const mk = (sig) => { const rs = rows.filter(r => r.holm === sig); return {type:'scatter', mode:'markers', name: sig ? 'Holm-significant' : 'not significant',
    y:rs.map(label), x:rs.map(r=>r.saved_s),
    error_x:{type:'data', symmetric:false, array:rs.map(r=>r.hi-r.saved_s), arrayminus:rs.map(r=>r.saved_s-r.lo), color:acc, thickness:1.6, width:0},
    marker:{size: present() ? 13 : 10, color: sig ? acc : css('--bg'), line:{color:acc, width:2}, symbol:'circle'},
    customdata:rs.map(r=>[r.lo, r.hi, r.p_holm, r.n]),
    hovertemplate:'<b>%{y}</b><br>%{x:.3f} s saved<br>95% CI %{customdata[0]:.3f} to %{customdata[1]:.3f}<br>Holm p = %{customdata[2]:.4g}, n = %{customdata[3]}<extra></extra>'}; };
  const data = [mk(true), mk(false)];
  const layout = baseLayout('forest', {showlegend:true,
    yaxis:{type:'category', categoryorder:'array', categoryarray:cats, tickmode:'array', tickvals:cats, tickfont:{family:css('--mono')}, gridcolor:css('--grid')},
    xaxis:{zeroline:true, zerolinecolor:css('--ink2'), zerolinewidth:1.5}});
  const sens = {}; DATA.tests_sens.forEach(r => sens[label(r)] = r);
  return {data, layout, table:{cols:['comparison','saved (s)','CI low','CI high','Holm p','significant','sensitivity saved (s)','sensitivity n','sensitivity significant'],
    rows:rows.map(r => { const q = sens[label(r)]; return [label(r), r.saved_s, r.lo, r.hi, r.p_holm, r.holm?'yes':'no', q.saved_s, q.n, q.holm?'yes':'no']; }),
    dp:[null,3,3,3,4,null,3,0,null]}};
}};

CHARTS.oracle = {panel:'p9', anim:'fade', build(){
  const d = DATA.oracle, sc = DATA.scenario_order, s = sizes();
  const tf = v => Math.sign(v)*Math.log10(1+Math.abs(v));
  const z = d.policies.map(p => d.regret[p].map(tf)), text = d.policies.map(p => d.regret[p].map(v => fmt(v,2)+' s'));
  const zmin = -0.3, zmax = Math.max(...z.flat());
  const mid = -zmin/(zmax-zmin);
  const data = [{type:'heatmap', x:sc.map(lab), y:d.policies, z, text, texttemplate:'%{text}', textfont:{size:s.tick+1, family:css('--mono')},
    zmin, zmax, showscale:false, xgap:3, ygap:3,
    colorscale:[[0,css('--accent')],[mid,css('--heat0')],[1,css('--heat1')]],
    customdata:d.policies.map(p => d.regret[p]),
    hovertemplate:'<b>%{y}</b> · %{x}<br>regret %{customdata:.3f} s<extra></extra>'}];
  const layout = baseLayout('oracle', {yaxis:{autorange:'reversed', tickfont:{family:css('--mono')}, gridcolor:'rgba(0,0,0,0)'},
    xaxis:{side:'bottom', gridcolor:'rgba(0,0,0,0)'}});
  return {data, layout, table:{cols:['scenario','oracle (s)'].concat(d.policies.map(p=>p+' regret (s)')),
    rows:sc.map((x,i)=>[lab(x), d.oracle_s[i]].concat(d.policies.map(p=>d.regret[p][i]))), dp:[null,3,3,3,3,3]}};
}};

/* ---------------- rendering ---------------- */
function plotConfig(){ return {responsive:true, displaylogo:false, displayModeBar: present() ? false : 'hover',
  modeBarButtonsToRemove:['select2d','lasso2d','autoScale2d','toggleSpikelines','hoverCompareCartesian','hoverClosestCartesian']}; }

function sliceData(data, frac){
  return data.map(t => {
    const c = Object.assign({}, t), len = (t.x || t.y || []).length, k = Math.max(1, Math.ceil(frac*len));
    ['x','y','text','customdata'].forEach(key => { if (Array.isArray(t[key])) c[key] = t[key].slice(0,k); });
    if (t.error_x) c.error_x = Object.assign({}, t.error_x, {array:t.error_x.array.slice(0,k), arrayminus:t.error_x.arrayminus.slice(0,k)});
    return c;
  });
}
function growData(data, frac){
  return data.map(t => { const c = Object.assign({}, t), key = t.orientation === 'h' ? 'x' : 'y'; c[key] = t[key].map(v => v*frac); return c; });
}
function fixedRanges(gd, layout){
  const L = JSON.parse(JSON.stringify(layout)), fl = gd._fullLayout;
  Object.keys(fl).filter(k => /^[xy]axis\d*$/.test(k)).forEach(k => {
    if (fl[k].type === 'category') return;
    L[k] = Object.assign(L[k] || {}, {range: fl[k].range.slice(), autorange:false});
  });
  return L;
}
function animate(gd, spec, kind){
  if (kind === 'fade'){ gd.classList.remove('fade-in'); void gd.offsetWidth; gd.classList.add('fade-in'); return; }
  const L = fixedRanges(gd, spec.layout), dur = kind === 'grow' ? 800 : 1100, t0 = performance.now();
  const token = gd._anim = (gd._anim || 0) + 1;
  const frame = now => {
    if (gd._anim !== token) return;
    const p = Math.min(1, (now - t0)/dur), e = 1 - Math.pow(1-p, 3);
    if (p < 1){ Plotly.react(gd, kind === 'grow' ? growData(spec.data, e) : sliceData(spec.data, e), L, plotConfig()); requestAnimationFrame(frame); }
    else Plotly.react(gd, spec.data, spec.layout, plotConfig());
  };
  requestAnimationFrame(frame);
}
function visible(el){ return !!(el && el.offsetParent !== null && el.clientWidth > 0); }
function renderChart(id){
  const el = document.getElementById('ch-'+id), c = CHARTS[id];
  if (!visible(el)) return;
  const spec = c.build();
  const first = !state.animated.has(id);
  state.animated.add(id); state.rendered.add(id);
  el._anim = (el._anim || 0) + 1;  // cancels a running intro animation
  Plotly.react(el, spec.data, spec.layout, plotConfig()).then(() => { if (first && !reduceMotion) animate(el, spec, c.anim); });
  ensureTable(id, spec.table);
}
function renderPanel(pid){ Object.keys(CHARTS).forEach(id => { if (CHARTS[id].panel === pid) renderChart(id); }); }
function rerenderVisible(){
  state.rendered.clear();
  if (present()) renderPanel(PANELS[state.idx].id);
  else PANELS.forEach(p => { const r = p.getBoundingClientRect(); if (r.bottom > 0 && r.top < innerHeight) renderPanel(p.id); });
}

/* chart frames: title lives in the plot; source + table under it */
function decorate(){
  const mult = document.getElementById('multiples');
  DATA.scenario_order.forEach(s => { const d = document.createElement('div'); d.className = 'chart'; d.id = 'ch-scen_'+s; mult.appendChild(d); });
  const srcAfter = new Map();
  Object.keys(CHARTS).forEach(id => {
    const el = document.getElementById('ch-'+id);
    if (CHARTS[id].small) return;
    const wrap = document.createElement('div'); wrap.className = 'chart-wrap'; wrap.style.cssText = 'display:flex;flex-direction:column;flex:1;min-height:0;min-width:0';
    el.parentNode.insertBefore(wrap, el); wrap.appendChild(el);
    const src = document.createElement('p'); src.className = 'src'; src.textContent = 'Source: ' + DATA.meta[id].src; wrap.appendChild(src);
    const det = document.createElement('details'); det.className = 'data'; det.id = 'tbl-'+id;
    det.innerHTML = '<summary>Data table (click a column to sort)</summary>'; wrap.appendChild(det);
  });
  const ms = document.createElement('p'); ms.className = 'src'; ms.textContent = 'Source: ' + DATA.meta['scen_'+DATA.scenario_order[0]].src;
  mult.parentNode.appendChild(ms);
  mult.parentNode.classList.add('col');
  const det = document.createElement('details'); det.className = 'data'; det.id = 'tbl-scen';
  det.innerHTML = '<summary>Data table (click a column to sort)</summary>'; mult.parentNode.appendChild(det);
}
const tables = new Set();
function ensureTable(id, t){
  if (!t) return;
  let host = document.getElementById('tbl-'+id), key = id;
  if (CHARTS[id].small){
    host = document.getElementById('tbl-scen'); key = 'scen';
    if (tables.has(key)) return;
    const rows = []; DATA.scenario_order.forEach(s => DATA.scenarios[s].forEach(p => rows.push([lab(s), p[0], p[1]])));
    t = {cols:['scenario','from token','workers'], rows, dp:[null,0,0]};
  }
  if (!host || tables.has(key)) return;
  tables.add(key);
  const wrap = document.createElement('div'); wrap.className = 'tbl-wrap';
  const tbl = document.createElement('table'); tbl.className = 'sortable';
  const fmtCell = (v, dp) => (typeof v === 'number' && dp != null) ? fmt(v, dp) : String(v);
  tbl.innerHTML = '<thead><tr>' + t.cols.map(c => `<th tabindex="0" scope="col">${c}</th>`).join('') + '</tr></thead>';
  const tb = document.createElement('tbody'); tbl.appendChild(tb);
  const draw = rows => { tb.innerHTML = rows.map(r => '<tr>' + r.map((v,i) => `<td>${fmtCell(v, t.dp[i])}</td>`).join('') + '</tr>').join(''); };
  draw(t.rows);
  tbl.querySelectorAll('th').forEach((th, i) => {
    const sort = () => {
      const dir = th.getAttribute('aria-sort') === 'ascending' ? 'descending' : 'ascending';
      tbl.querySelectorAll('th').forEach(h => h.removeAttribute('aria-sort')); th.setAttribute('aria-sort', dir);
      const m = dir === 'ascending' ? 1 : -1;
      draw(t.rows.slice().sort((a,b) => (typeof a[i] === 'number' && typeof b[i] === 'number') ? m*(a[i]-b[i]) : m*String(a[i]).localeCompare(String(b[i]))));
    };
    th.addEventListener('click', sort); th.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' '){ e.preventDefault(); e.stopPropagation(); sort(); } });
  });
  wrap.appendChild(tbl); host.appendChild(wrap);
}

/* ---------------- timeline: tabs and play ---------------- */
function setTimeline(s){
  stopPlay();
  state.tl = s;
  document.querySelectorAll('[data-tl]').forEach(el => {
    if (el.tagName === 'BUTTON') el.setAttribute('aria-selected', String(el.dataset.tl === s));
    else el.hidden = el.dataset.tl !== s;
  });
  document.getElementById('tl-readout').textContent = '';
  const el = document.getElementById('ch-timeline');
  if (visible(el)) Plotly.react(el, timelineSpec(null).data, timelineSpec(null).layout, plotConfig());
}
function stopPlay(){ if (state.playing){ cancelAnimationFrame(state.playing); state.playing = null; } document.getElementById('tl-play').textContent = '▶ Play'; }
function play(){
  const el = document.getElementById('ch-timeline');
  if (state.playing){ stopPlay(); return; }
  if (!visible(el)) return;
  const d = DATA.timeline[state.tl], conds = Object.keys(d.cum);
  const maxT = Math.max(...conds.map(c => d.cum[c][d.cum[c].length-1]));
  const wall = 12, speed = maxT / wall, t0 = performance.now(), out = document.getElementById('tl-readout');
  document.getElementById('tl-play').textContent = '■ Stop';
  const count = (arr, T) => { let lo = 0, hi = arr.length; while (lo < hi){ const m = (lo+hi) >> 1; if (arr[m] <= T) lo = m+1; else hi = m; } return lo; };
  const frame = now => {
    const T = (now - t0)/1000 * speed, slices = {};
    conds.forEach(c => slices[c] = count(d.cum[c], T));
    const spec = timelineSpec(slices);
    Plotly.react(el, spec.data, spec.layout, plotConfig());
    out.textContent = `t = ${Math.min(T, maxT).toFixed(1)} s (replay ×${speed.toFixed(1)}) · ` + conds.map(c => `${c} ${slices[c]}/${d.steps.length}`).join(' · ');
    if (T < maxT) state.playing = requestAnimationFrame(frame);
    else { state.playing = null; document.getElementById('tl-play').textContent = '▶ Replay'; }
  };
  state.playing = requestAnimationFrame(frame);
}

/* ---------------- modes, navigation ---------------- */
const tabs = document.getElementById('tabs');
PANELS.forEach((p, i) => {
  const a = document.createElement('a'); a.href = '#'+p.id; a.innerHTML = `<span class="tn">${i+1}</span>${p.dataset.label}`;
  a.addEventListener('click', e => { e.preventDefault(); go(i); });
  tabs.appendChild(a);
});
function markTab(i){ tabs.querySelectorAll('a').forEach((a,j) => a.classList.toggle('on', j === i)); const on = tabs.children[i]; if (on && on.scrollIntoView && !present()) on.scrollIntoView({block:'nearest', inline:'nearest'}); }

function go(i){
  i = Math.max(0, Math.min(N-1, i));
  state.idx = i;
  try { history.replaceState(null, '', '#'+PANELS[i].id); } catch(e){}
  if (present()){
    if (state.playing) stopPlay();
    PANELS.forEach((p,j) => p.classList.toggle('active', j === i));
    document.querySelector('#progress .bar').style.width = ((i+1)/N*100)+'%';
    document.querySelector('#progress .label').textContent = `${i+1} / ${N}`;
    requestAnimationFrame(() => renderPanel(PANELS[i].id));
  } else {
    PANELS[i].scrollIntoView({behavior: reduceMotion ? 'auto' : 'smooth', block:'start'});
  }
  markTab(i);
}
function setMode(m){
  state.mode = m; store.set('mode', m);
  document.body.classList.toggle('present', m === 'present');
  document.body.classList.toggle('explore', m === 'explore');
  document.getElementById('btn-mode').innerHTML = (m === 'present' ? 'Explore' : 'Present') + '<kbd>P</kbd>';
  stopPlay();
  go(state.idx);
  requestAnimationFrame(rerenderVisible);
}
function setNotes(on){ state.notes = on; store.set('notes', on ? '1' : '0'); document.body.classList.toggle('show-notes', on); document.getElementById('btn-notes').setAttribute('aria-pressed', String(on)); }
function setTheme(t){
  state.theme = t; store.set('theme', t);
  if (t === 'auto') document.documentElement.removeAttribute('data-theme'); else document.documentElement.setAttribute('data-theme', t);
  document.getElementById('btn-theme').textContent = 'Theme: ' + t;
  requestAnimationFrame(rerenderVisible);
}
function fullscreen(){
  try {
    const r = document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
    if (r && typeof r.catch === 'function') r.catch(() => {});
  } catch(e){}
}
let digits = '', digitTimer = null;
function digit(k){
  digits += k; clearTimeout(digitTimer);
  const jump = () => { const v = digits === '0' ? 10 : parseInt(digits, 10); digits = ''; if (v >= 1 && v <= N) go(v-1); };
  if (digits.length === 1 && parseInt(digits + '0', 10) <= N && digits !== '0') digitTimer = setTimeout(jump, 450);
  else jump();
}
document.addEventListener('keydown', e => {
  if (e.altKey || e.ctrlKey || e.metaKey) return;
  if (e.target.closest && e.target.closest('input,textarea,select,[contenteditable]')) return;
  const k = e.key;
  if (k === 'p' || k === 'P'){ setMode(present() ? 'explore' : 'present'); }
  else if (k === 'n' || k === 'N'){ setNotes(!state.notes); }
  else if (k === 'f' || k === 'F'){ fullscreen(); }
  else if (/^[0-9]$/.test(k)){ digit(k); }
  else if (present()){
    if (k === 'ArrowRight' || k === ' ' || k === 'PageDown'){ e.preventDefault(); go(state.idx+1); }
    else if (k === 'ArrowLeft' || k === 'PageUp' || k === 'Backspace'){ e.preventDefault(); go(state.idx-1); }
    else if (k === 'Home'){ go(0); } else if (k === 'End'){ go(N-1); }
    else if (k === 'Escape' && !document.fullscreenElement){ setMode('explore'); }
  }
});
let touchX = null;
document.addEventListener('touchstart', e => { touchX = e.touches[0].clientX; }, {passive:true});
document.addEventListener('touchend', e => { if (!present() || touchX === null) return; const dx = e.changedTouches[0].clientX - touchX; if (Math.abs(dx) > 60) go(state.idx + (dx < 0 ? 1 : -1)); touchX = null; });

document.getElementById('btn-mode').addEventListener('click', () => setMode(present() ? 'explore' : 'present'));
document.getElementById('btn-notes').addEventListener('click', () => setNotes(!state.notes));
document.getElementById('btn-theme').addEventListener('click', () => setTheme({auto:'light', light:'dark', dark:'auto'}[state.theme]));
document.getElementById('btn-fs').addEventListener('click', fullscreen);
document.querySelectorAll('button[data-tl]').forEach(b => b.addEventListener('click', () => setTimeline(b.dataset.tl)));
document.getElementById('tl-play').addEventListener('click', play);
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (state.theme === 'auto') rerenderVisible(); });

/* explore: render charts as panels scroll into view, highlight the tab */
const io = new IntersectionObserver(entries => {
  if (present()) return;
  entries.forEach(en => { if (en.isIntersecting){ renderPanel(en.target.id); } });
  const mid = innerHeight * 0.35;
  let cur = 0; PANELS.forEach((p,i) => { if (p.getBoundingClientRect().top <= mid) cur = i; });
  state.idx = cur; markTab(cur);
}, {threshold:[0, .15, .5]});
PANELS.forEach(p => io.observe(p));
/* charts follow their container (e.g. when the play readout or notes change the layout) */
const ro = new ResizeObserver(entries => entries.forEach(en => {
  const el = en.target;
  if (el.classList.contains('js-plotly-plot') && visible(el)) requestAnimationFrame(() => { try { Plotly.Plots.resize(el); } catch(e){} });
}));

window.addEventListener('scroll', () => {
  if (present()) return;
  const mid = innerHeight * 0.35; let cur = 0;
  PANELS.forEach((p,i) => { if (p.getBoundingClientRect().top <= mid) cur = i; });
  if (cur !== state.idx){ state.idx = cur; markTab(cur); }
}, {passive:true});

decorate();
document.querySelectorAll('.chart').forEach(el => ro.observe(el));
setTheme(store.get('theme') || 'auto');
setNotes(store.get('notes') === '1');
const h = location.hash.match(/^#p(\d+)$/);
if (h) state.idx = Math.max(0, Math.min(N-1, parseInt(h[1],10)-1));
setMode(store.get('mode') === 'present' ? 'present' : 'explore');
window.__dash = {go, setMode, state, N};
})();
</script>
</body>
</html>
"""


def main():
    data = build_data()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page(data), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e6:.2f} MB), {len(facts)} facts")


if __name__ == "__main__":
    main()
