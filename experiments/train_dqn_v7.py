"""
Stage 6A: DQN v7 (contextual bandit, gamma 0) trained only on the v7 load sweep.

Context = one (prompt, repeat, workers) block of data/processed/load_sweep_v7.csv,
in which every v7 arm was measured once. State = [external CPU % (mean during the
block's runs), available RAM MB (mean before runs), prompt tokens], scaled by
src.scheduler.adaptive.v7_state. Action = v7 arm. Reward = -decode s/token.
Every context gives full feedback (all arms measured).

Leave-one-prompt-out over prompts 1-3 (3 folds, fixed seeds). Per fold: mean
regret (s/token) vs the per-context best arm and % of contexts where the best arm
is picked, for the DQN, rule_lookup rebuilt from the fold's training prompts
(experiments/build_rule_v7.py logic), the full-data rule from experiments/e3_rule.yaml
(in-sample for the held-out prompt, shown for reference), and cores_heuristic.
Then the final model is trained on all three prompts:
data/processed/dqn_model_v7.pth. Outputs: experiments/e3_dqn_v7.csv and .txt.

    python experiments/train_dqn_v7.py
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SWEEP = "data/processed/load_sweep_v7.csv"
SEED = 42
EPISODES = 300


def prompt_token_counts(prompt_ids):
    from src.data.load_prompts import load_prompts
    from src.llm.load_model import load_tokenizer

    tok = load_tokenizer()
    prompts = load_prompts(csv_path=str(ROOT / "data" / "processed" / "prompts.csv"), limit=max(prompt_ids))
    return {pid: len(tok.encode(prompts[pid - 1])) for pid in prompt_ids}


def contexts(df, arms, tokens):
    from src.scheduler.adaptive import v7_state

    out = []
    for (pid, rep, w), g in df.groupby(["prompt_id", "repeat", "load_workers"]):
        spt = g.set_index("arm")["decode_sec_per_token"]
        if set(spt.index) != set(arms):
            continue
        cpu = float(g["external_cpu_during"].mean())
        avail = float(g["available_mb_before"].mean())
        out.append({"prompt_id": int(pid), "repeat": int(rep), "load_workers": int(w), "external_cpu": cpu,
                    "available_mb": avail, "prompt_tokens": tokens[pid],
                    "state": v7_state(cpu, avail, tokens[pid]), "spt": np.array([spt[a] for a in arms])})
    return out


def train(ctxs, n_arms, seed=SEED, episodes=EPISODES):
    import torch
    from src.rl.dqn_agent import DQNAgent

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = DQNAgent(state_size=3, action_size=n_arms, gamma=0.0)
    rng = np.random.default_rng(seed)
    for _ in range(episodes):
        for i in rng.permutation(len(ctxs)):
            c = ctxs[i]
            for a in range(n_arms):
                agent.remember(c["state"], a, -float(c["spt"][a]), c["state"], True)
            agent.train_step(batch_size=16)
        agent.update_target_network()
    agent.epsilon = 0.0
    return agent


def evaluate(pick, ctxs):
    regrets, hits = [], []
    for c in ctxs:
        a = pick(c)
        regrets.append(float(c["spt"][a] - c["spt"].min()))
        hits.append(int(a == int(np.argmin(c["spt"]))))
    return float(np.mean(regrets)), 100.0 * float(np.mean(hits))


def main():
    from build_rule_v7 import build_bins, level_table
    from src.scheduler.adaptive import CoresHeuristicPolicy, LookupRulePolicy

    df = pd.read_csv(ROOT / SWEEP)
    sweep_plan = yaml.safe_load((ROOT / "experiments" / "e1_plan.yaml").read_text(encoding="utf-8"))
    arms = list(sweep_plan["arms"])
    prompts = sorted(df["prompt_id"].unique())
    tokens = prompt_token_counts(prompts)
    full_rule = yaml.safe_load((ROOT / "experiments" / "e3_rule.yaml").read_text(encoding="utf-8"))
    full_lookup = LookupRulePolicy(full_rule["edges"], full_rule["arms"], patience=0)
    heuristic = CoresHeuristicPolicy(patience=0)

    def rule_pick(lookup):
        return lambda c: arms.index(lookup.arms[lookup.bin_of(c["external_cpu"])])

    def heur_pick(c):
        return arms.index(f"threads_{heuristic.snap(heuristic.target_threads(c['external_cpu']))}")

    rows = []
    for held in prompts:
        train_df = df[df["prompt_id"] != held]
        train_ctx = contexts(train_df, arms, tokens)
        test_ctx = contexts(df[df["prompt_id"] == held], arms, tokens)
        agent = train(train_ctx, len(arms))
        table, _ = level_table(train_df)
        bins, edges = build_bins(table, arms)
        fold_lookup = LookupRulePolicy(edges, [b["arm"] for b in bins], patience=0)
        for name, pick in [("dqn_v7", lambda c: int(np.argmax(agent.q_values(c["state"])))),
                           ("rule_lookup_fold", rule_pick(fold_lookup)),
                           ("rule_lookup_full_in_sample", rule_pick(full_lookup)),
                           ("cores_heuristic", heur_pick)]:
            regret, hit = evaluate(pick, test_ctx)
            rows.append({"held_out_prompt": int(held), "policy": name, "num_contexts": len(test_ctx),
                         "mean_regret_sec_per_token": round(regret, 5), "percent_best_arm": round(hit, 1),
                         "fold_rule": f"edges {edges} arms {[b['arm'] for b in bins]}"
                         if name == "rule_lookup_fold" else ""})
    folds = pd.DataFrame(rows)
    mean = folds.groupby("policy", sort=False)[["mean_regret_sec_per_token", "percent_best_arm"]].mean()

    final = train(contexts(df, arms, tokens), len(arms))
    model_path = ROOT / "data" / "processed" / "dqn_model_v7.pth"
    out_csv, out_txt = ROOT / "experiments" / "e3_dqn_v7.csv", ROOT / "experiments" / "e3_dqn_v7.txt"
    for p in (model_path, out_csv, out_txt):
        if p.exists():
            raise SystemExit(f"{p} exists; results are never overwritten")
    final.save(str(model_path))
    folds.to_csv(out_csv, index=False)

    dqn, rule = mean.loc["dqn_v7", "mean_regret_sec_per_token"], mean.loc["rule_lookup_fold", "mean_regret_sec_per_token"]
    verdict = ("DQN v7 beats rule_lookup offline (lower mean regret over held-out folds)" if dqn < rule else
               "DQN v7 does NOT beat rule_lookup offline (mean regret not lower over held-out folds)")
    full_ctx = contexts(df, arms, tokens)
    final_picks = pd.Series([arms[int(np.argmax(final.q_values(c["state"])))] for c in full_ctx]).value_counts()
    lines = [f"DQN v7 offline evaluation (experiments/train_dqn_v7.py), source: {SWEEP}",
             f"arms: {arms}; state: [external CPU %, available RAM MB, prompt tokens] (scaled); reward: -s/token",
             f"prompt tokens: {tokens}; seed {SEED}; {EPISODES} episodes; gamma 0; full feedback",
             "", "Leave-one-prompt-out folds:", folds.drop(columns=["fold_rule"]).to_string(index=False), "",
             "Fold rules (built from the two training prompts):"]
    lines += [f"  held-out {r.held_out_prompt}: {r.fold_rule}" for r in folds.itertuples() if r.fold_rule]
    lines += ["", "Mean over folds:", mean.round(5).to_string(), "", verdict,
              "(rule_lookup_full_in_sample uses all three prompts, so it has seen the held-out prompt.)",
              "", f"Final model (all three prompts): data/processed/dqn_model_v7.pth",
              f"final model picks on the sweep contexts: {final_picks.to_dict()}"]
    text = "\n".join(lines) + "\n"
    out_txt.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
