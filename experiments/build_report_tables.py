"""
Generates report tables from the source CSVs and writes them into docs/report/final_report.html
between <!-- BEGIN name ... --> and <!-- END name --> markers. Rerunning replaces the block.

Tables:
  v6_sweep_table  mean decode_sec_per_token per load_workers x thread_budget from
                  data/processed/load_sweep_v6.csv, the fastest budget per row, and the
                  10-thread / 2-thread ratio where 2 threads is fastest at 12+ workers.

    python experiments/build_report_tables.py
"""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "docs" / "report" / "final_report.html"


def v6_sweep_table():
    d = pd.read_csv(ROOT / "data" / "processed" / "load_sweep_v6.csv")
    t = d.pivot_table(index="load_workers", columns="thread_budget", values="decode_sec_per_token", aggfunc="mean")
    n = d.groupby(["load_workers", "thread_budget"]).size()
    budgets = list(t.columns)
    head = "".join(f'<th class="n">{b} threads</th>' for b in budgets)
    rows = [f"<table>", f"<tr><th>Load workers</th>{head}<th>Fastest budget</th></tr>"]
    for w, row in t.iterrows():
        best = row.idxmin()
        cells = "".join(
            f'<td class="n">{"<b>" if b == best else ""}{row[b]:.4f}{"</b>" if b == best else ""}</td>' for b in budgets)
        note = f"<b>{best} threads</b>"
        if best == 2 and 10 in budgets and w >= 12:
            note += f" ({row[10] / row[2]:.1f}&times; faster than 10 threads)"
        label = f"{w} (idle)" if w == 0 else f"{w}"
        rows.append(f"<tr><td>{label}</td>{cells}<td>{note}</td></tr>")
    rows.append("</table>")
    reps = sorted(n.unique())
    rows.append(f'<p class="small">Mean decode seconds per token over {"/".join(map(str, reps))} repeats per cell, '
                f"generated from <code>data/processed/load_sweep_v6.csv</code> ({len(d)} runs) by "
                f"<code>experiments/build_report_tables.py</code>.</p>")
    return "\n".join(rows)


TABLES = {"v6_sweep_table": v6_sweep_table}


def main():
    html = REPORT.read_text(encoding="utf-8")
    for name, fn in TABLES.items():
        pat = re.compile(rf"(<!-- BEGIN {name}[^>]*-->\n?).*?(<!-- END {name} -->)", re.S)
        if not pat.search(html):
            raise SystemExit(f"markers for {name} not found in {REPORT}")
        html = pat.sub(lambda m: m.group(1) + fn() + "\n" + m.group(2), html)
        print(f"{name}: written")
    REPORT.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
