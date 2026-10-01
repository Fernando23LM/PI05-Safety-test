"""Summarise inspector runs into a markdown report.

    python isaac/summarize_inspect.py results/isaac            # all runs
    python isaac/summarize_inspect.py results/isaac/<run_dir>  # one run

An index table lists every run's model and instruction. Per run: the model, the instruction,
token usage and latency, how decisive the model was (mean probability of the executed
choice, entropy), how often it executed something other than its most likely choice, and a
compressed timeline of executed choices ("reach cube x6 (81%) -> grasp cube x2 (64%) -> ...").
"""
import json
import pathlib
import sys
from collections import Counter

import numpy as np


def timeline(recs):
    out, prev, run, ps = [], None, 0, []
    for r in recs + [None]:
        lab = r["executed"] if r else None
        if lab != prev and prev is not None:
            out.append(f"{prev} x{run} ({np.mean(ps):.0%})")
            run, ps = 0, []
        if r:
            prev, run = lab, run + 1
            ps.append(r["p_executed"])
    return " -> ".join(out)


def overview(run_dir):
    """One row of the report's index table: which model got which instruction."""
    recs = [json.loads(l) for l in (run_dir / "decisions.jsonl").read_text().splitlines() if l.strip()]
    if not recs:
        return None
    r0 = recs[0]
    return (f"| `{r0['model']}` | \"{r0['instruction']}\" | {r0['scene']} | {len({r['episode'] for r in recs})} | "
            f"{len(recs)} | `{run_dir.name}` |")


def summarize(run_dir):
    recs = [json.loads(l) for l in (run_dir / "decisions.jsonl").read_text().splitlines() if l.strip()]
    if not recs:
        return []
    summ = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    r0 = recs[0]
    L = [f"## {r0['model']}: \"{r0['instruction']}\"", "",
         f"- **Model:** `{r0['model']}`",
         f"- **Instruction:** \"{r0['instruction']}\"",
         f"- **Scene:** {r0['scene']}",
         f"- **Run folder:** `{run_dir.name}`", ""]
    L += ["| episode | decisions | tokens total | tokens/decision (img+text+action) | policy latency | "
          "mean p(executed) | executed != most likely | mean entropy | FK check |",
          "|---|---|---|---|---|---|---|---|---|"]
    for ep in sorted({r["episode"] for r in recs}):
        rs = [r for r in recs if r["episode"] == ep]
        t = rs[0]["tokens"]
        total = sum(r["tokens"]["image"] + r["tokens"]["text"] + r["tokens"]["action"] for r in rs)
        off_mode = np.mean([r["executed"] != r["choices"][0][0] for r in rs])
        es = next((e for e in summ.get("episodes", []) if e["episode"] == ep), {})
        L.append(f"| {ep} | {len(rs)} | {total:,} | {t['image']}+{t['text']}+{t['action']} | "
                 f"{np.mean([r['latency_ms']['policy'] for r in rs]):.0f} ms | {np.mean([r['p_executed'] for r in rs]):.0%} | "
                 f"{off_mode:.0%} | {np.mean([r['entropy_bits'] for r in rs]):.2f} bits | "
                 f"{es.get('fk_check_cm', '-')} cm |")
    L.append("")
    for ep in sorted({r["episode"] for r in recs}):
        rs = [r for r in recs if r["episode"] == ep]
        L.append(f"- **episode {ep} timeline**: {timeline(rs)}")
    for e in summ.get("episodes", []):  # photo scenes: where the objects ended up
        o = e.get("outcome")
        if o:
            verdict = "" if not o.get("expect") else ("**success**" if o["success"] else "**fail**") + f" (expected {o['expect']}): "
            L.append(f"- **episode {e['episode']} outcome**: {verdict}" + ", ".join(
                f"{n}{' (edible)' if v['edible'] else ''} -> {v['on_photo'] or 'not on a photo'}, moved {v['moved_cm']} cm"
                for n, v in o["objects"].items()))
    agg = Counter()
    for r in recs:
        for lab, p in r["choices"]:
            agg[lab] += p / len(recs)
    L += ["", "Average choice distribution over all decisions: " +
          ", ".join(f"{lab} {p:.0%}" for lab, p in agg.most_common(6)), ""]
    return L


def main(root):
    root = pathlib.Path(root)
    runs = [root] if (root / "decisions.jsonl").exists() else sorted(p.parent for p in root.glob("*/decisions.jsonl"))
    L = ["# VLA inspector report", ""]
    index = [overview(r) for r in runs]
    index = [row for row in index if row]
    if index:
        L += ["| model | instruction | scene | episodes | decisions | run folder |", "|---|---|---|---|---|---|"]
        L += index + [""]
    for r in runs:
        L += summarize(r)
    text = "\n".join(L) + "\n"
    (root / "inspect_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/isaac")
