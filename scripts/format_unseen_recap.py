#!/usr/bin/env python3
"""
Print a clean summary of cohorts/unseen_benchmark_results.json.

Usage:
    python scripts/format_unseen_recap.py [path-to-json]
"""
import json
import os
import sys

DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cohorts", "unseen_benchmark_results.json",
)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    if not os.path.exists(path):
        print(f"results file not found: {path}")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    rows = []
    for label, obj_results in data.items():
        if not isinstance(obj_results, dict) or "__error__" in obj_results:
            rows.append((label, None, None, None, "ERROR"))
            continue
        ov = obj_results.get("__overall__")
        if ov is None:
            rows.append((label, None, None, None, "no overall"))
            continue
        sr = ov.get("success_rate", 0) * 100
        cr = ov.get("collision_rate", 0) * 100
        gd = ov.get("goal_dist", float("inf"))
        rows.append((label, sr, cr, gd, ""))

    rows.sort(key=lambda r: -(r[1] or -1))

    print("=" * 78)
    print(f"UNSEEN BENCHMARK — sorted by success rate  (33 val branches, 11/object)")
    print("=" * 78)
    print(f"{'Rank':<5} {'Model':<30} {'Success%':>10} {'Collision%':>12} {'GoalDist':>10}")
    print("-" * 78)
    for i, (label, sr, cr, gd, note) in enumerate(rows, 1):
        if sr is None:
            print(f"{i:<5} {label:<30} {note:>10}")
            continue
        print(f"{i:<5} {label:<30} {sr:>9.1f}% {cr:>11.1f}% {gd:>9.2f}m")
    print("=" * 78)

    print("\nPER-OBJECT BREAKDOWN")
    print("-" * 78)
    for label, obj_results in data.items():
        if not isinstance(obj_results, dict) or "__error__" in obj_results:
            continue
        print(f"\n{label}:")
        for obj_name, m in obj_results.items():
            if obj_name == "__overall__":
                continue
            print(
                f"  {obj_name[:40]:<40} "
                f"sr={m['success_rate']*100:>5.1f}%  "
                f"cr={m['collision_rate']*100:>5.1f}%  "
                f"gd={m['goal_dist']:>6.2f}m  n={m['n_eval']}"
            )

    # Identify the best non-DAgger BC variant for DAgger candidacy
    bc_rows = [r for r in rows if r[1] is not None and "DAgger" not in r[0]]
    if bc_rows:
        winner = bc_rows[0]
        print("\n" + "=" * 78)
        print(f"BEST BC VARIANT (DAgger candidate): {winner[0]}")
        print(f"  success={winner[1]:.1f}%  collision={winner[2]:.1f}%  goal_dist={winner[3]:.2f}m")
        print("=" * 78)


if __name__ == "__main__":
    main()
