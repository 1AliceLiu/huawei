"""Background watchdog: poll for the multi-stage summary and wake the agent when done.

The summary file is written atomically by ``run_multistage.py`` only after BOTH stages
finish, so a partially written file never triggers a wake-up.

    setsid python3 scripts/watchdog.py > logs/watchdog.log 2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT = ROOT / "runs" / "experiments" / "multistage_summary.json"


def build_prompt(report: dict) -> str:
    s1, s2 = report["stage1_results"], report["stage2_results"]
    return (
        "两阶段训练已完成。"
        f"Stage1(全量池) mAP50-95={s1['map50_95']:.4f} mAP50={s1['map50']:.4f}；"
        f"Stage2(HQ子集) mAP50-95={s2['map50_95']:.4f} mAP50={s2['map50']:.4f}；"
        f"净收益 ΔmAP50-95={report['delta_mAP50_95']:+.4f}。"
        f"请读取 {DEFAULT_REPORT} 并结合 frozen_val 的各类别指标分析召回/混淆，给出下一步建议。"
    )


def wake_agent(report: dict, dry_run: bool) -> None:
    prompt = build_prompt(report)
    print("\n[watchdog] training finished, waking agent...", flush=True)
    print("[watchdog] prompt:", prompt, flush=True)
    if dry_run:
        print("[watchdog] --dry-run: not invoking opencode", flush=True)
        return
    try:
        subprocess.run(["opencode", "run", prompt], cwd=str(ROOT), check=True)
        print("[watchdog] agent notified", flush=True)
    except FileNotFoundError:
        print("[watchdog] opencode not found on PATH - skipping wake-up", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"[watchdog] opencode exited with {e.returncode}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--interval", type=int, default=60, help="poll interval seconds")
    ap.add_argument("--timeout-hours", type=float, default=48.0)
    ap.add_argument("--dry-run", action="store_true", help="detect + print, do not call opencode")
    args = ap.parse_args()

    report = args.report.resolve()
    print(f"[watchdog] watching {report} (every {args.interval}s, cap {args.timeout_hours}h)", flush=True)
    deadline = time.time() + args.timeout_hours * 3600
    while time.time() < deadline:
        if report.is_file():
            try:
                data = json.loads(report.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                time.sleep(args.interval)
                continue
            wake_agent(data, args.dry_run)
            return
        time.sleep(args.interval)
    print("[watchdog] timed out waiting for the report", flush=True)


if __name__ == "__main__":
    main()
