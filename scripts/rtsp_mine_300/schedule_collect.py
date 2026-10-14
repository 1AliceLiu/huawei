"""周期采集调度：每 period 小时跑一次 collect.py（每轮 per-cycle 张，--resume 累积）。

目的：跨时段采样，捕获光照/车流变化。每轮 collect 自写 run_*.csv，本脚本另存 stdout。

用法：
  python scripts/rtsp_mine_300/schedule_collect.py --out-dir dataset/rtsp_mine_v4_icc \
      --per-cycle 300 --period-hours 2
  高速15路混采（collect_highway15.py，单实例 pid 锁）：
  python scripts/rtsp_mine_300/schedule_collect.py \
      --collect-script scripts/rtsp_mine_300/collect_highway15.py \
      --config scripts/rtsp_mine_300/config_highway15.yaml \
      --out-dir dataset/rtsp_mine_600_highway15_pending \
      --per-cycle 300 --period-hours 2 --extra-args "--max-minutes 90"
停止：在 <out-dir>/SCHEDULE_STOP 建任意文件（或结束进程）。
"""
import argparse
import csv
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILTER = ROOT / "scripts/rtsp_mine_300/filter_empty.py"


def reconcile_manifest(ds: Path):
    """删掉 manifest 中已不在 images/ 的行（filter 隔离后同步）。"""
    mp = ds / "manifest.csv"
    if not mp.is_file():
        return
    rows = list(csv.DictReader(open(mp, encoding="utf-8")))
    if not rows:
        return
    fields = list(rows[0].keys())
    keep = [r for r in rows if (ds / "images" / r["file"]).is_file()]
    with open(mp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(keep)
    return len(rows) - len(keep)


def post_filter(ds_arg: str, log):
    """激进：COCO 无车帧全部隔离到 _removed_empty/（非删除），并同步 manifest。"""
    ds = ROOT / ds_arg
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n=== post-filter: quarantine all COCO-no-vehicle frames ===\n")
        subprocess.run([sys.executable, str(FILTER), "--dataset", ds_arg,
                        "--apply", "--allow-bare-apply"], stdout=f, stderr=subprocess.STDOUT)
    dropped = reconcile_manifest(ds)
    q = len(list((ds / "_removed_empty").glob("*.jpg")))
    print(f"[post-filter] quarantine_total={q} manifest_dropped={dropped} "
          f"kept={len(list((ds/'images').glob('*.jpg')))}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="scripts/rtsp_mine_300/config.yaml")
    ap.add_argument("--collect-script", default="scripts/rtsp_mine_300/collect.py",
                    help="每轮执行的采集脚本（相对 ROOT 或绝对路径）")
    ap.add_argument("--extra-args", default="",
                    help="追加给采集脚本的额外参数，如 \"--max-minutes 90\"")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-cycle", type=int, default=300)
    ap.add_argument("--period-hours", type=float, default=2.0)
    ap.add_argument("--cycles", type=int, default=0, help="0=无限，按停止文件结束")
    ap.add_argument("--log-dir", default="output/rtsp_mine_300")
    ap.add_argument("--post-filter", action="store_true",
                    help="每轮采集后隔离 COCO 无车帧（激进）")
    args = ap.parse_args()

    out = ROOT / args.out_dir
    log_dir = ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    stop_file = out / "SCHEDULE_STOP"
    out.mkdir(parents=True, exist_ok=True)
    if stop_file.exists():
        stop_file.unlink()
    (log_dir / "schedule.pid").write_text(str(__import__("os").getpid()), encoding="utf-8")

    py = sys.executable
    collect = Path(args.collect_script)
    if not collect.is_absolute():
        collect = ROOT / collect
    if not collect.is_file():
        raise SystemExit(f"采集脚本不存在: {collect}")
    extra = shlex.split(args.extra_args) if args.extra_args else []
    n = 0
    while args.cycles == 0 or n < args.cycles:
        n += 1
        t0 = time.time()
        log = log_dir / f"schedule_cycle{n:03d}_{datetime.now():%Y%m%d_%H%M%S}.log"
        cmd = [py, str(collect), "--config", args.config, "--target", str(args.per_cycle),
               "--out-dir", args.out_dir, "--resume"] + extra
        print(f"[cycle {n}] {datetime.now():%F %T} start -> {log.name} :: {' '.join(cmd)}", flush=True)
        with open(log, "w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        total = len(list((out / "images").glob("*.jpg")))
        print(f"[cycle {n}] {datetime.now():%F %T} exit={rc} elapsed={time.time()-t0:.0f}s total_imgs={total}",
              flush=True)
        if args.post_filter:
            post_filter(args.out_dir, log)
        while time.time() - t0 < args.period_hours * 3600:
            if stop_file.exists():
                print("[stop] SCHEDULE_STOP found", flush=True)
                return
            time.sleep(30)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
