#!/usr/bin/env python3
"""
PathFinder - Track 03 (Samsung PRISM XR Hackathon 2026)
Degraded IMU CSV (EuRoC format)  ->  6-DoF trajectory (TUM format), one pose per IMU frame.

usage:  python main.py [<imu0/data.csv | dataset folder | dataset.zip>] [estimated_trajectory.txt]
                       [-o estimated_trajectory.txt] [--meta metadata.json] [--no-net] [--profile]
        With no input, it looks in the working directory for imu0/data.csv, mav0/imu0/data.csv, data.csv or a .zip.
"""
import argparse, json, os, sys, time

# Track 03 rule: single CPU thread -> pin every BLAS / OpenMP backend before numpy is imported
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser(description="PathFinder: IMU-only 6-DoF trajectory under sensor degradation")
    ap.add_argument("input", nargs="?", default=None, help="imu0/data.csv, a EuRoC folder, or a .zip of it")
    ap.add_argument("output_pos", nargs="?", default=None, metavar="output", help="output file (same as -o)")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--meta", default=None, help="write run statistics (json)")
    ap.add_argument("--no-net", action="store_true", help="disable the learned velocity prior (physics only)")
    ap.add_argument("--weights", default=None, help="alternative velocity-network weights (.npz)")
    ap.add_argument("--params", default=None, help="JSON overrides of estimator parameters (research / ablations)")
    ap.add_argument("--profile", action="store_true", help="measure peak heap (tracemalloc) and runtime")
    a = ap.parse_args()
    a.output = a.output or a.output_pos or "estimated_trajectory.txt"
    if a.input is None:
        for cand in ("imu0/data.csv", "mav0/imu0/data.csv", "data/imu0/data.csv", "data.csv"):
            if os.path.isfile(cand):
                a.input = cand
                break
        else:
            zips = sorted(f for f in os.listdir(".") if f.lower().endswith(".zip"))
            if len(zips) == 1:
                a.input = zips[0]
            else:
                ap.error("no input given and no imu0/data.csv, data.csv or single .zip in the working directory")
    od = os.path.dirname(os.path.abspath(a.output))
    os.makedirs(od, exist_ok=True)

    t0 = time.perf_counter()
    import numpy as np
    import pathfinder as PF
    import zipfile, numpy.lib.format, numpy.ma, encodings.cp437  # noqa: F401  module code, not algorithm data
    if a.profile:
        # algorithm-level heap (FAQ): measured after interpreter + numpy import
        import tracemalloc
        tracemalloc.start()
    path = PF.resolve_input(a.input)
    weights = None
    wpath = a.weights or os.path.join(HERE, "velnet.npz")
    if not a.no_net and os.path.isfile(wpath):
        with np.load(wpath) as z:   # fp16 weights stay fp16 in memory (cast per layer)
            weights = {k: z[k] for k in z.files}
    res = PF.estimate(path, P=json.loads(a.params) if a.params else None, weights=weights)
    del weights
    PF.write_tum(res, a.output)
    dt = time.perf_counter() - t0
    info = res["info"]
    info.update(runtime_s=round(dt, 4), ms_per_imu_second=round(1000 * dt / max(info["duration_s"], 1e-9), 3),
                output=os.path.abspath(a.output))
    if a.profile:
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        info.update(peak_heap_MB=round(peak / 1e6, 3), peak_heap_MiB=round(peak / 2 ** 20, 3))   # MB = 10^6 bytes
    if a.meta:
        with open(a.meta, "w") as f:
            json.dump(info, f, indent=2, default=float)
    print("PathFinder: %d poses -> %s  (%.2fs of IMU, %.3f s, %.2f ms per IMU-second%s)" % (
        info["rows"], a.output, info["duration_s"], dt, info["ms_per_imu_second"],
        (", peak heap %.2f MB" % info["peak_heap_MB"]) if a.profile else ""))


if __name__ == "__main__":
    main()
