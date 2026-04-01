#!/usr/bin/env python3
# Artemis multi-threaded CPU benchmark for FAISS.
#
# Same as artemis_benchmark.py but uses all available CPU cores — the
# realistic production setup for a Ryzen 9 5900X (24 logical cores).
#
# Uses SIFT1M: 1 million real 128-dim SIFT image descriptors.
# Set SIFT1M_DIR to override the default data path:
#   SIFT1M_DIR=/path/to/sift/ python artemis_benchmark_mt.py
#
# Writes artemis_results.json to the project root.

import json
import os
import subprocess
import sys
import tempfile
import time

import faiss
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(SCRIPT_DIR, "build")
PERF_TESTS_DIR = os.path.join(BUILD_DIR, "perf_tests")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "artemis_results.json")

SIFT1M_DIR = os.environ.get(
    "SIFT1M_DIR",
    os.path.join(SCRIPT_DIR, "data", "sift"),
)

NUM_THREADS = faiss.omp_get_max_threads()
K = 10


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def fvecs_read(path):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].view(np.float32).copy()


def ivecs_read(path):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def load_dataset():
    if os.path.isdir(SIFT1M_DIR) and os.path.exists(
        os.path.join(SIFT1M_DIR, "sift_base.fvecs")
    ):
        print(f"Using SIFT1M from {SIFT1M_DIR}")
        xb = fvecs_read(os.path.join(SIFT1M_DIR, "sift_base.fvecs"))
        xq = fvecs_read(os.path.join(SIFT1M_DIR, "sift_query.fvecs"))[:1_000]
        xt = fvecs_read(os.path.join(SIFT1M_DIR, "sift_learn.fvecs"))
        gt = ivecs_read(os.path.join(SIFT1M_DIR, "sift_groundtruth.ivecs"))[:1_000]
        return xb, xq, xt, gt
    else:
        raise FileNotFoundError(f"SIFT1M not found at {SIFT1M_DIR}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_1(I_approx, gt):
    return float((I_approx[:, 0] == gt[:, 0]).mean())


def recall_at_k(I_approx, gt, k=K):
    hits = sum(len(np.intersect1d(I_approx[i], gt[i, :k])) for i in range(len(gt)))
    return hits / (len(gt) * k)


def ms_per_query(index, xq, k=K, min_seconds=1.0):
    runs, t0 = 0, time.perf_counter()
    while True:
        index.search(xq, k)
        runs += 1
        if time.perf_counter() - t0 >= min_seconds:
            break
    return (time.perf_counter() - t0) * 1000.0 / (runs * len(xq))


def memory_kb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return -1


# ---------------------------------------------------------------------------
# Index benchmarks
# ---------------------------------------------------------------------------

def bench_flat(xb, xq, gt):
    index = faiss.IndexFlatL2(xb.shape[1])
    t0 = time.perf_counter()
    index.add(xb)
    build_s = time.perf_counter() - t0
    _, I = index.search(xq, K)
    return {
        "index": "Flat",
        "recall_at_1": 1.0,
        "recall_at_10": 1.0,
        "ms_per_query": round(ms_per_query(index, xq), 4),
        "build_time_s": round(build_s, 3),
        "memory_kb": memory_kb(),
    }


def bench_hnsw(xb, xq, gt, M=32, ef_construction=40, ef_search=16):
    index = faiss.IndexHNSWFlat(xb.shape[1], M)
    index.hnsw.efConstruction = ef_construction
    t0 = time.perf_counter()
    index.add(xb)
    build_s = time.perf_counter() - t0
    index.hnsw.efSearch = ef_search
    _, I = index.search(xq, K)
    return {
        "index": f"HNSW_M{M}_ef{ef_search}",
        "recall_at_1": round(recall_at_1(I, gt), 4),
        "recall_at_10": round(recall_at_k(I, gt), 4),
        "ms_per_query": round(ms_per_query(index, xq), 4),
        "build_time_s": round(build_s, 3),
        "memory_kb": memory_kb(),
    }


def bench_ivf_flat(xb, xq, xt, gt, nlist=1024, nprobe=16):
    quantizer = faiss.IndexFlatL2(xb.shape[1])
    index = faiss.IndexIVFFlat(quantizer, xb.shape[1], nlist)
    t0 = time.perf_counter()
    index.train(xt)
    index.add(xb)
    build_s = time.perf_counter() - t0
    index.nprobe = nprobe
    _, I = index.search(xq, K)
    return {
        "index": f"IVFFlat_nlist{nlist}_nprobe{nprobe}",
        "recall_at_1": round(recall_at_1(I, gt), 4),
        "recall_at_10": round(recall_at_k(I, gt), 4),
        "ms_per_query": round(ms_per_query(index, xq), 4),
        "build_time_s": round(build_s, 3),
        "memory_kb": memory_kb(),
    }


def bench_ivfpq(xb, xq, xt, gt, nlist=1024, m=8, bits=8, nprobe=16):
    quantizer = faiss.IndexFlatL2(xb.shape[1])
    index = faiss.IndexIVFPQ(quantizer, xb.shape[1], nlist, m, bits)
    t0 = time.perf_counter()
    index.train(xt)
    index.add(xb)
    build_s = time.perf_counter() - t0
    index.nprobe = nprobe
    _, I = index.search(xq, K)
    return {
        "index": f"IVFPQ_m{m}_nprobe{nprobe}",
        "recall_at_1": round(recall_at_1(I, gt), 4),
        "recall_at_10": round(recall_at_k(I, gt), 4),
        "ms_per_query": round(ms_per_query(index, xq), 4),
        "build_time_s": round(build_s, 3),
        "memory_kb": memory_kb(),
    }


# ---------------------------------------------------------------------------
# C++ Google Benchmark perf_tests (multi-threaded)
# ---------------------------------------------------------------------------

def run_cpp_perf_tests(results):
    cpp_benches = [
        "bench_scalar_quantizer_distance",
        "bench_scalar_quantizer_encode",
        "bench_scalar_quantizer_decode",
    ]
    for name in cpp_benches:
        binary = os.path.join(PERF_TESTS_DIR, name)
        if not os.path.isfile(binary):
            print(f"  [skip] {name} not found", file=sys.stderr)
            continue
        print(f"  {name}")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = f.name
        try:
            subprocess.run(
                [binary,
                 "--benchmark_out=" + tmp,
                 "--benchmark_out_format=json",
                 "--benchmark_repetitions=1"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            with open(tmp) as f:
                data = json.load(f)
        finally:
            os.unlink(tmp)

        for bench in data.get("benchmarks", []):
            if bench.get("run_type") == "aggregate":
                continue
            short = bench["name"].replace("/", "_").replace(" ", "_")
            prefix = name.replace("bench_scalar_quantizer_", "sq_")
            results.append({
                "index": f"cpp_{prefix}_{short}",
                "kernel_cpu_ns": round(bench["cpu_time"], 2),
            })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Using {NUM_THREADS} threads (all available cores)")

    xb, xq, xt, gt = load_dataset()
    print(f"Dataset: SIFT1M  xb={xb.shape}  xq={xq.shape}")

    results = []

    configs = [
        ("Flat (exact baseline)",    bench_flat,     {}),
        ("HNSW M=32 efSearch=16",    bench_hnsw,     {"M": 32, "ef_search": 16}),
        ("HNSW M=32 efSearch=64",    bench_hnsw,     {"M": 32, "ef_search": 64}),
        ("HNSW M=32 efSearch=128",   bench_hnsw,     {"M": 32, "ef_search": 128}),
        ("IVFFlat nprobe=16",        bench_ivf_flat, {"nprobe": 16}),
        ("IVFFlat nprobe=64",        bench_ivf_flat, {"nprobe": 64}),
        ("IVFPQ m=8 nprobe=16",      bench_ivfpq,    {"m": 8,  "nprobe": 16}),
        ("IVFPQ m=8 nprobe=64",      bench_ivfpq,    {"m": 8,  "nprobe": 64}),
    ]

    for label, fn, kwargs in configs:
        print(f"  {label}...")
        needs_train = fn in (bench_ivf_flat, bench_ivfpq)
        row = fn(xb, xq, xt, gt, **kwargs) if needs_train else fn(xb, xq, gt, **kwargs)
        results.append(row)
        print(f"    recall@1={row['recall_at_1']:.4f}  "
              f"recall@10={row['recall_at_10']:.4f}  "
              f"ms/q={row['ms_per_query']:.4f}  "
              f"build={row['build_time_s']:.2f}s  "
              f"mem={row['memory_kb']}KB")

    print("\nRunning C++ kernel microbenchmarks...")
    run_cpp_perf_tests(results)

    numeric_results = [
        {k: v for k, v in row.items() if isinstance(v, (int, float))}
        for row in results
    ]
    with open(OUTPUT_FILE, "w") as f:
        json.dump(numeric_results, f, indent=2)

    print(f"\nWrote {len(results)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()