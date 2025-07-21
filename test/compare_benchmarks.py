import glob
import json
import platform
import subprocess
import sys


def compare_benchmarks(ref_bench, new_bench):
    summary = []
    for name in ref_bench:
        if name not in new_bench:
            summary.append(f"{name}: Missing in new benchmark")
            continue
        mean_ref, std_ref = ref_bench[name]
        mean_new, std_new = new_bench[name]
        diff = mean_new - mean_ref
        # just consider changes larger than a 5% relative to the
        # worst case, plus the sum of the standard deviations of both
        # cases:
        threshold = std_ref + std_new + 0.05 * max(mean_ref, mean_new)
        if abs(diff) > threshold:
            pct = abs(diff) / mean_ref * 100
            direction, color = (
                (
                    f"regression {pct:.2f}%",
                    "\033[91m",
                )
                if diff > 0
                else (
                    f"progress {pct:.2f}%",
                    "\033[92m",
                )
            )
            summary.append(
                f"{color}{direction.upper()} \033[0m {name}: ({mean_ref:.6f} -> {mean_new:.6f}, "
                f"Δ={diff:.6f}, threshold={threshold:.6f})"
            )
    return summary


def get_latest_bench_file(commit_hash, benchmark_set, path=None):
    print("commit hash", commit_hash)
    if path is None:
        plat = platform.system()
        impl = platform.python_implementation()
        ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        bits = platform.architecture()[0]
        platform_path = f"{plat}-{impl}-{ver}-{bits}"
        path = f".benchmarks/{platform_path}/"

    files = sorted(
        glob.glob(path + f"*{benchmark_set}*{commit_hash}.json"),
        reverse=True,
    )
    return files[0] if files else None


def get_commit_hash(ref="HEAD"):
    return (
        subprocess.check_output(["git", "rev-parse", "--short", ref]).decode().strip()
    )


def load_benchmark(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    # Map: benchmark name -> (mean, stddev)
    result = {}
    for bench in data["benchmarks"]:
        name = bench["fullname"]
        mean = bench["stats"]["mean"]
        stddev = bench["stats"]["stddev"]
        result[name] = (mean, stddev)
    return result


if __name__ == "__main__":

    if len(sys.argv) == 1:
        main_hash = get_commit_hash("main")
        current_hash = get_commit_hash("HEAD")

        results = []
        for bench_set in (
            "gram",
            "commutators",
            "projections",
            "expect",
        ):
            print("\n", 60 * "-")
            ref_file = get_latest_bench_file("main", bench_set)
            new_file = get_latest_bench_file(current_hash, bench_set)

            if ref_file and new_file:
                # Do your comparison as before
                print(f"    Comparing {ref_file} (main) to {new_file} (current branch)")
                ref_bench = load_benchmark(ref_file)
                new_bench = load_benchmark(new_file)
                results = compare_benchmarks(ref_bench, new_bench)
                if not results:
                    print("No significant changes found.")
                else:
                    for line in results:
                        print(line)
            else:
                if not ref_file:
                    print(f"{bench_set} not found for {main_hash}.")
                if not new_file:
                    print(f"{bench_set} not found for {current_hash}.")
                continue

        sys.exit(0)
    elif len(sys.argv) == 3:
        ref_file, new_file = sys.argv[1], sys.argv[2]
        ref_bench = load_benchmark(ref_file)
        new_bench = load_benchmark(new_file)
        results = compare_benchmarks(ref_bench, new_bench)
        if not results:
            print("No significant changes found.")
        else:
            for line in results:
                print(line)
        sys.exit(0)
    else:
        print(f"Usage: {sys.argv[0]}")
        print(f"Usage: {sys.argv[0]} <reference_bench.json> <new_bench.json>")
        print(
            (
                "    without parameters, compare the current branch with the last"
                " benchmark of the master branch."
            )
        )
        sys.exit(1)
