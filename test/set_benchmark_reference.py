import glob
import os
import platform
import subprocess
import sys


def get_platform_path():
    plat = platform.system()
    impl = platform.python_implementation()
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    bits = platform.architecture()[0]
    return f"{plat}-{impl}-{ver}-{bits}"


def get_commit_hash(ref="main"):
    return (
        subprocess.check_output(["git", "rev-parse", "--short", ref]).decode().strip()
    )


def main():
    benchmark_sets = ["gram", "commutators", "projections", "expect"]
    platform_path = get_platform_path()
    bench_dir = f".benchmarks/{platform_path}/"
    if len(sys.argv) == 1:
        reference_branch = "main"
        reference_hash = get_commit_hash(reference_branch)
    elif len(sys.argv) == 2:
        reference_hash = sys.argv[1]
    else:
        print(f"Usage: {sys.argv[0]}")
        print(f"Usage: {sys.argv[0]} hash")
        print(
            (
                "    without parameters, set the reference to the last"
                " benchmark of the current main."
                " If a hash is provided, the reference is set to that hash."
            )
        )
        sys.exit(1)

    for bench_set in benchmark_sets:
        # Find files with main's commit hash
        # The files look like: *{bench_set}*<hash>.json
        pattern = os.path.join(bench_dir, f"*{bench_set}*{reference_hash}.json")
        files = sorted(glob.glob(pattern), reverse=True)
        if not files:
            print(f"no benchmark files found for {bench_set} in {reference_hash}")

        for filename in files:
            new_filename = filename.replace(f"{reference_hash}.json", "main.json")
            if not os.path.exists(new_filename):
                print(f"Linking {filename} -> {new_filename}")
                os.symlink(os.path.basename(filename), new_filename)
            else:
                print(f"Symlink {new_filename} already exists, skipping.")


if __name__ == "__main__":
    main()
