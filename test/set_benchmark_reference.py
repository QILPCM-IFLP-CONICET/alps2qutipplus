import glob
import os
import platform
import sys
import subprocess

def get_platform_path():
    plat = platform.system()
    impl = platform.python_implementation()
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    bits = platform.architecture()[0]
    return f"{plat}-{impl}-{ver}-{bits}"


def get_commit_hash(ref="main"):
    return subprocess.check_output(
        ["git", "rev-parse", "--short", ref]
    ).decode().strip()

def main():
    benchmark_sets = ["gram", "commutators", "projections", "expect"]
    platform_path = get_platform_path()
    bench_dir = f".benchmarks/{platform_path}/"
    main_hash = get_commit_hash("main")


    for bench_set in benchmark_sets:
        # Find files with main's commit hash
        # The files look like: *{bench_set}*<hash>.json
        pattern = os.path.join(bench_dir, f"*{bench_set}*{main_hash}.json")
        files = sorted(glob.glob(pattern), reverse=True)

        for filename in files:
            new_filename = filename.replace(f"{main_hash}.json", "main.json")
            if not os.path.exists(new_filename):
                print(f"Linking {filename} -> {new_filename}")
                os.symlink(os.path.basename(filename), new_filename)
            else:
                print(f"Symlink {new_filename} already exists, skipping.")

if __name__ == "__main__":
    main()
