"""One build parallelism policy for every project entry point (no Torch import)."""
import argparse
import os


def available_cpus():
    """Respect the process affinity; do not impose a historical worker cap."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def resolve_jobs(value):
    if value == "all":
        return available_cpus()
    if value == "half":
        return max(1, (available_cpus() + 1) // 2)
    try:
        jobs = int(value)
        if str(jobs) != str(value) or jobs < 1:
            raise ValueError
        return jobs
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("build jobs must be all, half, or a positive integer") from None


def add_build_jobs(parser, option="--jobs"):
    parser.add_argument(option, type=resolve_jobs, default="all", metavar="all|half|N",
                        help="build workers: all available CPUs (default), half, or an explicit positive count")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_build_jobs(parser)
    print(parser.parse_args().jobs)
