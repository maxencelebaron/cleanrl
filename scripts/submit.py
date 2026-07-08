#!/usr/bin/env python3
"""
Fill a SLURM template with experiment-specific values and submit via sbatch.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path


def fill_template(
    template_path: str,
    values: dict
) -> str:
    template = Path(template_path).read_text()
    return re.sub(
        r"\{\{(\w+)\}\}", lambda m: str(values.get(m.group(1), m.group(0))),
        template
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="scripts/default.slurm_template")
    parser.add_argument("--env-ids", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--gres", default="gpu:1")
    parser.add_argument("--gpus-per-task", default="1")
    parser.add_argument("--cpus-per-gpu", default="4")
    parser.add_argument("--cpus-per-task", default="4")
    parser.add_argument("--ntasks", default="1")
    parser.add_argument("--time", default="24:00:00")
    parser.add_argument("--nodes", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    n_envs = len(args.env_ids)
    n_seeds = len(args.seeds)

    values = {
        "gres": args.gres,
        "gpus_per_task": args.gpus_per_task,
        "cpus_per_gpu": args.cpus_per_gpu,
        "cpus_per_task": args.cpus_per_task,
        "ntasks": args.ntasks,
        "time": args.time,
        "nodes": args.nodes,
        "array": f"0-{n_envs * n_seeds - 1}",
        "env_ids": "(" + " ".join(f'"{e}"' for e in args.env_ids) + ")",
        "seeds": "(" + " ".join(str(s) for s in args.seeds) + ")",
        "len_seeds": str(n_seeds),
        "command": args.command,
    }

    script = fill_template(args.template, values)

    if args.dry_run:
        print(script)
        sys.exit(0)

    result = subprocess.run(
        ["sbatch"],
        input=script,
        text=True,
        capture_output=True
    )
    print(result.stdout, result.stderr)
