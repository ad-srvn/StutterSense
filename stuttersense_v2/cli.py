"""Command-line entrypoint for StutterSense 2.0."""

from __future__ import annotations

import argparse
from dataclasses import replace

from .config import load_config
from .experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the unvalidated StutterSense 2.0 pipeline")
    parser.add_argument("--config", required=True, help="Path to a v2 JSON configuration")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), help="Override configured device")
    parser.add_argument("--output-root", help="Override the configured output root")
    parser.add_argument("--resume-from", help="Resume a matching incomplete run directory")
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    config = load_config(arguments.config)
    if arguments.device:
        config = replace(config, training=replace(config.training, device=arguments.device))
    if arguments.output_root or arguments.resume_from:
        config = replace(config, output=replace(
            config.output,
            root=arguments.output_root or config.output.root,
            resume_from=arguments.resume_from or config.output.resume_from,
        ))
    config.validate()
    output = run_experiment(config)
    print(f"Completed unvalidated run: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
