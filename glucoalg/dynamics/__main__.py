"""Lazy command dispatch for the causal dynamics workflow."""
import argparse
import importlib
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = {"collect": "collect", "branches": "branches", "train": "train",
                "forecast": "validate", "response": "response", "rollout": "rollout",
                "tune": "tuning"}
    parser.add_argument("command", choices=commands)
    args = parser.parse_args(argv[:1])
    return importlib.import_module(f"glucoalg.dynamics.{commands[args.command]}").main(argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
