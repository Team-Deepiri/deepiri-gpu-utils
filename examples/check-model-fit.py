#!/usr/bin/env python3
"""Example: check whether a specific Ollama model fits this host."""

from __future__ import annotations

import json
import sys

from deepiri_gpu_utils.model_fit import model_fit_check


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args or args[0] in ("-h", "--help"):
        print("Usage: check-model-fit.py <model> [--json]", file=sys.stderr)
        return 0 if args and args[0] in ("-h", "--help") else 2

    as_json = "--json" in args
    model = next(a for a in args if not a.startswith("-"))
    result = model_fit_check(model)

    if as_json:
        from dataclasses import asdict

        print(json.dumps(asdict(result), indent=2, sort_keys=True))
    else:
        print(f"{result.model}: {result.fit} (suitable={result.suitable})")
        print(result.reason)

    return 0 if result.suitable else 1


if __name__ == "__main__":
    raise SystemExit(main())
