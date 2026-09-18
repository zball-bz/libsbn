#!/usr/bin/env python3
"""Generate the complete, reviewed production profiles; never run a fit here.

Experimental fitters write candidates outside src/. Promotion edits versioned
profile data and templates, then regenerates these explicit outputs.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def experimental_output(path):
    """Fitting a partial model must not overwrite a production implementation."""
    path = Path(path).resolve()
    for directory in (ROOT / "src", ROOT / "include", ROOT / "config"):
        if path.is_relative_to(directory):
            raise ValueError("fit output must be outside production directories; "
                             "promote through config/tuning and generate_tuning.py")
    return path


def render(entry):
    profile = json.loads((ROOT / entry["profile"]).read_text())
    required = {"schema", "id", "kind", "target", "domain", "evidence", "parameters"}
    if not required <= profile.keys() or profile["schema"] != 1:
        raise ValueError(f"incomplete profile metadata: {entry['profile']}")
    text = (ROOT / entry["template"]).read_text()
    for marker, key in entry["bindings"].items():
        value = profile["parameters"][key]
        if value["type"] == "numbers":
            tokens = value["tokens"]
            if not tokens or not all(NUMBER.fullmatch(t) and math.isfinite(float(t)) for t in tokens):
                raise ValueError(f"invalid numeric tokens: {key}")
            replacement = ",".join(tokens)
        elif value["type"] == "sha256":
            replacement = value["value"]
            if not re.fullmatch(r"[0-9a-f]{64}", replacement):
                raise ValueError(f"invalid evidence digest: {key}")
        else:
            raise ValueError(f"unknown parameter kind: {key}")
        token = "SBN3_CFG_" + marker
        if not re.search(r"\b" + token + r"\b", text):
            raise ValueError(f"unused template binding: {token}")
        text = re.sub(r"\b" + token + r"\b", lambda _: replacement, text)
    if re.search(r"\bSBN3_CFG_\w+\b", text):
        raise ValueError(f"unresolved template marker: {entry['template']}")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true")
    action.add_argument("--write", action="store_true")
    action.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "config/generated.json").read_text())
    seen = set()
    for entry in manifest["outputs"]:
        output = entry["output"]
        if output in seen:
            raise ValueError(f"duplicate generated output: {output}")
        seen.add(output)
        text = render(entry)
        target = (args.out_dir or ROOT) / output
        if args.write or args.out_dir:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.read_text() != text:
                target.write_text(text)
        elif not target.exists() or target.read_text() != text:
            raise SystemExit(f"stale generated file: {output}; run generate_tuning.py --write")
        digest = hashlib.sha256(text.encode()).hexdigest()
        print(f"OK {output} {digest}")


if __name__ == "__main__":
    main()
