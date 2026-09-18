#!/usr/bin/env python3
"""Generate/check native option ownership and reject unregistered switches."""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render(registry):
    lines = ["#pragma once", "// Generated from config/native-variants.json; no runtime environment choices."]
    for row in registry["entries"]:
        name, default = row["name"], row["default"]
        if default is not None:
            lines += [f"#ifndef {name}", f"#define {name} {default}", "#endif"]
    lines += ["#ifndef SBN3_EXPERIMENTAL_NATIVE"]
    for row in registry["entries"]:
        name, default = row["name"], row["default"]
        condition = (f"defined({name})" if default is None else
                     "CR_NP < 4 || CR_NP > 10" if name == "CR_NP" else
                     f"{name} != {48 if name == 'CR_SLOTO' else default}")
        lines += [f"#if {condition}", f'#error "{name} override requires SBN3_EXPERIMENTAL_NATIVE"', "#endif"]
    lines += ["#endif", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    registry = json.loads((ROOT / "config/native-variants.json").read_text())
    entries = {row["name"]: row for row in registry["entries"]}
    if len(entries) != len(registry["entries"]):
        raise SystemExit("duplicate native option")
    target = ROOT / "src/backend/ntt_p48/build_contract.hpp"
    wanted = render(registry)
    if args.write:
        target.write_text(wanted)
    elif not target.exists() or target.read_text() != wanted:
        raise SystemExit("stale native build contract; run check_native_variants.py --write")
    used = set()
    for path in (ROOT / "src/backend/ntt_p48").glob("*.hpp"):
        for directive in re.findall(r"^\s*#\s*(?:if|ifdef|ifndef)\b([^\n]*)", path.read_text(), re.M):
            used.update(re.findall(r"\bCR_[A-Z0-9_]+\b", directive))
    if used - entries.keys():
        raise SystemExit(f"unregistered native switches: {sorted(used - entries.keys())}")
    manifest = json.loads((ROOT / "config/sources.json").read_text())
    for unit in manifest["entries"]:
        for flag in unit.get("extra_flags", []):
            if flag.startswith(("-DCR_", "-DSBN3_EXPERIMENTAL_NATIVE", "-DSBN3_P48_LEAF")):
                raise SystemExit(f"production override in {unit['path']}: {flag}")
    print(f"OK: {len(entries)} registered native options; explicit production defaults")


if __name__ == "__main__":
    main()
