#!/usr/bin/env python3
"""Check planning metadata without invoking compilers or benchmarks."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from reference_paths import reference_root


V3 = Path(__file__).resolve().parents[1]
REPO = reference_root()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(relative):
    path = V3 / relative
    return json.loads(path.read_text(encoding="utf-8"))


def inside(base, relative):
    require(isinstance(relative, str), "path must be a string")
    require(not Path(relative).is_absolute(), f"absolute path: {relative}")
    path = (base / relative).resolve()
    require(path.is_relative_to(base.resolve()), f"path escapes root: {relative}")
    return path


def snapshot_id(rows):
    encoded = "".join(
        f"{r['path']}\0{r['bytes']}\0{r['sha256']}\n" for r in rows
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def check_documents():
    paths = [V3 / "README.md", *sorted((V3 / "docs").glob("*.md"))]
    for path in paths:
        require(path.is_file(), f"missing document: {path}")
        outside, fenced = [], False
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("```"):
                fenced = not fenced
            elif not fenced:
                outside.append(line)
        require(not fenced, f"unclosed code fence: {path.name}")
        for target in re.findall(r"\]\(([^)]+)\)", "\n".join(outside)):
            if "://" in target or target.startswith("#"):
                continue
            name = target.split("#", 1)[0]
            require((path.parent / name).exists(), f"broken link in {path.name}: {target}")
    return len(paths)


def check(verify_donors):
    if verify_donors:
        reference_root(True)
    config = read_json("config/native.json")
    sources = read_json("config/sources.json")
    donors = read_json("config/provenance/donors.json")
    snapshot = read_json("config/provenance/source-snapshot.json")
    for obj in (config, sources, donors, snapshot):
        require(obj["schema_version"] == 1, "unsupported metadata schema")
    require(config["toolchain"]["cc"] == "clang", "v1 C compiler must be clang")
    require(config["toolchain"]["cxx"] == "clang++", "v1 C++ compiler must be clang++")
    require(config["delivery"] == "native_only", "v1 delivery is native only")
    require(not sources["allow_wildcards"], "production source wildcards are forbidden")

    donor_ids = [item["id"] for item in donors["items"]]
    require(len(set(donor_ids)) == len(donor_ids), "duplicate donor id")
    rows = snapshot["files"]
    names = [r["path"] for r in rows]
    require(names == sorted(set(names)), "snapshot paths must be sorted and unique")
    require(len(rows) == snapshot["file_count"], "snapshot file count mismatch")
    require(sum(r["bytes"] for r in rows) == snapshot["total_bytes"], "snapshot size mismatch")
    require(snapshot_id(rows) == snapshot["snapshot_id"], "snapshot manifest ID mismatch")
    for item in donors["items"]:
        for name in item["sources"] + item["gate_donors"]:
            if verify_donors:
                require(inside(REPO, name).is_file(), f"missing donor: {name}")
            require(name in names, f"donor not covered by snapshot: {name}")
    for row in rows:
        path = inside(REPO, row["path"])
        if verify_donors:
            require(path.is_file(), f"snapshot source missing: {row['path']}")
        require(isinstance(row["bytes"], int) and row["bytes"] >= 0, "bad snapshot size")
        require(re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), "bad SHA-256 encoding")
        if verify_donors:
            data = path.read_bytes()
            require(len(data) == row["bytes"], f"source size changed: {row['path']}")
            require(hashlib.sha256(data).hexdigest() == row["sha256"],
                    f"source hash changed: {row['path']}")

    entries = sources["entries"]
    by_path = {}
    groups = config["groups"]
    for entry in entries:
        require(set(sources["required_entry_fields"]).issubset(entry), "incomplete TU entry")
        name = entry["path"]
        require(not any(c in name for c in "*?[]"), f"wildcard source: {name}")
        require(name not in by_path, f"duplicate TU: {name}")
        require(inside(V3, name).is_file(), f"missing registered TU: {name}")
        require(entry["language"] in ("c", "cxx", "asm", "asm_cpp"), f"bad language: {name}")
        require(entry["group"] in groups, f"unknown target group: {name}")
        require(set(entry["requires"]).issubset(groups[entry["group"]]["feature_closure"]),
                f"requires exceeds target group: {name}")
        require(set(entry["donor_ids"]).issubset(donor_ids), f"unknown donor id: {name}")
        by_path[name] = entry
    active, done = set(), set()

    def visit(name):
        require(name in by_path, f"unregistered TU dependency: {name}")
        require(name not in active, f"cyclic TU dependency: {name}")
        if name in done:
            return
        active.add(name)
        for dep in by_path[name]["depends_on"]:
            visit(dep)
        active.remove(name)
        done.add(name)

    for name in by_path:
        visit(name)
    docs = check_documents()
    mode = "hashes verified" if verify_donors else "hash verification not requested"
    print(f"OK: {docs} documents, {len(donor_ids)} donor groups, {len(rows)} source records ({mode})")
    print(f"Registered library TUs: {len(entries)}. This is metadata validation, not a build or arithmetic gate.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-donors", action="store_true")
    args = parser.parse_args()
    try:
        check(args.verify_donors)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
