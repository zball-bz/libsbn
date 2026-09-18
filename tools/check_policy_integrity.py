#!/usr/bin/env python3
"""Audit reproducibility, numerical-domain ownership, and retired code markers."""
import argparse
import hashlib
from reference_paths import experiments_root
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Coefficient vectors are documented by their template feature order; larger
# packed tables must carry an explicit record layout.
MAX_UNLAID_TOKENS = 64


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-evidence", action="store_true", help="Require and hash the optional raw experiment evidence")
    args = parser.parse_args()
    unavailable = []
    subprocess.run(["python3", str(ROOT / "tools/generate_tuning.py"), "--check"], check=True)
    subprocess.run(["python3", str(ROOT / "tools/check_native_variants.py")], check=True)
    manifest = json.loads((ROOT / "config/generated.json").read_text())
    profiles = []
    gaps = []
    laid_out = []
    for path in sorted({entry["profile"] for entry in manifest["outputs"]}):
        profile = json.loads((ROOT / path).read_text())
        if not profile["evidence"]:
            raise SystemExit(f"missing profile provenance: {path}")
        for evidence in profile["evidence"]:
            source = experiments_root() / evidence["path"]
            if "sha256" in evidence and not re.fullmatch(r"[0-9a-f]{64}", evidence["sha256"]):
                raise SystemExit(f"invalid evidence hash: {path}")
            if not source.exists():
                if args.verify_evidence:
                    raise SystemExit(f"missing experimental evidence: {source}")
                unavailable.append({"profile": path, **evidence})
                continue
            if "sha256" in evidence and sha(source) != evidence["sha256"]:
                raise SystemExit(f"evidence changed: {source}")
        for name, parameter in profile["parameters"].items():
            if parameter["type"] != "numbers":
                continue
            if not parameter.get("unit"):
                raise SystemExit(f"missing parameter unit: {path}:{name}")
            tokens = parameter["tokens"]
            layout = parameter.get("layout")
            # Packed record tables (more than one record) must declare their
            # record stride and column meanings; a template decode stride that
            # no longer divides the token count is a schema drift, not a fit.
            if layout is None:
                if len(tokens) > MAX_UNLAID_TOKENS:
                    raise SystemExit(f"packed table without record layout: {path}:{name}")
                continue
            stride, columns = layout.get("stride"), layout.get("columns")
            if not isinstance(stride, int) or stride < 1 or not isinstance(columns, list) \
                    or len(columns) != stride or len(set(columns)) != stride:
                raise SystemExit(f"inconsistent record layout: {path}:{name}")
            if len(tokens) % stride:
                raise SystemExit(f"token count {len(tokens)} not a multiple of record stride {stride}: {path}:{name}")
            decoder = layout.get("decoder", "")
            template = decoder.split(" ")[0] if decoder else ""
            if template and not (ROOT / template).exists():
                raise SystemExit(f"layout decoder template missing: {path}:{name}")
            if template and f"%{stride}==0" not in (ROOT / template).read_text().replace(" ", ""):
                raise SystemExit(f"template decode stride does not match layout stride {stride}: {path}:{name}")
            laid_out.append({"profile": path, "parameter": name, "stride": stride, "records": len(tokens) // stride})
        gaps += [{"profile": path, **gap} for gap in profile.get("provenance_gaps", [])]
        profiles.append({"path": path, "sha256": sha(ROOT / path)})
    domain = json.loads((ROOT / "config/numeric-domain.json").read_text())
    if sha(ROOT / domain["implementation"]) != domain["implementation_sha256"]:
        raise SystemExit("numerical domain changed without its version/evidence record")
    retired = r"\b(?:fdbg|row2p_env|acol_env|tophook_on|col_kernel|side_sw|nolatt)\b"
    for path in (ROOT / "src").rglob("*"):
        if path.suffix not in (".cpp", ".hpp", ".h"):
            continue
        body = re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.S)
        if re.search(retired, body):
            raise SystemExit(f"retired experiment reintroduced: {path}")
        if path.name != "tuning.hpp" and '"backend/pq16/tuning.hpp"' in body:
            raise SystemExit(f"combined numerical/cost include in production: {path}")
    report = {"schema": 1, "status": "passed", "profiles": profiles,
              "numeric_domain": domain["id"], "recorded_provenance_gaps": gaps,
              "external_evidence_unavailable": unavailable, "raw_evidence_required": args.verify_evidence,
              "record_layouts": laid_out,
              "checks": ["deterministic production generation", "explicit native options",
                         "profile units and provenance identities (hash raw evidence when available)", "packed-table record layouts",
                         "numerical-domain ownership",
                         "retired branch markers", "separate numerical/cost includes"]}
    folder = ROOT / "build"
    folder.mkdir(exist_ok=True)
    (folder / "policy-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if unavailable:
        print(f"INFO: {len(unavailable)} external evidence entries unavailable; use --verify-evidence for the full gate.")
    print(f"OK: {len(profiles)} production profiles; {len(laid_out)} packed record tables with layouts; "
          f"{len(gaps)} inherited provenance gaps explicitly recorded")


if __name__ == "__main__":
    main()
