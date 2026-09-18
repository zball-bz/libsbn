# Validation

    make check
    make check-extended
    make check-asan
    make check-tsan
    make audit

The manifest in config/tests.json declares tests and timeouts. Focused checks can
use tools/run_extended.py --only-test NAME. Arithmetic reference tests use the
repository's independent test oracle, not a GMP link in production or maintained
tests. tools/check_oracle.py cross-checks that oracle against Python integers.

The audit verifies the explicit TU list, public C symbol coverage, ISA boundaries,
allocation restrictions, dependency hashes and generated-policy ownership.
Production code generation is deterministic from committed values and templates.

Optional historical verification requires libsbn_experiments:

    make verify-donors
    make verify-evidence
    python3 tools/audit_build.py --verify-donors

Without that checkout, ordinary gates validate provenance identities and report
which external evidence was not available. They do not claim to have rechecked
missing raw measurements. Performance sweeps and their temperature/memory
protocols live in the research repository, separately from functional acceptance.
