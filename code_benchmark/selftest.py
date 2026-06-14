#!/usr/bin/env python3
"""
Validate every task harness against its own reference solution.

For each tasks/*.json this splices the (private) `reference` solution into the
harness and runs it exactly as a model's answer would be run. A correct harness
must report ALL checks PASS. Anything else means the harness — not a model — is
buggy, and the task should be fixed before benchmarking.

Does not contact the llama server. Run after every `build_tasks.py` change:
    python build_tasks.py && python selftest.py
"""
import glob
import json
import os
import sys

from runners import available_languages, run_solution

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    avail = available_languages()
    paths = sorted(glob.glob(os.path.join(HERE, "tasks", "*.json")))
    if not paths:
        sys.exit("!! no tasks — run build_tasks.py first")

    ok_all = True
    for p in paths:
        with open(p) as f:
            t = json.load(f)
        if t["language"] not in avail:
            print(f"SKIP {t['id']:<26} ({t['language']} toolchain missing)")
            continue
        ref = t.get("reference")
        if not ref:
            print(f"WARN {t['id']:<26} no reference solution to self-test")
            continue
        res = run_solution(t["language"], ref, t["harness_template"], t.get("timeout_s", 15))
        n = t["num_checks"]
        passed = len(res["passed"])
        good = passed == n and not res["failed"]
        ok_all = ok_all and good
        mark = "PASS" if good else "FAIL"
        print(f"{mark} {t['id']:<26} {passed}/{n}  stage={res['stage']}")
        if not good:
            if res["failed"]:
                print(f"      failed checks: {res['failed']}")
            if res["stderr"].strip():
                print("      stderr tail:")
                for line in res["stderr"].strip().splitlines()[-12:]:
                    print(f"        {line}")

    print("\n" + ("ALL HARNESSES OK" if ok_all else "SOME HARNESSES BROKEN — fix before benchmarking"))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
