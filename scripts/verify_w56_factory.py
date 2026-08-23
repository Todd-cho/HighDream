"""Verify the frozen W56 profile and the merged W53/W56/W57-59 live routing.

Compares this repo's run_unreal_inference.build_action_provider() output
against the desktop-provided reference copy (Downloads/run_unreal_inference.py,
2026-08-23) for every affected mode, and checks build_w56_config() against the
frozen W53 profile plus the documented W56 diff
(RL_TRAINING_ADDENDUM_W56_KO.txt).
"""
from __future__ import annotations

import importlib.util
from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

REFERENCE_PATH = Path(r"C:\Users\young\Downloads\run_unreal_inference.py")

EXPECTED_W56_DIFF_FROM_W53 = {
    "controller_name": ("w53", "w56"),
    "terminal_track_exit_range_m": (2100.0, 4500.0),
    "terminal_track_prelock_ata_deg": (0.0, 15.0),
    "terminal_track_prelock_range_m": (0.0, 3000.0),
    "terminal_vertical_unload_el_deg": (6.0, 0.0),
    "terminal_vertical_unload_ratio": (0.4, 0.7),
}


def load_reference_module():
    spec = importlib.util.spec_from_file_location("reference_run_unreal_inference", REFERENCE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    from dogfight.ai.rule_profiles import build_w53_config, build_w56_config
    import run_unreal_inference as ours

    ok = True

    # 1) W56 frozen profile matches W53 plus exactly the documented diff.
    b53 = asdict(build_w53_config())
    b56 = asdict(build_w56_config())
    actual_diff = {k: (b53[k], b56[k]) for k in b53 if b53[k] != b56[k]}
    if actual_diff != EXPECTED_W56_DIFF_FROM_W53:
        print("FAIL: W56 profile diverges from the documented W53->W56 diff")
        print("  expected:", EXPECTED_W56_DIFF_FROM_W53)
        print("  actual:  ", actual_diff)
        ok = False
    else:
        print(f"PASS: W56 profile == W53 profile except the {len(actual_diff)} documented fields")

    # 2) Our routing (rule_profiles shortcuts + merged big block) matches the
    #    reference file's build_action_provider() for every affected mode.
    if not REFERENCE_PATH.exists():
        print(f"SKIP: reference file not found at {REFERENCE_PATH}")
        return 0 if ok else 1

    reference = load_reference_module()
    for mode in ("w53", "w54", "w55", "w56", "w57", "w58", "w59"):
        ours_cfg = asdict(ours.build_action_provider(SimpleNamespace(mode=mode)).cfg)
        ref_cfg = asdict(reference.build_action_provider(SimpleNamespace(mode=mode)).cfg)
        if ours_cfg != ref_cfg:
            diffs = {k: (ours_cfg[k], ref_cfg[k]) for k in ours_cfg if ours_cfg[k] != ref_cfg[k]}
            print(f"FAIL: mode={mode} diverges from reference file")
            for k, v in diffs.items():
                print("   ", k, "ours=", v[0], "reference=", v[1])
            ok = False
        else:
            print(f"PASS: mode={mode} matches reference file ({len(ours_cfg)} fields)")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
