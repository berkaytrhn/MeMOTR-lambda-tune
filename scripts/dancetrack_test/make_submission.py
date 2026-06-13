#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Step 2/2 — Package MeMOTR DanceTrack results into a codabench.org submission.
#
# The DanceTrack test-set evaluation server (codabench.org) requires a zip whose
# *root contains a folder named exactly "tracker"*, holding one MOT-format
# .txt file per sequence:
#
#     tracker.zip
#     └── tracker/
#         ├── dancetrack0003.txt
#         ├── dancetrack0009.txt
#         └── ...
#
# Each line of a .txt file:
#     <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,-1,-1,-1
#
# `submit_engine.py` already writes the per-sequence files in this exact line
# format to <SUBMIT_DIR>/<SPLIT>/tracker/. This script validates them and zips
# them with the required internal "tracker/" folder name (regardless of the
# source directory's actual name).
#
# Usage:
#   python scripts/dancetrack_test/make_submission.py \
#       --submit-dir ./outputs/memotr_dancetrack --split test \
#       --data-root ./dataset            # optional: check all test seqs present
#
#   # or point straight at a results folder:
#   python scripts/dancetrack_test/make_submission.py \
#       --tracker-dir ./outputs/memotr_dancetrack/test/tracker
# ------------------------------------------------------------------------------
import argparse
import os
import sys
import zipfile


def parse_args():
    p = argparse.ArgumentParser(
        description="Package DanceTrack test results into a codabench.org submission zip."
    )
    p.add_argument("--submit-dir", type=str, default=None,
                   help="Model output dir; results are read from <submit-dir>/<split>/tracker/.")
    p.add_argument("--split", type=str, default="test",
                   help="Data split that was run (default: test).")
    p.add_argument("--tracker-dir", type=str, default=None,
                   help="Directly specify the folder of <seq>.txt files "
                        "(overrides --submit-dir/--split).")
    p.add_argument("--output", type=str, default=None,
                   help="Output zip path (default: <tracker-dir>/../tracker.zip).")
    p.add_argument("--data-root", type=str, default=None,
                   help="Optional dataset root. If given, verifies that every "
                        "sequence in <data-root>/DanceTrack/<split>/ has a result file.")
    p.add_argument("--allow-missing", action="store_true",
                   help="Package anyway even if some sequences are missing "
                        "(by default an incomplete submission is rejected).")
    return p.parse_args()


def resolve_tracker_dir(args):
    if args.tracker_dir is not None:
        return os.path.abspath(args.tracker_dir)
    if args.submit_dir is None:
        sys.exit("ERROR: provide either --tracker-dir or --submit-dir.")
    return os.path.abspath(os.path.join(args.submit_dir, args.split, "tracker"))


def validate_result_file(path):
    """Return (num_lines, list_of_problem_strings) for a single result txt."""
    problems = []
    n = 0
    with open(path, "r") as f:
        for ln, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            n += 1
            fields = line.split(",")
            if len(fields) != 10:
                problems.append(f"line {ln}: expected 10 comma-separated fields, got {len(fields)}")
                continue
            # frame and id must be integers
            try:
                int(float(fields[0]))
                int(float(fields[1]))
            except ValueError:
                problems.append(f"line {ln}: frame/id are not numeric -> {fields[0]!r},{fields[1]!r}")
    return n, problems


def expected_sequences(data_root, split):
    seq_root = os.path.join(data_root, "DanceTrack", split)
    if not os.path.isdir(seq_root):
        sys.exit(f"ERROR: --data-root given but '{seq_root}' does not exist.")
    return sorted(
        d for d in os.listdir(seq_root)
        if os.path.isdir(os.path.join(seq_root, d))
    )


def main():
    args = parse_args()
    tracker_dir = resolve_tracker_dir(args)

    if not os.path.isdir(tracker_dir):
        sys.exit(f"ERROR: tracker dir not found: {tracker_dir}\n"
                 f"       Run step 1 (run_test.sh) first.")

    txt_files = sorted(f for f in os.listdir(tracker_dir) if f.endswith(".txt"))
    if not txt_files:
        sys.exit(f"ERROR: no .txt result files in {tracker_dir}.")

    result_seqs = {os.path.splitext(f)[0] for f in txt_files}

    print("=" * 62)
    print(" Packaging DanceTrack codabench submission")
    print(f"   tracker dir : {tracker_dir}")
    print(f"   sequences   : {len(txt_files)}")
    print("=" * 62)

    # --- Optional completeness check against the dataset --------------------
    if args.data_root is not None:
        exp = expected_sequences(args.data_root, args.split)
        missing = [s for s in exp if s not in result_seqs]
        extra = sorted(result_seqs - set(exp))
        print(f"   dataset has {len(exp)} '{args.split}' sequences.")
        if extra:
            print(f"   WARNING: {len(extra)} result file(s) have no matching "
                  f"sequence: {', '.join(extra)}")
        if missing:
            print(f"   MISSING result for {len(missing)} sequence(s): "
                  f"{', '.join(missing)}")
            if not args.allow_missing:
                sys.exit("ERROR: submission is incomplete. Re-run inference, or "
                         "pass --allow-missing to package anyway.")
        else:
            print("   OK: every dataset sequence has a result file.")

    # --- Validate line format & collect stats -------------------------------
    total_lines = 0
    empty_files = []
    any_problem = False
    for f in txt_files:
        path = os.path.join(tracker_dir, f)
        n, problems = validate_result_file(path)
        total_lines += n
        if n == 0:
            empty_files.append(f)
        if problems:
            any_problem = True
            print(f"   FORMAT ISSUE in {f}:")
            for pr in problems[:5]:
                print(f"       - {pr}")
            if len(problems) > 5:
                print(f"       - ... and {len(problems) - 5} more")

    if empty_files:
        print(f"   NOTE: {len(empty_files)} empty result file(s) "
              f"(no tracks): {', '.join(empty_files)}")
    if any_problem:
        sys.exit("ERROR: malformed result lines detected (expected "
                 "'<frame>,<id>,<bb_left>,<bb_top>,<bb_w>,<bb_h>,<conf>,-1,-1,-1'). "
                 "Fix before submitting.")
    print(f"   format OK: {total_lines} total result lines across "
          f"{len(txt_files)} files.")

    # --- Build the zip with internal folder forced to "tracker/" ------------
    if args.output is not None:
        out_zip = os.path.abspath(args.output)
    else:
        out_zip = os.path.join(os.path.dirname(tracker_dir), "tracker.zip")
    os.makedirs(os.path.dirname(out_zip) or ".", exist_ok=True)
    if os.path.exists(out_zip):
        os.remove(out_zip)

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in txt_files:
            # arcname forces the required "tracker/" root folder.
            zf.write(os.path.join(tracker_dir, f), arcname=os.path.join("tracker", f))

    size_mb = os.path.getsize(out_zip) / 1e6
    print("-" * 62)
    print(f"   wrote: {out_zip}  ({size_mb:.2f} MB)")
    print("   zip layout:  tracker.zip -> tracker/<seq>.txt")
    print("   Upload this zip to the DanceTrack test server on codabench.org.")
    print("=" * 62)


if __name__ == "__main__":
    main()
