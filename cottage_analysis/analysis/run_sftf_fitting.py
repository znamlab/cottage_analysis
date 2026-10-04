import argparse
import json
import subprocess
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

sys.path.append(os.getcwd())
from cottage_analysis.analysis.gratings import (
    SFTF_CLEANING,
    SFTF_SYNC,
    analyze_grating_responses,
    format_sftf_trials,
    sftf_fit_outdated,
    summarize_sftf_fit,
)
from cottage_analysis.analysis.fit_gaussian_blob import fit_sftf_tuning

META_NAME = "fit_meta.json"


def read_meta(output_dir):
    """Load fit_meta.json next to neurons_df.pkl, or None if absent/unreadable."""
    meta_file = output_dir / META_NAME
    if not meta_file.exists():
        return None
    try:
        return json.loads(meta_file.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"Could not read {meta_file} ({e}); treating as missing.")
        return None


def write_meta(output_dir, niter, summary, source="cluster"):
    """Record how this fit was produced, so a weak fit is never mistaken for a
    strong one. niter is what lets us refuse to overwrite a better fit."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        commit = None
    meta = {
        "source": source,
        "niter": niter,
        "sync": SFTF_SYNC,
        "cleaning": SFTF_CLEANING,
        **summary,
        "fitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        # pickles are version-sensitive, so record what wrote them
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    (output_dir / META_NAME).write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def run_cluster_analysis(project, mouse, session, protocol, input_base_dir, niter, force=False):
    print(f"Starting analysis for: {mouse} / {session}")
    
    #1. Set up paths
    projects_path = Path(input_base_dir)
    results_base = Path("/camp/lab/znamenskiyp/home/shared/projects") / project / "sftf_fitting"
    output_dir = results_base / mouse / session / protocol
    print(f"Output Directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Named by sync version, so trials cached with a misaligned sync are never reused
    trials_file = output_dir / f"trials_df_{SFTF_SYNC}.pkl"
    neurons_file = output_dir / "neurons_df.pkl"

    #1b. Refuse to downgrade an existing fit
    # A notebook fit (low niter) must never clobber a cluster fit (high niter),
    # and neither should a cluster re-run with fewer iterations. Run order stops
    # mattering; only fit quality does.
    existing = read_meta(output_dir)
    # A fit made with a different sync or response cleaning is not comparable,
    # whatever its niter
    outdated = sftf_fit_outdated(existing) if neurons_file.exists() else None
    if outdated:
        print(f"Existing fit is outdated ({outdated}); refitting regardless of niter.")
    elif neurons_file.exists() and existing and not force:
        if existing.get("niter", 0) >= niter:
            print(
                f"Existing fit at {neurons_file} used niter={existing.get('niter')} "
                f"(source={existing.get('source')}, {existing.get('fitted_at')}), "
                f"which is >= the requested niter={niter}."
            )
            print("Nothing to do. Pass --force to overwrite anyway.")
            return
        print(
            f"Existing fit used niter={existing.get('niter')}; "
            f"refitting with niter={niter} and overwriting."
        )

    #2. Load or extract data
    if trials_file.exists():
        print(f"Found existing trials file at {trials_file}")
        trials_df = pd.read_pickle(trials_file)
    else:
        print("Generating trials dataframe from raw data...")
        try:
            trials_df, _ = analyze_grating_responses(
                project=project,
                session=f"{mouse}_{session}",
                protocol_base=protocol
            )
            print(f"Saving trials_df...")
            trials_df.to_pickle(trials_file)
            
        except Exception as e:
            print(f"Could not extract data. Error: {e}")
            # Exit with error code 1 so SLURM knows it failed
            sys.exit(1)

    #3. Format data for fitting
    print("Formatting data for Gaussian fitter...")
   
    if trials_df.empty:
        print("Trials dataframe is empty. Check your raw data.")
        sys.exit(1)

    # One integer column per ROI; inf becomes NaN. NaN trials are left out of
    # each ROI's fit rather than filled, and nothing is clipped.
    trials_df_formatted = format_sftf_trials(trials_df)

    # 4. Run fitting
    print(f"Running Gaussian Fit with niter={niter}...")

    try:
        neurons_df = fit_sftf_tuning(trials_df_formatted, niter=niter)
    except RuntimeError as e:
        print(f"Crash during fitting: {e}")
        sys.exit(1)

    summary = summarize_sftf_fit(neurons_df)
    print(
        f"Data quality: {summary['n_rois_with_nonfinite_trials']}/{summary['n_rois']} ROIs had "
        f"NaN/inf trials (left out), {summary['n_rois_with_extreme_trials']} had |dF/F| > 10 "
        f"(kept, flagged in n_extreme_trials), {summary['n_rois_not_fit']} had too few trials to fit."
    )

    # 5. Save results
    print(f"Analysis Complete! Saving to {neurons_file}")
    neurons_df.to_pickle(neurons_file)
    meta = write_meta(output_dir, niter=niter, summary=summary, source="cluster")
    print(f"Wrote {output_dir / META_NAME}: {meta}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SFTF Analysis on Cluster")
    
    # Required Arguments
    parser.add_argument("--project", type=str, required=True, help="Project Name (e.g. toksozi_in-vivo-BRISC)")
    parser.add_argument("--mouse", type=str, required=True, help="Mouse Name (e.g. BRAC10754.8c)")
    parser.add_argument("--session", type=str, required=True, help="Session Name (e.g. S20251028)")
    
    # Optional Arguments (with defaults)
    parser.add_argument("--protocol", type=str, default="SFTF", help="Protocol Name")
    parser.add_argument("--base_dir", type=str, default="/camp/lab/znamenskiyp/home/shared/projects/", help="Path to raw data input")
    parser.add_argument("--niter", type=int, default=20, help="Number of fitting iterations (default 20)")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing fit even if it used more iterations")

    args = parser.parse_args()

    run_cluster_analysis(
        project=args.project,
        mouse=args.mouse,
        session=args.session,
        protocol=args.protocol,
        input_base_dir=args.base_dir,
        niter=args.niter,
        force=args.force
    )