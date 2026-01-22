import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

sys.path.append(os.getcwd()) 
from cottage_analysis.analysis.gratings import analyze_grating_responses
from cottage_analysis.analysis.fit_gaussian_blob import fit_sftf_tuning

def run_cluster_analysis(project, mouse, session, protocol, input_base_dir, niter):
    print(f"Starting analysis for: {mouse} / {session}")
    
    #1. Set up paths
    projects_path = Path(input_base_dir)
    results_base = Path("/camp/lab/znamenskiyp/home/shared/projects") / project / "sftf_fitting"
    output_dir = results_base / mouse / session / protocol
    print(f"Output Directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trials_file = output_dir / "trials_df.pkl"
    neurons_file = output_dir / "neurons_df.pkl"

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

    # Stack the arrays (Response Matrix)
    response_matrix = np.stack(trials_df['dff_stim'].apply(lambda x: np.mean(x, axis=0)).values)
    
    responses_df = pd.DataFrame(
        response_matrix, 
        columns=np.arange(response_matrix.shape[1]), 
        index=trials_df.index
    )
    
    # Combine Stimulus info + Neural Responses
    trials_df_formatted = pd.concat(
        [trials_df[['SpatialFrequency', 'TemporalFrequency', 'Angle']], responses_df], 
        axis=1
    )

    #4. Clean data
    print("Clipping extremes to prevent overflows)...")
    numeric_cols = trials_df_formatted.select_dtypes(include=[np.number]).columns
    
    # Replace Infinity with NaN
    trials_df_formatted[numeric_cols] = trials_df_formatted[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # Fill NaNs with 0 (assuming silence where data is missing)
    trials_df_formatted[numeric_cols] = trials_df_formatted[numeric_cols].fillna(0)
    
    # Clip values to prevent exp() explosions (e.g. keeping dF/F between -10 and +10)
    trials_df_formatted[numeric_cols] = trials_df_formatted[numeric_cols].clip(lower=-10, upper=10)

    # 5. Run fitting
    print(f"Running Gaussian Fit with niter={niter}...")

    try:
        neurons_df = fit_sftf_tuning(trials_df_formatted, niter=niter)
    except RuntimeError as e:
        print(f"Crash during fitting: {e}")
        sys.exit(1)

    # 6. Save results
    print(f"Analysis Complete! Saving to {neurons_file}")
    neurons_df.to_pickle(neurons_file)
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

    args = parser.parse_args()

    run_cluster_analysis(
        project=args.project,
        mouse=args.mouse,
        session=args.session,
        protocol=args.protocol,
        input_base_dir=args.base_dir,
        niter=args.niter
    )