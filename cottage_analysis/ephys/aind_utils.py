from pathlib import Path
import spikeinterface as si
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_aind_analyzer(session_path):
    """
    Locates and loads the SpikeInterface SortingAnalyzer from an AIND session folder.

    Args:
        session_path (str or Path): Path to the AIND sorted session folder.

    Returns:
        si.SortingAnalyzer: The loaded analyzer.
    """
    session_path = Path(session_path)
    if not session_path.exists():
        raise FileNotFoundError(f"Session path {session_path} does not exist.")

    # AIND pipeline structure for independent runs:
    # results/postprocessing/postprocessed_{recording_name}
    postprocessed_folders = list(session_path.glob("**/postprocessed_*"))
    if not postprocessed_folders:
        if session_path.name.startswith("postprocessed_"):
            postprocessed_folders = [session_path]
        else:
            raise FileNotFoundError(
                f"Could not find any 'postprocessed_*' folder in {session_path}"
            )

    analyzer_folder = postprocessed_folders[0]
    logger.info(f"Loading SortingAnalyzer from {analyzer_folder}")
    return si.load(analyzer_folder)


def get_aind_curation_labels(session_path, analyzer=None):
    """
    Locates and reads the unit_classifier CSV from an AIND session folder.
    Optionally attaches labels to the provided analyzer.

    Args:
        session_path (str or Path): Path to the AIND sorted session folder.
        analyzer (si.SortingAnalyzer, optional): Analyzer to attach labels to.

    Returns:
        pd.DataFrame or None: Curation labels DataFrame if found.
    """
    session_path = Path(session_path)
    curation_files = list(session_path.glob("**/unit_classifier_*.csv"))

    if not curation_files:
        logger.warning("No unit_classifier_*.csv found.")
        return None

    curation_file = curation_files[0]
    logger.info(f"Loading curation labels from {curation_file}")
    curation_df = pd.read_csv(curation_file)

    if analyzer is not None and "decoder_label" in curation_df.columns:
        labels = curation_df["decoder_label"].values
        analyzer.sorting.set_property("label", labels)
        logger.info(f"Attached curation labels to analyzer units.")

    return curation_df


def load_aind_metrics(analyzer, metrics=None):
    """
    Returns quality metrics for the analyzer, computing them if missing.

    Args:
        analyzer (si.SortingAnalyzer): The SI analyzer.
        metrics (list, optional): List of metrics to compute if missing.

    Returns:
        pd.DataFrame: Quality metrics DataFrame.
    """
    if not analyzer.has_extension("quality_metrics"):
        logger.info(
            f"Quality metrics not found in analyzer. Computing {metrics or 'default'} metrics..."
        )
        analyzer.compute("quality_metrics", metric_names=metrics)

    return analyzer.get_extension("quality_metrics").get_data()


def get_curated_units(analyzer, label="SUA"):
    """
    Returns unit IDs filtered by a curation label.

    Args:
        analyzer (si.SortingAnalyzer): Analyzer with 'label' property attached.
        label (str): Label to filter by (e.g., 'SUA'). If None or empty, returns all IDs.

    Returns:
        np.ndarray: Filtered unit IDs.
    """
    if not label:
        return analyzer.unit_ids

    if "label" not in analyzer.sorting.get_property_keys():
        logger.warning("No 'label' property found in sorting. Returning all units.")
        return analyzer.unit_ids

    unit_labels = analyzer.sorting.get_property("label")
    keep_mask = unit_labels == label
    filtered_unit_ids = analyzer.unit_ids[keep_mask]

    logger.info(
        f"Filtered to {len(filtered_unit_ids)} units with label '{label}' (out of {len(analyzer.unit_ids)})"
    )
    return filtered_unit_ids
