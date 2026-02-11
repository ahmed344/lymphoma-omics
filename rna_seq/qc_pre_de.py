from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydeseq2.dds import DeseqDataSet
from sklearn.decomposition import PCA


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for RNA-seq QC.

    Args:
        None (type: None): Arguments are parsed from the command line.

    Returns:
        argparse.Namespace: Parsed arguments used to configure the QC run.
    """
    parser = argparse.ArgumentParser(
        description="Pre-DE RNA-seq quality-control workflow on AnnData counts."
    )
    parser.add_argument(
        "--adata-path",
        type=Path,
        default=Path("/workspaces/lymphoma-omics/data/Diana/rna_seq/adata.h5ad"),
        help="Path to input AnnData (.h5ad) with raw counts.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/workspaces/lymphoma-omics/data/Diana/rna_seq/qc_pre_de"),
        help="Directory to store QC outputs.",
    )
    parser.add_argument(
        "--condition-col",
        type=str,
        default="Condition",
        help="Column in adata.obs for biological condition.",
    )
    parser.add_argument(
        "--patient-col",
        type=str,
        default="Patient_ID",
        help="Column in adata.obs for patient identifier.",
    )
    parser.add_argument(
        "--batch-col",
        type=str,
        default=None,
        help="Optional column in adata.obs for sequencing batch.",
    )
    parser.add_argument(
        "--lane-col",
        type=str,
        default=None,
        help="Optional column in adata.obs for sequencing lane.",
    )
    parser.add_argument(
        "--min-samples-expressed",
        type=int,
        default=2,
        help="Minimum number of samples in which a gene must be non-zero.",
    )
    parser.add_argument(
        "--min-library-size",
        type=float,
        default=100000.0,
        help="Absolute minimum sample library size threshold.",
    )
    parser.add_argument(
        "--mad-multiplier",
        type=float,
        default=3.0,
        help="MAD multiplier for robust low-library threshold.",
    )
    parser.add_argument(
        "--n-pcs",
        type=int,
        default=2,
        help="Number of principal components to compute (>=2 recommended).",
    )
    parser.add_argument(
        "--n-cpus",
        type=int,
        default=8,
        help="Number of CPUs to use in PyDESeq2.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible PCA.",
    )
    return parser.parse_args()


def validate_obs_columns(
    obs: pd.DataFrame,
    condition_col: str,
    patient_col: Optional[str],
    batch_col: Optional[str],
    lane_col: Optional[str],
) -> None:
    """Validate required and optional sample metadata columns.

    Args:
        obs (pd.DataFrame): Sample metadata table from ``adata.obs``.
        condition_col (str): Required condition column name.
        patient_col (Optional[str]): Optional patient ID column name.
        batch_col (Optional[str]): Optional batch column name.
        lane_col (Optional[str]): Optional lane column name.

    Returns:
        None: Raises ``ValueError`` when required columns are missing.
    """
    missing_required = [col for col in [condition_col] if col not in obs.columns]
    if missing_required:
        raise ValueError(
            f"Missing required metadata column(s): {missing_required}. "
            "Please provide a valid --condition-col."
        )

    for optional_col, cli_name in [
        (patient_col, "--patient-col"),
        (batch_col, "--batch-col"),
        (lane_col, "--lane-col"),
    ]:
        if optional_col is not None and optional_col not in obs.columns:
            raise ValueError(
                f"Metadata column '{optional_col}' from {cli_name} was not found in adata.obs."
            )


def to_counts_dataframe(adata_obj: ad.AnnData) -> pd.DataFrame:
    """Convert ``AnnData.X`` counts matrix to a dense DataFrame.

    Args:
        adata_obj (ad.AnnData): Input AnnData with samples in rows and genes in columns.

    Returns:
        pd.DataFrame: Dense counts DataFrame with sample index and gene columns.
    """
    matrix = adata_obj.X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    counts_df = pd.DataFrame(
        np.asarray(matrix), index=adata_obj.obs_names, columns=adata_obj.var_names
    )
    return counts_df


def compute_sample_metrics(counts_df: pd.DataFrame) -> pd.DataFrame:
    """Compute sample-level QC metrics from raw counts.

    Args:
        counts_df (pd.DataFrame): Raw counts with samples as rows and genes as columns.

    Returns:
        pd.DataFrame: Sample metrics including library size and sparsity.
    """
    total_counts = counts_df.sum(axis=1)
    zero_fraction = (counts_df == 0).sum(axis=1) / counts_df.shape[1]
    metrics = pd.DataFrame(
        {
            "sample_id": counts_df.index,
            "library_size": total_counts.values,
            "zero_fraction": zero_fraction.values,
        }
    ).set_index("sample_id")
    return metrics


def compute_gene_metrics(counts_df: pd.DataFrame) -> pd.DataFrame:
    """Compute gene-level QC metrics from raw counts.

    Args:
        counts_df (pd.DataFrame): Raw counts with samples as rows and genes as columns.

    Returns:
        pd.DataFrame: Gene metrics including non-zero sample count and sparsity.
    """
    non_zero_samples = (counts_df > 0).sum(axis=0)
    zero_fraction = (counts_df == 0).sum(axis=0) / counts_df.shape[0]
    metrics = pd.DataFrame(
        {
            "gene_id": counts_df.columns,
            "non_zero_samples": non_zero_samples.values,
            "zero_fraction": zero_fraction.values,
        }
    ).set_index("gene_id")
    return metrics


def robust_low_library_threshold(
    library_sizes: pd.Series,
    min_library_size: float,
    mad_multiplier: float,
) -> float:
    """Calculate low-library threshold with robust and absolute cutoffs.

    Args:
        library_sizes (pd.Series): Per-sample total count values.
        min_library_size (float): Absolute minimum accepted library size.
        mad_multiplier (float): Multiplier applied to median absolute deviation.

    Returns:
        float: Final threshold used to flag low-library samples.
    """
    median_val = float(np.median(library_sizes.values))
    mad_val = float(np.median(np.abs(library_sizes.values - median_val)))
    robust_threshold = median_val - mad_multiplier * mad_val
    return max(float(min_library_size), robust_threshold)


def filter_low_expression_genes(
    counts_df: pd.DataFrame,
    min_samples_expressed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Filter genes with too few non-zero samples.

    Args:
        counts_df (pd.DataFrame): Raw counts with samples as rows and genes as columns.
        min_samples_expressed (int): Minimum number of samples where a gene must be non-zero.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Filtered counts DataFrame and boolean mask of kept genes.
    """
    keep_mask = (counts_df > 0).sum(axis=0) >= min_samples_expressed
    filtered_counts = counts_df.loc[:, keep_mask]
    return filtered_counts, keep_mask


def normalize_and_log_transform(
    counts_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    condition_col: str,
    n_cpus: int,
) -> pd.DataFrame:
    """Run DESeq2 size-factor normalization and log2 transform for QC distances.

    Args:
        counts_df (pd.DataFrame): Count matrix with samples as rows and genes as columns.
        obs_df (pd.DataFrame): Sample metadata table aligned to ``counts_df`` rows.
        condition_col (str): Condition column used in DESeq2 design formula.
        n_cpus (int): Number of CPU workers for PyDESeq2.

    Returns:
        pd.DataFrame: Log2-transformed normalized counts for QC analyses.
    """
    design = f"~{condition_col}"
    qc_adata = ad.AnnData(
        X=counts_df.to_numpy(dtype=np.int64),
        obs=obs_df.copy(),
        var=pd.DataFrame(index=counts_df.columns),
    )

    dds = DeseqDataSet(
        adata=qc_adata,
        design=design,
        refit_cooks=True,
        n_cpus=n_cpus,
    )
    dds.deseq2()
    normed_counts = np.asarray(dds.layers["normed_counts"])
    transformed = np.log2(normed_counts + 1.0)
    transformed_df = pd.DataFrame(
        transformed, index=counts_df.index, columns=counts_df.columns
    )
    return transformed_df


def run_pca(
    transformed_df: pd.DataFrame,
    n_pcs: int,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Compute PCA coordinates from transformed expression matrix.

    Args:
        transformed_df (pd.DataFrame): Transformed matrix with samples as rows.
        n_pcs (int): Number of principal components to compute.
        random_state (int): Random seed for PCA reproducibility.

    Returns:
        tuple[pd.DataFrame, np.ndarray]: PCA coordinates DataFrame and explained variance ratio array.
    """
    pca = PCA(n_components=n_pcs, random_state=random_state)
    pcs = pca.fit_transform(transformed_df.values)
    pca_df = pd.DataFrame(
        pcs,
        index=transformed_df.index,
        columns=[f"PC{i + 1}" for i in range(n_pcs)],
    )
    return pca_df, pca.explained_variance_ratio_


def pca_outlier_flags(pca_df: pd.DataFrame) -> pd.Series:
    """Flag potential PCA outliers using Euclidean distance IQR rule.

    Args:
        pca_df (pd.DataFrame): PCA coordinates indexed by sample ID.

    Returns:
        pd.Series: Boolean outlier flag per sample ID.
    """
    distances = np.sqrt((pca_df[["PC1", "PC2"]] ** 2).sum(axis=1))
    q1 = distances.quantile(0.25)
    q3 = distances.quantile(0.75)
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr
    return distances > threshold


def correlation_outlier_flags(corr_df: pd.DataFrame) -> pd.Series:
    """Flag potential outliers using mean sample correlation IQR rule.

    Args:
        corr_df (pd.DataFrame): Sample-to-sample correlation matrix.

    Returns:
        pd.Series: Boolean outlier flag per sample ID.
    """
    np.fill_diagonal(corr_df.values, np.nan)
    mean_corr = corr_df.mean(axis=1, skipna=True)
    q1 = mean_corr.quantile(0.25)
    q3 = mean_corr.quantile(0.75)
    iqr = q3 - q1
    threshold = q1 - 1.5 * iqr
    return mean_corr < threshold


def create_category_color_series(series: pd.Series) -> pd.Series:
    """Convert categorical series values into deterministic color assignments.

    Args:
        series (pd.Series): Categorical metadata indexed by sample ID.

    Returns:
        pd.Series: Color hex strings indexed by sample ID.
    """
    categories = series.astype("string").fillna("NA").astype(str)
    unique_categories = sorted(categories.unique())
    palette = sns.color_palette("tab20", n_colors=max(2, len(unique_categories)))
    color_map = {cat: palette[i] for i, cat in enumerate(unique_categories)}
    return categories.map(color_map)


def build_clustermap_colors(
    obs_df: pd.DataFrame,
    condition_col: str,
    patient_col: Optional[str],
    batch_col: Optional[str],
    lane_col: Optional[str],
) -> Optional[pd.DataFrame]:
    """Build clustermap annotation colors for available metadata columns.

    Args:
        obs_df (pd.DataFrame): Sample metadata table indexed by sample ID.
        condition_col (str): Condition metadata column name.
        patient_col (Optional[str]): Patient metadata column name.
        batch_col (Optional[str]): Batch metadata column name.
        lane_col (Optional[str]): Lane metadata column name.

    Returns:
        Optional[pd.DataFrame]: Per-sample color annotations or ``None`` when unavailable.
    """
    color_cols: dict[str, pd.Series] = {}
    for col in [condition_col, patient_col, batch_col, lane_col]:
        if col is not None and col in obs_df.columns:
            color_cols[col] = create_category_color_series(obs_df[col])
    if not color_cols:
        return None
    return pd.DataFrame(color_cols, index=obs_df.index)


def save_library_size_plot(
    sample_metrics: pd.DataFrame,
    low_library_threshold: float,
    out_path: Path,
) -> None:
    """Save bar plot of sample library sizes with low-threshold marker.

    Args:
        sample_metrics (pd.DataFrame): Sample-level QC metrics indexed by sample ID.
        low_library_threshold (float): Threshold used to flag low-library samples.
        out_path (Path): Output path for PNG plot.

    Returns:
        None: Writes a PNG file to ``out_path``.
    """
    plot_df = sample_metrics.sort_values("library_size").reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x="sample_id", y="library_size", color="#4C78A8")
    plt.axhline(low_library_threshold, color="red", linestyle="--", linewidth=1.2)
    plt.xticks(rotation=60, ha="right", fontsize=8)
    plt.ylabel("Library size (total counts)")
    plt.xlabel("Sample")
    plt.title("Library size per sample")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_pca_plot(
    pca_with_meta: pd.DataFrame,
    condition_col: str,
    patient_col: Optional[str],
    explained_variance: np.ndarray,
    out_path: Path,
) -> None:
    """Save PCA scatter plot colored by condition and optionally styled by patient.

    Args:
        pca_with_meta (pd.DataFrame): DataFrame containing PC columns and metadata columns.
        condition_col (str): Condition column used for point color.
        patient_col (Optional[str]): Optional patient column (kept for API compatibility).
        explained_variance (np.ndarray): PCA explained variance ratio array.
        out_path (Path): Output path for PNG plot.

    Returns:
        None: Writes a PNG file to ``out_path``.
    """
    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=pca_with_meta,
        x="PC1",
        y="PC2",
        hue=condition_col,
        s=100,
    )
    plt.xlabel(f"PC1 ({explained_variance[0]:.1%})")
    plt.ylabel(f"PC2 ({explained_variance[1]:.1%})")
    plt.title("PCA quality control")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_correlation_clustermap(
    corr_df: pd.DataFrame,
    colors_df: Optional[pd.DataFrame],
    out_path: Path,
) -> None:
    """Save hierarchically clustered sample-correlation heatmap.

    Args:
        corr_df (pd.DataFrame): Sample-to-sample correlation matrix.
        colors_df (Optional[pd.DataFrame]): Optional sample annotation colors.
        out_path (Path): Output path for PNG plot.

    Returns:
        None: Writes a PNG file to ``out_path``.
    """
    cluster_grid = sns.clustermap(
        corr_df,
        cmap="viridis",
        vmin=0,
        vmax=1,
        linewidths=0.2,
        row_colors=colors_df,
        col_colors=colors_df,
        figsize=(11, 10),
    )
    cluster_grid.figure.suptitle("Sample-to-sample Pearson correlation", y=1.02)
    cluster_grid.figure.tight_layout()
    cluster_grid.savefig(out_path, dpi=300)
    plt.close(cluster_grid.figure)


def write_qc_summary(
    out_path: Path,
    n_samples: int,
    n_genes_before: int,
    n_genes_after: int,
    low_library_threshold: float,
    low_library_samples: list[str],
    pca_outliers: list[str],
    corr_outliers: list[str],
) -> None:
    """Write a concise plain-text summary of QC results.

    Args:
        out_path (Path): Output path for summary text file.
        n_samples (int): Number of analyzed samples.
        n_genes_before (int): Number of genes before filtering.
        n_genes_after (int): Number of genes after filtering.
        low_library_threshold (float): Library size threshold used for low-count flagging.
        low_library_samples (list[str]): Sample IDs flagged by low library size.
        pca_outliers (list[str]): Sample IDs flagged as PCA distance outliers.
        corr_outliers (list[str]): Sample IDs flagged as low-correlation outliers.

    Returns:
        None: Writes summary text to ``out_path``.
    """
    lines = [
        "RNA-seq pre-DE QC summary",
        f"Samples analyzed: {n_samples}",
        f"Genes before expression filter: {n_genes_before}",
        f"Genes after expression filter: {n_genes_after}",
        f"Genes removed by expression filter: {n_genes_before - n_genes_after}",
        f"Low-library threshold: {low_library_threshold:.2f}",
        f"Low-library samples ({len(low_library_samples)}): {', '.join(low_library_samples) if low_library_samples else 'None'}",
        f"PCA outlier samples ({len(pca_outliers)}): {', '.join(pca_outliers) if pca_outliers else 'None'}",
        f"Low-correlation outlier samples ({len(corr_outliers)}): {', '.join(corr_outliers) if corr_outliers else 'None'}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the complete pre-DE RNA-seq QC workflow and write outputs.

    Args:
        None (type: None): Configuration is provided through command-line arguments.

    Returns:
        None: Produces QC tables and figures in the output directory.
    """
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.adata_path)
    obs_df = adata.obs.copy()
    validate_obs_columns(
        obs=obs_df,
        condition_col=args.condition_col,
        patient_col=args.patient_col,
        batch_col=args.batch_col,
        lane_col=args.lane_col,
    )

    counts_df = to_counts_dataframe(adata_obj=adata)

    sample_metrics = compute_sample_metrics(counts_df=counts_df)
    gene_metrics = compute_gene_metrics(counts_df=counts_df)
    low_library_threshold = robust_low_library_threshold(
        library_sizes=sample_metrics["library_size"],
        min_library_size=args.min_library_size,
        mad_multiplier=args.mad_multiplier,
    )
    sample_metrics["low_library_flag"] = sample_metrics["library_size"] < low_library_threshold

    filtered_counts_df, keep_gene_mask = filter_low_expression_genes(
        counts_df=counts_df,
        min_samples_expressed=args.min_samples_expressed,
    )
    gene_metrics["pass_expression_filter"] = keep_gene_mask.values

    transformed_df = normalize_and_log_transform(
        counts_df=filtered_counts_df,
        obs_df=obs_df.loc[filtered_counts_df.index].copy(),
        condition_col=args.condition_col,
        n_cpus=args.n_cpus,
    )

    pca_df, explained_variance = run_pca(
        transformed_df=transformed_df,
        n_pcs=max(2, args.n_pcs),
        random_state=args.random_state,
    )
    corr_df = transformed_df.T.corr(method="pearson")

    pca_flags = pca_outlier_flags(pca_df=pca_df)
    corr_flags = correlation_outlier_flags(corr_df=corr_df.copy())

    sample_metrics["pca_outlier_flag"] = pca_flags.reindex(sample_metrics.index, fill_value=False).values
    sample_metrics["correlation_outlier_flag"] = corr_flags.reindex(sample_metrics.index, fill_value=False).values

    pca_with_meta = pca_df.join(obs_df, how="left")
    clustermap_colors = build_clustermap_colors(
        obs_df=obs_df,
        condition_col=args.condition_col,
        patient_col=args.patient_col,
        batch_col=args.batch_col,
        lane_col=args.lane_col,
    )

    sample_metrics.to_csv(args.out_dir / "qc_metrics_samples.csv")
    gene_metrics.to_csv(args.out_dir / "qc_metrics_genes.csv")

    filtering_summary = pd.DataFrame(
        [
            {
                "n_samples": counts_df.shape[0],
                "n_genes_before_filter": counts_df.shape[1],
                "n_genes_after_filter": filtered_counts_df.shape[1],
                "n_genes_removed": counts_df.shape[1] - filtered_counts_df.shape[1],
                "min_samples_expressed": args.min_samples_expressed,
                "low_library_threshold": low_library_threshold,
                "n_low_library_samples": int(sample_metrics["low_library_flag"].sum()),
            }
        ]
    )
    filtering_summary.to_csv(args.out_dir / "qc_filtering_summary.csv", index=False)

    pca_with_meta.to_csv(args.out_dir / "pca_coordinates.csv")
    pd.DataFrame(
        {
            "PC": [f"PC{i + 1}" for i in range(len(explained_variance))],
            "explained_variance_ratio": explained_variance,
        }
    ).to_csv(args.out_dir / "pca_explained_variance.csv", index=False)
    save_pca_plot(
        pca_with_meta=pca_with_meta,
        condition_col=args.condition_col,
        patient_col=args.patient_col,
        explained_variance=explained_variance,
        out_path=args.out_dir / "pca_condition_patient.png",
    )

    corr_df.to_csv(args.out_dir / "sample_correlation_matrix.csv")
    save_correlation_clustermap(
        corr_df=corr_df,
        colors_df=clustermap_colors,
        out_path=args.out_dir / "sample_correlation_clustermap.png",
    )
    save_library_size_plot(
        sample_metrics=sample_metrics,
        low_library_threshold=low_library_threshold,
        out_path=args.out_dir / "library_size_per_sample.png",
    )

    write_qc_summary(
        out_path=args.out_dir / "qc_summary.txt",
        n_samples=counts_df.shape[0],
        n_genes_before=counts_df.shape[1],
        n_genes_after=filtered_counts_df.shape[1],
        low_library_threshold=low_library_threshold,
        low_library_samples=sample_metrics.index[sample_metrics["low_library_flag"]].tolist(),
        pca_outliers=sample_metrics.index[sample_metrics["pca_outlier_flag"]].tolist(),
        corr_outliers=sample_metrics.index[sample_metrics["correlation_outlier_flag"]].tolist(),
    )

    print(f"QC completed. Outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
