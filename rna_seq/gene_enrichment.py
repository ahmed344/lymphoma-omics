# ---  # Jupytext notebook metadata header.
# jupyter:  # Jupytext metadata root.
#   jupytext:  # Jupytext configuration block.
#     formats: ipynb,py:percent  # Sync IPYNB with percent-format script.
#     text_representation:  # Jupytext text format configuration.
#       extension: .py  # Store notebook as Python script.
#       format_name: percent  # Use percent format with cell markers.
#       format_version: '1.3'  # Jupytext format version.
#       jupytext_version: 1.18.1  # Jupytext version used for metadata.
#   kernelspec:  # Kernel specification for notebook.
#     display_name: base  # Kernel display name.
#     language: python  # Kernel language.
#     name: python3  # Kernel name.
# ---  # End of Jupytext metadata header.

# %% [markdown]
# # Gene enrichment analysis (GSEA prerank)  # Notebook title.
# This notebook runs a Hallmark GSEA prerank analysis using the full list of detected genes.  # Description.

# %%
from __future__ import annotations  # Postpone evaluation of annotations.

from pathlib import Path  # Path utilities.

import pandas as pd  # Tabular data handling.
import gseapy as gp  # GSEApy library for enrichment analysis.
from gseapy.plot import barplot, dotplot  # Plot helpers for GSEA results.

from config_loader import get_config_section, load_config, render_template


def resolve_gmt_dataset_name(gmt_path: Path, configured_name: str | None) -> str:
    """Resolve dataset name token used in output directory templates.

    Args:
        gmt_path (Path): Filesystem path to the selected GMT file.
        configured_name (str | None): Optional explicit dataset name from config.

    Returns:
        str: Dataset name token for template substitution.
    """
    if configured_name:
        return configured_name
    return gmt_path.stem.replace(".", "_")


# %% [markdown]
# ## Config  # Configuration section.

# %%
CONFIG = load_config()
GLOBAL_CFG = get_config_section(config=CONFIG, section="global", required_keys=["condition"])
GSEA_CFG = get_config_section(
    config=CONFIG,
    section="gene_enrichment",
    required_keys=[
        "results_csv_template",
        "significant_csv_template",
        "gmt_path",
        "output_dir_template",
        "required_columns",
        "rank",
        "prerank",
        "outputs",
        "plots",
        "ora",
    ],
)

CONDITION = GLOBAL_CFG["condition"]
GMT_PATH = Path(GSEA_CFG["gmt_path"])
GMT_DATASET_NAME = resolve_gmt_dataset_name(
    gmt_path=GMT_PATH,
    configured_name=GSEA_CFG.get("gmt_dataset_name"),
)
template_values = {
    "condition": CONDITION,
    "gmt_dataset_name": GMT_DATASET_NAME,
}
RESULTS_CSV = Path(render_template(GSEA_CFG["results_csv_template"], template_values))
SIGNIFICANT_CSV = Path(render_template(GSEA_CFG["significant_csv_template"], template_values))
OUT_DIR = Path(render_template(GSEA_CFG["output_dir_template"], template_values))
OUT_DIR.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists.

RUN_ORA = GSEA_CFG["ora"]["enabled"]

# %% [markdown]
# ## Load results  # Load DE results.

# %%
results_df = pd.read_csv(RESULTS_CSV)  # Read full differential expression results.

results_df = results_df.dropna(  # Remove rows with missing values.
    subset=GSEA_CFG["required_columns"]  # Required columns for ranking.
).copy()  # Copy to avoid chained assignment warnings.
results_df["gene_name"] = results_df["gene_name"].astype(str)  # Ensure gene symbols are strings.

# %% [markdown]
# ## Build ranked list (full detected gene list)  # Ranked list section.

# %%
ranked_list = results_df[["gene_name", "stat"]].copy()  # Select required columns.
ranked_list = ranked_list.groupby(  # Deduplicate gene symbols.
    GSEA_CFG["rank"]["deduplicate_by"], as_index=False  # Group by gene symbol.
)[GSEA_CFG["rank"]["score_column"]].max()  # Keep max ranking metric per gene.
ranked_list = ranked_list.sort_values(  # Sort genes by ranking metric.
    GSEA_CFG["rank"]["score_column"],
    ascending=GSEA_CFG["rank"]["sort_ascending"],
)  # End sort call.
ranked_list.head()  # Preview top ranked genes.

# %% [markdown]
# ## Run GSEA prerank  # GSEA execution section.

# %%
pre_res = gp.prerank(  # Run GSEA prerank analysis.
    rnk=ranked_list,  # Ranked list of all detected genes.
    gene_sets=str(GMT_PATH),  # Hallmark gene set GMT path.
    outdir=str(OUT_DIR),  # Output directory for GSEApy artifacts.
    seed=GSEA_CFG["prerank"]["seed"],  # Seed for reproducibility.
    min_size=GSEA_CFG["prerank"]["min_size"],  # Minimum gene set size.
    max_size=GSEA_CFG["prerank"]["max_size"],  # Maximum gene set size.
    verbose=GSEA_CFG["prerank"]["verbose"],  # Verbose logging during run.
    permutation_num=GSEA_CFG["prerank"]["permutation_num"],  # Number of permutations for enrichment analysis.
)  # End prerank call.

# %% [markdown]
# ## Save results  # Results export section.

# %%
pre_res.res2d.to_csv(  # Save GSEA results table.
    OUT_DIR / GSEA_CFG["outputs"]["results_csv"]  # Results CSV output path.
)  # Save GSEA results table.
ranked_list.to_csv(  # Save the ranked list used for GSEA.
    OUT_DIR / GSEA_CFG["outputs"]["ranked_list_csv"],  # Ranked list CSV output path.
    index=False,  # Skip index column.
)  # Save the ranked list used for GSEA.

# %% [markdown]
# ## Visualize top enriched pathways  # Visualization section.

# %%
barplot(  # Render bar plot of top enriched terms.
    pre_res.res2d,  # GSEA results DataFrame.
    column=GSEA_CFG["plots"]["column"],  # Column used for cutoff/ordering.
    title=GSEA_CFG["plots"]["title"],  # Plot title.
    cutoff=GSEA_CFG["plots"]["cutoff"],  # FDR threshold.
    top_term=GSEA_CFG["plots"]["top_term"],  # Number of top terms to show.
    figsize=tuple(GSEA_CFG["plots"]["figsize"]),  # Figure size.
    ofname=str(OUT_DIR / GSEA_CFG["outputs"]["barplot_png"]),  # Output PNG path.
)  # End barplot call.

dotplot(  # Render dot plot of top enriched terms.
    pre_res.res2d,  # GSEA results DataFrame.
    column=GSEA_CFG["plots"]["column"],  # Column used for cutoff/ordering.
    title=GSEA_CFG["plots"]["title"],  # Plot title.
    cutoff=GSEA_CFG["plots"]["cutoff"],  # FDR threshold.
    top_term=GSEA_CFG["plots"]["top_term"],  # Number of top terms to show.
    figsize=tuple(GSEA_CFG["plots"]["figsize"]),  # Figure size.
    ofname=str(OUT_DIR / GSEA_CFG["outputs"]["dotplot_png"]),  # Output PNG path.
)  # End dotplot call.

# %% [markdown]
# ## Optional ORA on significant genes (disabled by default)  # ORA section.

# %%
if RUN_ORA:  # Run only when explicitly enabled.
    sig_df = pd.read_csv(SIGNIFICANT_CSV).dropna(  # Load significant genes.
        subset=["gene_name"]  # Require gene symbol column.
    ).copy()  # Copy to avoid chained assignment warnings.
    gene_list = sig_df["gene_name"].astype(str).unique().tolist()  # Unique gene list.

    ora_res = gp.enrichr(  # Run ORA with Enrichr via gseapy.
        gene_list=gene_list,  # List of significant genes.
        gene_sets=str(GMT_PATH),  # Hallmark gene set GMT path.
        outdir=str(OUT_DIR / GSEA_CFG["ora"]["outdir_name"]),  # ORA output directory.
        cutoff=GSEA_CFG["ora"]["cutoff"],  # FDR cutoff for ORA results.
    )  # End ORA call.
    ora_res.results.to_csv(  # Save ORA results.
        OUT_DIR / GSEA_CFG["ora"]["results_csv"],  # ORA results CSV path.
        index=False,  # Skip index column.
    )  # End ORA results save.

# %%
