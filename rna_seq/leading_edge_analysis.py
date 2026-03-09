# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Leading edge analysis (GSEA prerank results)
# This notebook parses leading-edge genes from GSEA prerank results and visualizes them.

# %%
from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config_loader import get_config_section, load_config, render_template


def resolve_gmt_dataset_name(gmt_path: Path, configured_name: str | None) -> str:
    """Resolve dataset name token used in enrichment output templates.

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
# ## Config

# %%
CONFIG = load_config()
GLOBAL_CFG = get_config_section(config=CONFIG, section="global", required_keys=["condition"])
GSEA_CFG = get_config_section(
    config=CONFIG,
    section="gene_enrichment",
    required_keys=["gmt_path"],
)
LE_CFG = get_config_section(
    config=CONFIG,
    section="leading_edge_analysis",
    required_keys=[
        "gsea_results_csv_template",
        "de_results_csv_template",
        "output_dir_template",
        "columns",
        "analysis",
        "outputs",
        "plots",
    ],
)
CONDITION = GLOBAL_CFG["condition"]
GMT_DATASET_NAME = resolve_gmt_dataset_name(
    gmt_path=Path(GSEA_CFG["gmt_path"]),
    configured_name=GSEA_CFG.get("gmt_dataset_name"),
)
template_values = {
    "condition": CONDITION,
    "gmt_dataset_name": GMT_DATASET_NAME,
}

GSEA_RESULTS_CSV = Path(
    render_template(LE_CFG["gsea_results_csv_template"], template_values)
)

OUT_DIR = Path(
    render_template(LE_CFG["output_dir_template"], template_values)
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DE_RESULTS_CSV = Path(
    render_template(LE_CFG["de_results_csv_template"], template_values)
)

COLUMNS_CFG = LE_CFG["columns"]
ANALYSIS_CFG = LE_CFG["analysis"]
OUTPUTS_CFG = LE_CFG["outputs"]
PLOTS_CFG = LE_CFG["plots"]

TERM_COL = COLUMNS_CFG["term"]
LEAD_GENES_COL = COLUMNS_CFG["lead_genes"]
FDR_COL = COLUMNS_CFG["fdr"]
DE_GENE_COL = COLUMNS_CFG["de_gene"]
DE_LOG2FC_COL = COLUMNS_CFG["de_log2fc"]
TOP_TERMS = ANALYSIS_CFG["top_terms"]
TOP_GENES = ANALYSIS_CFG["top_genes"]
FDR_CUTOFF = ANALYSIS_CFG["fdr_cutoff"]

# %% [markdown]
# ## Load results

# %%
results_df = pd.read_csv(GSEA_RESULTS_CSV)

# %% [markdown]
# ## Filter terms by FDR

# %%
filtered_df = results_df.copy()
if FDR_COL in filtered_df.columns:
    filtered_df = filtered_df[filtered_df[FDR_COL] <= FDR_CUTOFF].copy()

# %% [markdown]
# ## Parse leading edge genes per term

# %%
lead_genes_series = (
    filtered_df[[TERM_COL, LEAD_GENES_COL]]
    .dropna(subset=[LEAD_GENES_COL])
    .assign(
        lead_genes=lambda df: df[LEAD_GENES_COL]
        .astype(str)
        .str.split(ANALYSIS_CFG["lead_gene_delimiter"])
    )
)

lead_genes_long = lead_genes_series.explode("lead_genes").rename(
    columns={"lead_genes": "lead_gene"}
)
lead_genes_long["lead_gene"] = lead_genes_long["lead_gene"].str.strip()
lead_genes_long = lead_genes_long[lead_genes_long["lead_gene"] != ""]

# %% [markdown]
# ## Load differential expression for log2 fold change

# %%
de_df = (
    pd.read_csv(DE_RESULTS_CSV)[[DE_GENE_COL, DE_LOG2FC_COL]]
    .dropna(subset=[DE_GENE_COL])
    .set_index(DE_GENE_COL)
)
lead_genes_long = lead_genes_long.join(de_df, on="lead_gene", how="left")

# %% [markdown]
# ## Summarize leading edge genes

# %%
gene_counts = (
    lead_genes_long["lead_gene"].value_counts().rename_axis("lead_gene").reset_index(name="count")
)
term_counts = (
    lead_genes_long[TERM_COL].value_counts().rename_axis(TERM_COL).reset_index(name="lead_gene_count")
)

gene_counts.to_csv(OUT_DIR / OUTPUTS_CFG["gene_counts_csv"], index=False)
term_counts.to_csv(OUT_DIR / OUTPUTS_CFG["term_counts_csv"], index=False)
lead_genes_long.to_csv(OUT_DIR / OUTPUTS_CFG["long_table_csv"], index=False)

# %% [markdown]
# ## Visualize: Top genes by leading-edge frequency

# %%
top_genes_df = gene_counts.head(TOP_GENES)

plt.figure(figsize=tuple(PLOTS_CFG["top_genes"]["figsize"]))
plt.barh(top_genes_df["lead_gene"][::-1], top_genes_df["count"][::-1])
plt.xlabel(PLOTS_CFG["top_genes"]["xlabel"])
plt.ylabel(PLOTS_CFG["top_genes"]["ylabel"])
plt.title(PLOTS_CFG["top_genes"]["title"])
plt.gcf().subplots_adjust(left=PLOTS_CFG["top_genes"]["left_adjust"])
plt.tight_layout()
plt.savefig(
    OUT_DIR / OUTPUTS_CFG["top_genes_png"], dpi=PLOTS_CFG["top_genes"]["dpi"]
)
plt.show()

# %% [markdown]
# ## Visualize: Terms by leading-edge gene count

# %%
top_terms_df = term_counts.head(TOP_TERMS)

plt.figure(figsize=tuple(PLOTS_CFG["top_terms"]["figsize"]))
plt.barh(top_terms_df[TERM_COL][::-1], top_terms_df["lead_gene_count"][::-1])
plt.xlabel(PLOTS_CFG["top_terms"]["xlabel"])
plt.ylabel(PLOTS_CFG["top_terms"]["ylabel"])
plt.title(PLOTS_CFG["top_terms"]["title"])
plt.gcf().subplots_adjust(left=PLOTS_CFG["top_terms"]["left_adjust"])
plt.tight_layout()
plt.savefig(
    OUT_DIR / OUTPUTS_CFG["top_terms_png"], dpi=PLOTS_CFG["top_terms"]["dpi"]
)
plt.show()

# %% [markdown]
# ## Visualize: Clustered heatmap of leading-edge genes by term

# %%
heatmap_df = lead_genes_long.pivot_table(
    index=TERM_COL,
    columns="lead_gene",
    values=DE_LOG2FC_COL,
    aggfunc="mean",
)

top_heatmap_genes = gene_counts.head(TOP_GENES)["lead_gene"].tolist()
heatmap_subset = heatmap_df.reindex(columns=top_heatmap_genes).fillna(0)

max_val = max(abs(heatmap_subset.min().min()), abs(heatmap_subset.max().max()))

sns.set_theme(style="white")
cluster_grid = sns.clustermap(
    heatmap_subset,
    cmap=PLOTS_CFG["clustermap"]["cmap"],
    center=PLOTS_CFG["clustermap"]["center"],
    vmin=-max_val,
    vmax=max_val,
    figsize=tuple(PLOTS_CFG["clustermap"]["figsize"]),
    dendrogram_ratio=tuple(PLOTS_CFG["clustermap"]["dendrogram_ratio"]),
    linewidths=PLOTS_CFG["clustermap"]["linewidths"],
    linecolor=PLOTS_CFG["clustermap"]["linecolor"],
)
cluster_grid.cax.remove()
cbar_ax = cluster_grid.figure.add_axes(PLOTS_CFG["clustermap"]["cbar_axes"])
cluster_grid.figure.colorbar(
    cluster_grid.ax_heatmap.collections[0],
    cax=cbar_ax,
    label=PLOTS_CFG["clustermap"]["cbar_label"],
)
cbar_ax.yaxis.set_label_position("left")
cbar_ax.yaxis.tick_left()
cluster_grid.figure.suptitle(PLOTS_CFG["clustermap"]["title"])
cluster_grid.figure.subplots_adjust(top=0.95)
cluster_grid.savefig(
    OUT_DIR / OUTPUTS_CFG["clustermap_png"], dpi=PLOTS_CFG["clustermap"]["dpi"]
)
