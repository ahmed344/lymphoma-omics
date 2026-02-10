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

# %% [markdown]
# ## Config

# %%
GSEA_RESULTS_CSV = Path(
    "/workspaces/lymphoma-omics/data/Diana/rna_seq/gene_set_enrichment_analysis/gsea_hallmark_prerank/gsea_hallmark_prerank_results.csv"
)

OUT_DIR = Path(
    "/workspaces/lymphoma-omics/data/Diana/rna_seq/gene_set_enrichment_analysis/gsea_hallmark_prerank/leading_edge_analysis"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DE_RESULTS_CSV = Path(
    "/workspaces/lymphoma-omics/data/Diana/rna_seq/differential_expression/differential_expression_results.csv"
)

TERM_COL = "Term"
LEAD_GENES_COL = "Lead_genes"
FDR_COL = "FDR q-val"
DE_GENE_COL = "gene_name"
DE_LOG2FC_COL = "log2FoldChange"

TOP_TERMS = 50
TOP_GENES = 100
FDR_CUTOFF = 0.05

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
        lead_genes=lambda df: df[LEAD_GENES_COL].astype(str).str.split(";")
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

gene_counts.to_csv(OUT_DIR / "leading_edge_gene_counts.csv", index=False)
term_counts.to_csv(OUT_DIR / "leading_edge_term_counts.csv", index=False)
lead_genes_long.to_csv(OUT_DIR / "leading_edge_long_table.csv", index=False)

# %% [markdown]
# ## Visualize: Top genes by leading-edge frequency

# %%
top_genes_df = gene_counts.head(TOP_GENES)

plt.figure(figsize=(14, 18))
plt.barh(top_genes_df["lead_gene"][::-1], top_genes_df["count"][::-1])
plt.xlabel("Leading-edge frequency")
plt.ylabel("Gene")
plt.title("Top leading-edge genes across Hallmark terms")
plt.gcf().subplots_adjust(left=0.45)
plt.tight_layout()
plt.savefig(OUT_DIR / "leading_edge_top_genes.png", dpi=150)
plt.show()

# %% [markdown]
# ## Visualize: Terms by leading-edge gene count

# %%
top_terms_df = term_counts.head(TOP_TERMS)

plt.figure(figsize=(14, 18))
plt.barh(top_terms_df[TERM_COL][::-1], top_terms_df["lead_gene_count"][::-1])
plt.xlabel("Leading-edge gene count")
plt.ylabel("Term")
plt.title("Top Hallmark terms by leading-edge gene count")
plt.gcf().subplots_adjust(left=0.55)
plt.tight_layout()
plt.savefig(OUT_DIR / "leading_edge_top_terms.png", dpi=150)
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
    cmap="vlag",
    center=0,
    vmin=-max_val,
    vmax=max_val,
    figsize=(25, 8),
    dendrogram_ratio=(0.04, 0.1),
    linewidths=0.2,
    linecolor="lightgrey",
)
cluster_grid.cax.remove()
cbar_ax = cluster_grid.figure.add_axes([-0.01, 0.2, 0.008, 0.6])
cluster_grid.figure.colorbar(
    cluster_grid.ax_heatmap.collections[0],
    cax=cbar_ax,
    label="Log2 Fold Change",
)
cbar_ax.yaxis.set_label_position("left")
cbar_ax.yaxis.tick_left()
cluster_grid.figure.suptitle("Leading-edge Functional Landscape (Color = Log2FC)")
cluster_grid.figure.subplots_adjust(top=0.95)
cluster_grid.savefig(OUT_DIR / "leading_edge_clustermap.png", dpi=500)
