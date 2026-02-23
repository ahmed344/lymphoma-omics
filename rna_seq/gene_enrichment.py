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


# %% [markdown]
# ## Config  # Configuration section.

# %%
# condition
CONDITION = 'IDH2'  # condition
RESULTS_CSV = Path(  # Input DE results CSV path.
    f"/workspaces/lymphoma-omics/data/Diana/rna_seq/differential_expression/differential_expression_results_{CONDITION}.csv"  # CSV path.
)  # Input DE results CSV path.
SIGNIFICANT_CSV = Path(  # Input significant genes CSV path.
    f"/workspaces/lymphoma-omics/data/Diana/rna_seq/differential_expression/differential_expression_significant_genes_{CONDITION}.csv"  # CSV path.
)  # Input significant genes CSV path.
GMT_PATH = Path(  # Hallmark GMT path.
    "/workspaces/lymphoma-omics/data/Diana/rna_seq/gene_set_enrichment_analysis/gene_sets/h.all.v2026.1.Hs.symbols.gmt"  # GMT path.
)  # Hallmark GMT path.

OUT_DIR = Path(  # Output directory for GSEA results and plots.
    f"/workspaces/lymphoma-omics/data/Diana/rna_seq/gene_set_enrichment_analysis/gsea_hallmark_prerank_{CONDITION}"  # Output path.
)  # Output directory for GSEA results and plots.
OUT_DIR.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists.

RUN_ORA = False  # Toggle optional ORA on significant genes.

# %% [markdown]
# ## Load results  # Load DE results.

# %%
results_df = pd.read_csv(RESULTS_CSV)  # Read full differential expression results.

results_df = results_df.dropna(  # Remove rows with missing values.
    subset=["gene_name", "stat"]  # Required columns for ranking.
).copy()  # Copy to avoid chained assignment warnings.
results_df["gene_name"] = results_df["gene_name"].astype(str)  # Ensure gene symbols are strings.

# %% [markdown]
# ## Build ranked list (full detected gene list)  # Ranked list section.

# %%
ranked_list = results_df[["gene_name", "stat"]].copy()  # Select required columns.
ranked_list = ranked_list.groupby(  # Deduplicate gene symbols.
    "gene_name", as_index=False  # Group by gene symbol.
)["stat"].max()  # Keep max ranking metric per gene.
ranked_list = ranked_list.sort_values(  # Sort genes by ranking metric.
    "stat", ascending=False  # Highest scores first.
)  # End sort call.
ranked_list.head()  # Preview top ranked genes.

# %% [markdown]
# ## Run GSEA prerank  # GSEA execution section.

# %%
pre_res = gp.prerank(  # Run GSEA prerank analysis.
    rnk=ranked_list,  # Ranked list of all detected genes.
    gene_sets=str(GMT_PATH),  # Hallmark gene set GMT path.
    outdir=str(OUT_DIR),  # Output directory for GSEApy artifacts.
    seed=42,  # Seed for reproducibility.
    min_size=15,  # Minimum gene set size.
    max_size=500,  # Maximum gene set size.
    verbose=True,  # Verbose logging during run.
    permutation_num=10**4,  # Number of permutations for enrichment analysis.
)  # End prerank call.

# %% [markdown]
# ## Save results  # Results export section.

# %%
pre_res.res2d.to_csv(  # Save GSEA results table.
    OUT_DIR / "gsea_hallmark_prerank_results.csv"  # Results CSV output path.
)  # Save GSEA results table.
ranked_list.to_csv(  # Save the ranked list used for GSEA.
    OUT_DIR / "gsea_hallmark_prerank_ranked_list.csv",  # Ranked list CSV output path.
    index=False,  # Skip index column.
)  # Save the ranked list used for GSEA.

# %% [markdown]
# ## Visualize top enriched pathways  # Visualization section.

# %%
barplot(  # Render bar plot of top enriched terms.
    pre_res.res2d,  # GSEA results DataFrame.
    column="FDR q-val",  # Column used for cutoff/ordering.
    title="Hallmark GSEA prerank (top terms)",  # Plot title.
    cutoff=0.05,  # FDR threshold.
    top_term=50,  # Number of top terms to show.
    figsize=(8, 10),  # Figure size.
    ofname=str(OUT_DIR / "gsea_hallmark_prerank_barplot.png"),  # Output PNG path.
)  # End barplot call.

dotplot(  # Render dot plot of top enriched terms.
    pre_res.res2d,  # GSEA results DataFrame.
    column="FDR q-val",  # Column used for cutoff/ordering.
    title="Hallmark GSEA prerank (top terms)",  # Plot title.
    cutoff=0.05,  # FDR threshold.
    top_term=50,  # Number of top terms to show.
    figsize=(8, 10),  # Figure size.
    ofname=str(OUT_DIR / "gsea_hallmark_prerank_dotplot.png"),  # Output PNG path.
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
        outdir=str(OUT_DIR / "ora_significant_genes"),  # ORA output directory.
        cutoff=0.05,  # FDR cutoff for ORA results.
    )  # End ORA call.
    ora_res.results.to_csv(  # Save ORA results.
        OUT_DIR / "ora_significant_genes_results.csv",  # ORA results CSV path.
        index=False,  # Skip index column.
    )  # End ORA results save.

# %%
