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

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import anndata as ad
from pathlib import Path
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

from config_loader import get_config_section, load_config

# %%
CONFIG = load_config()
GLOBAL_CFG = get_config_section(config=CONFIG, section="global")
DE_CFG = get_config_section(
    config=CONFIG,
    section="differential_expression",
    required_keys=["adata_path", "results_dir", "deseq", "contrast", "significance", "volcano"],
)

# condition
CONDITION = GLOBAL_CFG["condition"]
ADATA_PATH = Path(DE_CFG["adata_path"])
RESULTS_DIR = Path(DE_CFG["results_dir"])
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SIGNIFICANCE_CFG = DE_CFG["significance"]
VOLCANO_CFG = DE_CFG["volcano"]
TOP_LABEL_CFG = VOLCANO_CFG["top_gene_labels"]
# %%
# load the adata object
adata = ad.read_h5ad(ADATA_PATH)
adata

# %%
# 1. Initialize the DeseqDataSet
# design_factors is the column in adata.obs you want to compare
dds = DeseqDataSet(
    adata=adata,
    design=f"~{CONDITION}", 
    refit_cooks=DE_CFG["deseq"]["refit_cooks"],
    n_cpus=GLOBAL_CFG["n_cpus"],
)

# 2. Run the DESeq2 normalization and dispersion estimation
print("Running DESeq2...")
dds.deseq2()



# %%
# 3. Run the Statistical Test (Differential Expression)
# contrast = ["ColumnName", "Test_Group", "Reference_Group"]
# Here we compare AITL (Test) vs Control (Ref)
stat_res = DeseqStats(
    dds, 
    contrast=[
        CONDITION,
        f"{CONDITION}{DE_CFG['contrast']['test_suffix']}",
        f"{CONDITION}{DE_CFG['contrast']['reference_suffix']}",
    ],
    n_cpus=GLOBAL_CFG["n_cpus"],
)

# Calculate p-values and fold changes
stat_res.summary()

# %%
# 4. Extract the results
results_df = stat_res.results_df
results_df.head()

# %%
final_res = results_df.join(adata.var[['gene_name', 'gene_biotype']])
# Now you can easily filter for significant genes:
significant_genes = final_res[
    (final_res["padj"] < SIGNIFICANCE_CFG["padj_max"])
    & (abs(final_res["log2FoldChange"]) > SIGNIFICANCE_CFG["abs_log2fc_min"])
]
significant_genes

# %%
# save the results
final_res.to_csv(RESULTS_DIR / f'differential_expression_results_{CONDITION}.csv')
significant_genes.to_csv(RESULTS_DIR / f'differential_expression_significant_genes_{CONDITION}.csv')

# %%
# Prepare the data
volcano_data = results_df.dropna(subset=['log2FoldChange', 'padj']).copy()

# Add a column for color categories
def map_color(row):
    """Assign a significance category for volcano plot points.

    Args:
        row (pd.Series): Differential expression result row.

    Returns:
        str: Category label used for volcano plot coloring.
    """
    padj_thresh = VOLCANO_CFG["threshold_lines"]["padj"]
    pos_thresh = VOLCANO_CFG["threshold_lines"]["log2fc_positive"]
    neg_thresh = VOLCANO_CFG["threshold_lines"]["log2fc_negative"]
    if row["padj"] < padj_thresh and row["log2FoldChange"] > pos_thresh:
        return "Upregulated (Mutated)"
    if row["padj"] < padj_thresh and row["log2FoldChange"] < neg_thresh:
        return "Downregulated (WT)"
    return "Not Significant"

volcano_data['significance'] = volcano_data.apply(map_color, axis=1)

# Create the Plot
plt.figure(figsize=tuple(VOLCANO_CFG["figsize"]))

sns.scatterplot(
    data=volcano_data,
    x='log2FoldChange',
    y=-np.log10(volcano_data['padj']),
    hue='significance',
    palette={
        "Upregulated (Mutated)": VOLCANO_CFG["colors"]["upregulated"],
        "Downregulated (WT)": VOLCANO_CFG["colors"]["downregulated"],
        "Not Significant": VOLCANO_CFG["colors"]["not_significant"],
    },
    s=VOLCANO_CFG["point_size"],
    edgecolor=None,
    alpha=VOLCANO_CFG["alpha"],
)

# Add lines and labels
plt.axhline(
    -np.log10(VOLCANO_CFG["threshold_lines"]["padj"]),
    ls="--",
    color="black",
    alpha=0.3,
    label="p-adj threshold",
)
plt.axvline(
    VOLCANO_CFG["threshold_lines"]["log2fc_positive"],
    ls="--",
    color="black",
    alpha=0.3,
)
plt.axvline(
    VOLCANO_CFG["threshold_lines"]["log2fc_negative"],
    ls="--",
    color="black",
    alpha=0.3,
)
plt.axvline(
    VOLCANO_CFG["threshold_lines"]["center_x"],
    ls="-",
    color="grey",
    alpha=0.9,
    linewidth=1,
)

# Set the xlim so that zero is visually centered
# We'll set the axis limits to be symmetric
xmax = np.ceil(np.max(np.abs(volcano_data['log2FoldChange'])))
plt.xlim(-xmax, xmax)

plt.title(
    f"Volcano Plot: {CONDITION} mutated vs wild type",
    fontsize=VOLCANO_CFG["labels"]["title_fontsize"],
)
plt.xlabel("log2 Fold Change", fontsize=VOLCANO_CFG["labels"]["axis_label_fontsize"])
plt.ylabel(
    "-log10(Adjusted P-value)",
    fontsize=VOLCANO_CFG["labels"]["axis_label_fontsize"],
)
plt.xticks(fontsize=VOLCANO_CFG["labels"]["tick_fontsize"])
plt.yticks(fontsize=VOLCANO_CFG["labels"]["tick_fontsize"])
plt.grid(alpha=0.3)
plt.legend(fontsize=VOLCANO_CFG["labels"]["legend_fontsize"])

# Optional: Label top genes
top_upregulated_genes = significant_genes[
    significant_genes["log2FoldChange"] > VOLCANO_CFG["threshold_lines"]["log2fc_positive"]
].nsmallest(TOP_LABEL_CFG["n_up"], "padj")
top_downregulated_genes = significant_genes[
    significant_genes["log2FoldChange"] < VOLCANO_CFG["threshold_lines"]["log2fc_negative"]
].nsmallest(TOP_LABEL_CFG["n_down"], "padj")
top_genes = pd.concat([top_upregulated_genes, top_downregulated_genes])
for index, row in top_genes.iterrows():
    plt.text(
        row["log2FoldChange"] + TOP_LABEL_CFG["x_offset"],
        -np.log10(row["padj"]) + TOP_LABEL_CFG["y_offset"],
        row["gene_name"],
        fontsize=TOP_LABEL_CFG["fontsize"],
    )
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / f"differential_expression_volcano_plot_{CONDITION}.png",
    dpi=VOLCANO_CFG["save_dpi"],
)
plt.show()