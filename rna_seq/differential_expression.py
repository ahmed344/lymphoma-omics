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

# %%
# Define the paths
ADATA_PATH = Path('/workspaces/lymphoma-omics/data/Diana/rna_seq/adata.h5ad')
RESULTS_DIR = Path('/workspaces/lymphoma-omics/data/Diana/rna_seq/differential_expression')

# condition
CONDITION = 'IDH2'
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
    refit_cooks=True,
    n_cpus=8 # Adjust based on your machine
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
    contrast=[CONDITION, f"{CONDITION}mut", f"{CONDITION}wt"],
    n_cpus=8
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
significant_genes = final_res[(final_res['padj'] < 0.05) & (abs(final_res['log2FoldChange']) > 1)]
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
    if row['padj'] < 0.05 and row['log2FoldChange'] > 1:
        return 'Upregulated (Mutated)'
    elif row['padj'] < 0.05 and row['log2FoldChange'] < -1:
        return 'Downregulated (WT)'
    else:
        return 'Not Significant'

volcano_data['significance'] = volcano_data.apply(map_color, axis=1)

# Create the Plot
plt.figure(figsize=(10, 8))

sns.scatterplot(
    data=volcano_data,
    x='log2FoldChange',
    y=-np.log10(volcano_data['padj']),
    hue='significance',
    palette={
        'Upregulated (Mutated)': '#d62728',
        'Downregulated (WT)': '#1f77b4',
        'Not Significant': 'lightgrey'
    },
    s=10,
    edgecolor=None,
    alpha=0.7
)

# Add lines and labels
plt.axhline(-np.log10(0.05), ls='--', color='black', alpha=0.3, label='p-adj = 0.05')
plt.axvline(1, ls='--', color='black', alpha=0.3)
plt.axvline(-1, ls='--', color='black', alpha=0.3)
plt.axvline(0, ls='-', color='grey', alpha=0.9, linewidth=1)  # Center the x-axis at zero

# Set the xlim so that zero is visually centered
# We'll set the axis limits to be symmetric
xmax = np.ceil(np.max(np.abs(volcano_data['log2FoldChange'])))
plt.xlim(-xmax, xmax)

plt.title(f'Volcano Plot: {CONDITION} mutated vs wild type', fontsize=20)
plt.xlabel('log2 Fold Change', fontsize=16)
plt.ylabel('-log10(Adjusted P-value)', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(alpha=0.3)
plt.legend(fontsize=14)

# Optional: Label top genes
top_upregulated_genes = significant_genes[significant_genes['log2FoldChange'] > 1].nsmallest(20, 'padj')
top_downregulated_genes = significant_genes[significant_genes['log2FoldChange'] < -1].nsmallest(20, 'padj')
top_genes = pd.concat([top_upregulated_genes, top_downregulated_genes])
for index, row in top_genes.iterrows():
    plt.text(
        row['log2FoldChange'] + 0.05,
        -np.log10(row['padj']) - 0.02,
        row['gene_name'],
        fontsize=4
    )
plt.tight_layout()
plt.savefig(RESULTS_DIR / f'differential_expression_volcano_plot_{CONDITION}.png', dpi=500)
plt.show()