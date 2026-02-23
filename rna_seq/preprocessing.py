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
import anndata as ad
import pandas as pd
import os

# %%
# Load the dataframe neglecting the first line
df = pd.read_csv('/workspaces/lymphoma-omics/data/Diana/FeatureCounts/221208_AITL_FeatureCounts_GRCh38.csv', sep='\t', skiprows=1)

# Separate the dataframe into two parts: one with the gene annotations and one with the counts
gene_annotations = df.iloc[:, :9]
counts = df.iloc[:, 9:]

# Remove the Undetermined columns
counts = counts.loc[:, ~counts.columns.str.contains('Undetermined')]

# Extract the file paths from the counts columns
file_paths = counts.columns.str.split('/').str[-1].values

# Change the name of the counts columns
counts.columns = [col[52:-26] for col in counts.columns]

# Extract the base sample name (everything before _L001 or _L002)
def get_base_name(col):
    if col.endswith('_L001') or col.endswith('_L002'):
        return col.rsplit('_', 2)[0]  # Remove the last part (_L001 or _L002)
    return col

# Group columns by their base name and sum
counts = counts.T.groupby(get_base_name).sum().T

counts.shape, gene_annotations.shape

# %%
counts.head()

# %%
gene_annotations.head()

# %%
data = []

for path in file_paths:
    # Get just the filename (e.g., "ORA27_S4_L001_bowtie2_GRCh38.sorted.bam")
    filename = os.path.basename(path)
    
    # Split by underscore
    parts = filename.split('_')
    
    # Extract parts
    sample_base = parts[0]  # "ORA27" or "AITL11"
    s_index = parts[1]      # "S4" or "S13"
    lane = parts[2]         # "L001" or "L002"
    
    # Create the ID that matches your Count Matrix columns
    # This is the most important step!
    matrix_id = f"{sample_base}"
    
    # Determine Condition based on the name
    if "AITL" in sample_base:
        source = "AITL" # Positive/Disease
    elif "ORA" in sample_base:
        source = "ORA" # Or whatever ORA stands for
    else:
        source = "Unknown"

    data.append({
        'Sample_ID': matrix_id,      # This will be your linker to the count matrix
        'Source': source,      # Crucial for DESeq2 design
        'Patient_ID': sample_base,   # Useful for batch effect checking
        'S_Index': s_index,          # 
    })

# Create DataFrame
df_metadata = pd.DataFrame(data)

# Load the metadata xlsx file
metadata = pd.read_excel('/workspaces/lymphoma-omics/data/Diana/FeatureCounts/Infos patients RNAseq AITL DLM 2.xlsx')

# Merge the metadata with the sample_metadata
df_metadata = df_metadata.merge(metadata, left_on='Patient_ID', right_on='ID EQ9', how='left')

# Drop the columns that are not needed
df_metadata = df_metadata.drop(columns=['ID EQ9'])

# DEDUPLICATE: Since L001 and L002 produce the same Sample_ID, keep only one row per sample
sample_metadata = df_metadata.drop_duplicates(subset='Sample_ID')
sample_metadata = sample_metadata.set_index('Sample_ID')

# Display the result
sample_metadata.head()

# %%
# Remove all samples with no counts
counts = counts.loc[:, counts.sum(axis=0) > 0]
sample_metadata = sample_metadata.loc[counts.columns]

# Remove all genes with no counts
gene_annotations = gene_annotations[counts.sum(axis=1) > 0]
counts = counts[counts.sum(axis=1) > 0]

# Reindex the counts and gene_annotations to the gene id
gene_annotations = gene_annotations.set_index('Geneid')
counts = counts.set_index(gene_annotations.index)

counts.shape, gene_annotations.shape, sample_metadata.shape

# %%

adata = ad.AnnData(X=counts.T, obs=sample_metadata, var=gene_annotations)

# %%
adata.var

# %%
# save the adata object
adata.write_h5ad('/workspaces/lymphoma-omics/data/Diana/rna_seq/adata.h5ad', compression='gzip')
