import pandas as pd
from edgeR import DGEList, calcNormFactors, estimateDisp, glmLRT

# Load data
gene_expression_data = pd.read_csv('data/gene_expression_data.csv')
control_samples = pd.read_csv('data/control_samples.csv')
treatment_samples = pd.read_csv('data/treatment_samples.csv')

# Prepare DGEList object
counts_matrix = gene_expression_data.set_index('Sample_ID').dropna(axis=1)
sample_info = pd.concat([control_samples, treatment_samples], ignore_index=True)
dge = DGEList(counts_matrix)

# Normalize counts
dge = calcNormFactors(dge)

# Estimate dispersion
design_matrix = pd.get_dummies(sample_info['Condition'], drop_first=True)
dge = estimateDisp(dge, design_matrix)

# Perform differential expression analysis
contrast_matrix = design_matrix.copy()
contrast_matrix['Treatment'] = 1
fit = glmLRT(dge, contrast=contrast_matrix)

# Get results
results = fit.table
results.to_csv('results/edgeR_results.csv', index=False)

print("edgeR analysis complete. Results saved to 'results/edgeR_results.csv'.")
