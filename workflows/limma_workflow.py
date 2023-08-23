import pandas as pd
import limma

# Load data
gene_expression_data = pd.read_csv('data/gene_expression_data.csv')
control_samples = pd.read_csv('data/control_samples.csv')
treatment_samples = pd.read_csv('data/treatment_samples.csv')

# Prepare expression matrix
counts_matrix = gene_expression_data.set_index('Sample_ID')
sample_info = pd.concat([control_samples, treatment_samples], ignore_index=True)

# Prepare DGEList object
dge = limma.DGEList(counts=counts_matrix, group=sample_info['Condition'])

# Normalize counts
dge = limma.calcNormFactors(dge)

# Fit linear model
design_matrix = pd.get_dummies(sample_info['Condition'], drop_first=True)
fit = limma.lmFit(dge, design_matrix)

# Perform differential expression analysis
contrast_matrix = design_matrix.copy()
contrast_matrix['Treatment'] = 1
fit = limma.contrastFit(fit, contrast=contrast_matrix)

# Get results
results = limma.eBayes(fit)
results_df = pd.DataFrame(results)
results_df.to_csv('results/limma_results.csv', index=False)

print("limma analysis complete. Results saved to 'results/limma_results.csv'.")
