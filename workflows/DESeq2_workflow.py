import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
pandas2ri.activate()
from rpy2.robjects.packages import importr

# Load data
gene_expression_data = pd.read_csv('data/gene_expression_data.csv')
control_samples = pd.read_csv('data/control_samples.csv')
treatment_samples = pd.read_csv('data/treatment_samples.csv')

# Prepare DESeqDataSet object
counts_matrix = gene_expression_data.set_index('Sample_ID').dropna(axis=1)
sample_info = pd.concat([control_samples, treatment_samples], ignore_index=True)

DESeq2 = importr('DESeq2')
dds = DESeq2.DESeqDataSetFromMatrix(countData=counts_matrix, colData=sample_info, design=robjects.Formula('~ Condition'))

# Perform differential expression analysis
dds = DESeq2.DESeq(dds)
res = DESeq2.results(dds)
res = DESeq2.lfcShrink(dds, res=res, type="apeglm")

# Convert results to pandas DataFrame
results = pandas2ri.ri2py_dataframe(res)

# Save results
results.to_csv('results/DESeq2_results.csv', index=False)

print("DESeq2 analysis complete. Results saved to 'results/DESeq2_results.csv'.")
