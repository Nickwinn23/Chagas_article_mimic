### PCA Plot ###

# Imports
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler

# Read in data
# Data can be downloaded from original paper
df = pd.read_csv("")

# Drop rows with empty index
df['Sample_ID'].replace(' ', np.nan, inplace=True)
df.dropna(subset=['Sample_ID'], inplace=True)

# Drop Average rows and copy a new dataframe
df_new = df[~df['Sample_ID'].str.startswith('Average')].copy()

# New column to show what group each sample belongs to
df_new['Source'] = df_new['Sample_ID'].apply(lambda x: 'HD' if x.startswith('HD') else ('CCC' if x.startswith('CCC') else 'IND'))

# Replace all empty strings with zeros in the DataFrame
df_new.replace(' ', 0, inplace=True)

# Filter out 'IND' samples and reset the index
df_filtered = df_new[df_new['Source'] != 'IND'].reset_index(drop=True)

# Store non-numerical columns as variables for later
gene_column = df_filtered['Sample_ID']
source_column = df_filtered['Source']

# Drop non-numerical columns
df_pca = df_filtered.drop(['Sample_ID', 'Source'], axis=1)

# Standardize the data
scaler = StandardScaler(with_mean=True)
df_pca_standardized = scaler.fit_transform(df_pca)

# Run PCA
pca = PCA(n_components=4)  
pca_result = pca.fit_transform(df_pca_standardized)

# Extract gene names
column_names = df_pca.columns

# Extract PC1 loadings
pc1_loadings = pca.components_[0]
loadings_df = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(pca.n_components_)], index=column_names)

# Identify genes with positive and negative correlations with principal component 1
positive_genes = loadings_df[loadings_df['PC1'] > 0]['PC1'].sort_values(ascending=False).head(26).index
negative_genes = loadings_df[loadings_df['PC1'] < 0]['PC1'].sort_values().head(10).index
                             
# Output the results
print("Genes positively correlated with PC1: " + str(positive_genes.tolist()))
print("Genes negatively correlated with PC1: " + str(negative_genes.tolist()))

# Create dataframe of principal component values
pca_df = pd.DataFrame(data=pca_result, columns=['PC' + str(i) for i in range(1, pca_result.shape[1] + 1)])

# Add source and gene column back
pca_df['Source'] = source_column
pca_df['Sample_ID'] = gene_column

# Print the explained variance for each principal component
print("Explained Variance:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print("PC" + str(i + 1) + ": " + str(var))

# Conduct a two-tailed unpaired t-test between CCC I patients and healthy donors for PC1 scores
ccc_i_scores = pca_df.loc[pca_df['Source'] == 'CCC', 'PC1']
hd_scores = pca_df.loc[pca_df['Source'] == 'HD', 'PC1']

t_statistic, p_value = ttest_ind(ccc_i_scores, hd_scores)
print("Two-tailed unpaired t-test for PC1 scores: p-value = " + str(p_value))

# Conduct a two-tailed unpaired t-test between CCC I patients and healthy donors for PC2 scores
ccc_i_scores = pca_df.loc[pca_df['Source'] == 'CCC', 'PC2']
hd_scores = pca_df.loc[pca_df['Source'] == 'HD', 'PC2']

t_statistic, p_value = ttest_ind(ccc_i_scores, hd_scores)
print("Two-tailed unpaired t-test for PC2 scores: p-value = " + str(p_value))

# Conduct a two-tailed unpaired t-test between CCC I patients and healthy donors for PC3 scores
ccc_i_scores = pca_df.loc[pca_df['Source'] == 'CCC', 'PC3']
hd_scores = pca_df.loc[pca_df['Source'] == 'HD', 'PC3']

t_statistic, p_value = ttest_ind(ccc_i_scores, hd_scores)
print("Two-tailed unpaired t-test for PC3 scores: p-value = " + str(p_value))

# Conduct a two-tailed unpaired t-test between CCC I patients and healthy donors for PC4 scores
ccc_i_scores = pca_df.loc[pca_df['Source'] == 'CCC', 'PC4']
hd_scores = pca_df.loc[pca_df['Source'] == 'HD', 'PC4']

t_statistic, p_value = ttest_ind(ccc_i_scores, hd_scores)
print("Two-tailed unpaired t-test for PC4 scores: p-value = " + str(p_value))

# Set the color palette
palette = {'HD': 'blue', 'CCC': 'red'}

# Scatter plot PC1 vs. PC2
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Source', data=pca_df, palette=palette)
plt.title('PC1 vs. PC2')
plt.xlabel('Principal Component 1 (PC1) - Explained Variance: ' + str(round(pca.explained_variance_ratio_[0]*100, 1)) + '%')
plt.ylabel('Principal Component 2 (PC2) - Explained Variance: ' + str(round(pca.explained_variance_ratio_[1]*100, 1)) + '%')
plt.legend()
plt.show()

# Scatter plot PC1 vs. PC3
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC3', hue='Source', data=pca_df, palette=palette)
plt.title('PC1 vs. PC3')
plt.xlabel('Principal Component 1 (PC1) - Explained Variance: ' + str(round(pca.explained_variance_ratio_[0]*100, 1)) + '%')
plt.ylabel('Principal Component 3 (PC3) - Explained Variance: ' + str(round(pca.explained_variance_ratio_[2]*100, 1)) + '%')
plt.legend()
plt.show()

# 3D Scatter plot for PC1, PC2, and PC3
fig = plt.figure(figsize=(14, 14))
ax = fig.add_subplot(111, projection='3d')

# Iterate through rows and set color based on the "Source" column
for index, row in pca_df.iterrows():
    color = 'red' if row['Source'] == 'CCC' else 'blue'
    ax.scatter(row['PC1'], row['PC2'], row['PC3'], c=color, marker='o')

ax.set_xlabel('PC1 - Explained Variance: ' + str(round(pca.explained_variance_ratio_[0]*100, 1)) + '%')
ax.set_ylabel('PC2 - Explained Variance: ' + str(round(pca.explained_variance_ratio_[1]*100, 1)) + '%')
ax.set_zlabel('PC3 - Explained Variance: ' + str(round(pca.explained_variance_ratio_[2]*100, 1)) + '%')
ax.set_title('3D Scatter Plot of PC1, PC2, and PC3')

# Create custom legend
ax.scatter([], [], [], c='red', label='CCC')
ax.scatter([], [], [], c='blue', label='HD')
ax.legend()

plt.show()



### Volcano Plot ###

# Imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Read in dataframe as df
# Data can downloaded from original paper
df = pd.read_csv("")

# This is how the paper calculated log fold change

#               (avg gene expression CCC)
# Log2FC = log2 (-----------------------)
#               (avg gene expression HD )

### Calculate Log2FC ###
# Select row of averages for CCC and HD samples
average_ccc_row = df[df['Sample_ID'] == 'Average CCC I']
average_hd_row = df[df['Sample_ID'] == 'Average HD']

# locate integers, convert to floats, take mean
average_ccc = average_ccc_row.iloc[:, 1:].astype(float).mean()
average_hd = average_hd_row.iloc[:, 1:].astype(float).mean()

# Calculate LOG2FC
LOG2FC = np.log2(average_ccc/average_hd)
# Negative value means downregulation of CCC samples compared to HD

### Calculate -log10pvalue ###
# Idenfity healthy donor (HD) samples and chronic chagas disease with cardiac alterations patients (CCC)
hd_rows = df[df['Sample_ID'].fillna('').str.startswith('HD')]
ccc_rows = df[df['Sample_ID'].fillna('').str.startswith('CCC')]

# Use iloc to locate only numeric values
hd_value = hd_rows.iloc[2:, 1:]
ccc_value = ccc_rows.iloc[2:, 1:]

# Fill NaN values
hd_values = hd_value.fillna(0)
ccc_values = ccc_value.fillna(0)

# Convert dataframe into array for ttest
hd_array = hd_values.to_numpy()
ccc_array = ccc_values.to_numpy()

# Replace emptry values with 0
hd_array = np.where(hd_array == ' ', 0, hd_array)
ccc_array = np.where(ccc_array == ' ', 0, ccc_array)

# Convert the arrays to numeric values
hd_array = hd_array.astype(float)
ccc_array = ccc_array.astype(float)

# Perform the unpaired two-tailed t-test for each gene
t_statistic, p_value = ttest_ind(ccc_array, hd_array, axis=0, nan_policy='omit')

# Convert p-value into -log10 p-value
log_p_value = -np.log10(p_value)

# Convert nan to zero
logpvalue = np.nan_to_num(log_p_value, nan=0.0)

# Volcano plot
plt.figure(figsize=(10, 6))
plt.scatter(LOG2FC, logpvalue, alpha=0.5)

# Color data points red or blue based on upregulation or downregulation respectively
red_condition = (LOG2FC > 0.6) & (logpvalue > -np.log10(0.05))
blue_condition = (LOG2FC < -0.6) & (logpvalue > -np.log10(0.05))
plt.scatter(LOG2FC[red_condition], logpvalue[red_condition], color='red', alpha=0.5, label='Upregulated (p<0.05)')
plt.scatter(LOG2FC[blue_condition], logpvalue[blue_condition], color='blue', alpha=0.5, label='Downregulated (p<0.05)')

# Other data points are colored black
other_condition = ~(red_condition | blue_condition)
plt.scatter(LOG2FC[other_condition], logpvalue[other_condition], color='black', alpha=0.5, label='Non-differentially expressed')

# Establish horizontal p-value lines
plt.axhline(-np.log10(0.05), color='purple', linestyle='--', label='p=0.05')
plt.axhline(-np.log10(0.01), color='purple', linestyle='--', label='p=0.01')
plt.axhline(-np.log10(0.001), color='purple', linestyle='--', label='p=0.001')
plt.axhline(-np.log10(0.0001), color='purple', linestyle='--', label='p=0.0001')

# Establish vertical Log2FC lines
plt.axvline(-0.6, color='grey', linestyle='-', label='log2FC = -0.6')
plt.axvline(0.6, color='grey', linestyle='-', label='log2FC = 0.6')
plt.axvline(-1, color='grey', linestyle='-', label='log2FC = -1')
plt.axvline(1, color='grey', linestyle='-', label='log2FC = 1')

# Label each point with respective gene name
gene_names = df.columns
for i, txt in enumerate(gene_names[:len(LOG2FC)]):
    if red_condition[i] or blue_condition[i]:
        plt.text(LOG2FC[i], logpvalue[i], str(txt), fontsize=8)

# Label vertical Log2FC lines
plt.text(-0.6, 13, '-0.6', fontsize=8, ha='center')
plt.text(0.6, 13, '0.6', fontsize=8, ha='center')

# Label horizontal pvalue lines
plt.text(20.5, -np.log10(0.05), 'p=0.05', fontsize=8, va='bottom', ha='right', color='purple')
plt.text(20.5, -np.log10(0.01), 'p=0.01', fontsize=8, va='bottom', ha='right', color='purple')
plt.text(20.75, -np.log10(0.001), 'p=0.001', fontsize=8, va='bottom', ha='right', color='purple')
plt.text(21, -np.log10(0.0001), 'p=0.0001', fontsize=8, va='bottom', ha='right', color='purple')

# Label axes and title
plt.title('Volcano Plot')
plt.xlabel('Log2FC')
plt.ylabel('-Log10p-value')

# The paper has -18, 18 x-axis range
plt.xlim(-18, 18)
plt.show()