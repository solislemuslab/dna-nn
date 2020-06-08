# Visualizations

- [visualizations.ipynb](https://github.com/solislemuslab/dna-nn/blob/master/lareina/visualizations.ipynb) - Visualized the sequence and response data of both Pseudomonas aeruginosa and Staphylococcus aureus.

# Phylogenetic Analyses

- [phylo_pseudo.Rmd](https://github.com/solislemuslab/dna-nn/blob/master/lareina/phylo_pseudo.Rmd) & [phylo_staph.Rmd](https://github.com/solislemuslab/dna-nn/blob/master/lareina/phylo_staph.Rmd) - Performed phylogenetic analysis by visualizing NJ, UPGMA, and Maximum-likelihood trees, checked goodness of fit.  

# Model Training

- [pseudo_clf.ipynb](https://github.com/solislemuslab/dna-nn/blob/master/lareina/pseudo_clf.ipynb) & [staph_clf.ipynb](https://github.com/solislemuslab/dna-nn/blob/master/lareina/staph_clf.ipynb) - With binary response data, encoded gene sequences by label encoding and one-hot encoding, imputed missing nucleotides by naive imputation, then trained Logistic, SVM and Random Forest classifiers. 
- [pseudo_reg.ipynb](https://github.com/solislemuslab/dna-nn/blob/master/lareina/pseudo_reg.ipynb) & [staph_reg.ipynb](https://github.com/solislemuslab/dna-nn/blob/master/lareina/staph_reg.ipynb) - With numerical response data, encoded gene sequences by label encoding and one-hot encoding, imputed missing nucleotides by naive imputation, then trained Logistic, SVM and Random Forest regressors. 
- [results](https://github.com/solislemuslab/dna-nn/tree/master/lareina/results) - Contains measurements for the combinations of different data pre-processing options and machine learning models. 
- [model_selection.ipynb](https://github.com/solislemuslab/dna-nn/blob/master/lareina/model_selection.ipynb) - Performed full model selection decision trees in hopes of finding out the model with best prediction performance. 
