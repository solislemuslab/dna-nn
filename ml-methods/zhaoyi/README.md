## Folders

- [practice](https://github.com/solislemuslab/dna-nn/tree/master/zhaoyi/practice) - Playground for testing out libraries and can be ignored.
- [result](https://github.com/solislemuslab/dna-nn/tree/master/zhaoyi/result) - Stores the pre-processing options, model choices, and different scores of machine learning model in CSV formt.

## Scripts

- [load.py](https://github.com/solislemuslab/dna-nn/blob/master/zhaoyi/load.py) - Help to load and combine FASTA and CSV file into a Pandas DataFrame.
- [phylo.py](https://github.com/solislemuslab/dna-nn/blob/master/zhaoyi/phylo.py) - Help to build and draw phylogenetic trees.

## Jupyter notebooks

### Analysis

- [pseudomonas_explore.ipynb](https://github.com/solislemuslab/dna-nn/blob/master/zhaoyi/pseudomonas_explore.ipynb)/[staphylococcus_explore.ipynb](https://github.com/solislemuslab/dna-nn/blob/master/zhaoyi/staphylococcus_explore.ipynb) - Explore the data (finding characteristics of the data, plotting UPGMA phylogenetic trees, training tentative machine learning models) and imputing the whole-genome sequences naively.
- [pseudomonas_nucleotide.ipynb](https://github.com/solislemuslab/dna-nn/blob/master/zhaoyi/pseudomonas_nucleotide.ipynb)/[staphylococcus_nucleotide.ipynb](https://github.com/solislemuslab/dna-nn/blob/master/zhaoyi/staphylococcus_nucleotide.ipynb) - Perform data pre-processing using nucleotide data and store pre-processed data as Numpy array files.
- [pseudomonas_codon.ipynb](https://github.com/solislemuslab/dna-nn/blob/master/zhaoyi/pseudomonas_codon.ipynb)/[staphylococcus_codon.ipynb](https://github.com/solislemuslab/dna-nn/blob/master/zhaoyi/staphylococcus_codon.ipynb) - Perform data pre-processing using codon data and store pre-processed data as Numpy array files.
- [pseudomonas_model.ipynb](https://github.com/solislemuslab/dna-nn/blob/master/zhaoyi/pseudomonas_model.ipynb)/[staphylococcus_model.ipynb](https://github.com/solislemuslab/dna-nn/blob/master/zhaoyi/staphylococcus_model.ipynb) - Train both classifiers and regressors on all pre-processsed data files, and populate results of all models as CSV files.

Note: Some pre-processing or model training may take a while to execute.

### Plotting

- [figure.ipynb](https://github.com/solislemuslab/dna-nn/blob/master/zhaoyi/figure.ipynb) - Produce most of the figures in this project. See the notebook for those figures.
- [decision_tree.ipynb](https://github.com/solislemuslab/dna-nn/blob/master/zhaoyi/decision_tree.ipynb) - Train decision trees on pre-processing and model choices to identify the best data analysis combinations. Visualize performance of all models.