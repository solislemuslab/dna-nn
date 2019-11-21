## Data

Copied from `NIH/R01-neural-nets/Oct2019/preliminary-data`.

We have two datasets

- Data input (genomes) of Staph bacteria: `core_gene_alignment-narsa.aln` 
    - rows=bacterial strains (individuals)
    - columns=nucleotide sites (features)
- Labels for Staph bacteria (0=susceptible to antibiotic, 1=resistant to antibiotic): `responses-staph.csv`

- Data input (genomes) of Pseudomonas bacteria: `concatenated.fasta`
    - rows=bacterial strains (individuals)
    - columns=nucleotide sites (features)
- Labels for Pseudomonas bacteria (0=susceptible to antibiotic, 1=resistant to antibiotic): `responses-pseudo.csv`

## Analyses

Next steps: We want to fit statistical/machine-learning models to accomplish two tasks:
- feature selection: identify genes associated with antibiotic-resistance
- prediction: for a given new genome, can we predict whether it will be antibiotic-resistant or not

Methods:
- regression (we need to explore penalized regression because we have more features than individuals)
- random forest
- neural networks
- ...
