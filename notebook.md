# Data

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

## Data description (meeting 1/29)
- 2 datasets:
    - Pseudomonas Matrix: 122 by 483,333; missingness rate: 9.70% on average, 17.13% highest
        - Two antibiotics: carb and toby. Each label vector is true/false
            - carb: 79% false, 21% true; highly unbalanced, so we need to take this into account when we penalized wrong predictions. That is, a naive prediction predicting everything to be false will be ~79% accurate
            - toby: 95% false, 5% true
        - 261,868 variant columns out of the 483,333
        - __~125__ different condons in total
    - Staphylococcus Matrix: 125 by 983,088; missingness rate 1.19% on average, 13.03% highest
        - One antibiotic (unnamed). The label vector is 17% true, 83% false
        - 496,218 variant columns out of 983,088
        - __~116__ different condons in total
        - only one sequence contains 10 N nucleotides

Ideas to encode the nucleotides ACGT:
- ASCII code (disadvantage: are they still treated as numbers or categories?)
- One-hot encoding: A->0001, C->0010, G->0100, T->1000, - -> 0000

Questions:
- Complete the description of the data in terms of dimensions, missingness, balance of labels
- How do we treat missingness? Do we try to impute?
- Better to use only variant columns which reduces dimension by half approx, or to convert each codon (3 nucleotides) which reduces dimension by third?
  - grouping nucleotides into condons seems to significantly reduce the number of features in both data sets

Next steps:
- Calculate distance matrix from both datasets (Pseudomonas 122 by 122; Staph 125 by 125). Distances are defined as differences in genome sequence
- With distance matrix, we can cluster the bacteria into groups
- We can see if these clusters match the labels (resistant/susceptible) by plotting them
- Investigate transfer learning, data augmentation, and standard statistics methods like regression

# Analyses

Next steps: We want to fit statistical/machine-learning models to accomplish two tasks:
- feature selection: identify genes associated with antibiotic-resistance
- prediction: for a given new genome, can we predict whether it will be antibiotic-resistant or not

Methods:
- regression (we need to explore penalized regression because we have more features than individuals)
- random forest
- neural networks
- ...

## Main difficulties in this project
- Input data are categories/letters: ACGT, cannot be treated as numbers 1234
- There is correlation among rows and among columns. For example, every three letters in the genome corresponds to one "codon"
- Biologists want to know how many individual bacteria they need to sequence to train the method with high prediction accuracy 

## Previous work

Claudia had fit naive neural networks and random forest in Julia. All scripts in `scripts/previous-work`:
- `notebook.md`: explains pre-processing of the data
- `*.jl`: julia scripts, described in `notebook.md`