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

### Main difficulties in this project
- Input data are categories/letters: ACGT, cannot be treated as numbers 1234
- There is correlation among rows and among columns. For example, every three letters in the genome corresponds to one "codon"
- Biologists want to know how many individual bacteria they need to sequence to train the method with high prediction accuracy 

### Previous work

Claudia had fit naive neural networks and random forest in Julia. All scripts in `scripts/previous-work`:
- `notebook.md`: explains pre-processing of the data
- `*.jl`: julia scripts, described in `notebook.md`