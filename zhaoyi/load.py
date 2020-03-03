from collections import Counter
import pandas as pd
import numpy as np
from Bio import SeqIO

## Load Pseudomonas data
# load responses-pseudo.csv - the result of antibiotic resistence of pseudomonas
# 122 * 2 (2 antibiotics)
resp = pd.read_csv('../data/pseudo/responses-pseudo.csv', names=['id', 'lab-id', 'carb', 'toby'], skiprows=1)
resp.drop('lab-id', axis=1, inplace=True)

# load concatenated.fasta - the gene sequence of pseudomonas
# 122 * (483333 -> 261868)
src = SeqIO.parse('../data/pseudo/concatenated.fasta', 'fasta')
data = [(records_pseudo.id, records_pseudo.seq._data) for records_pseudo in src]
seq = pd.DataFrame(data=data, columns=['id', 'sequence'])

# merge DataFrames of two files into one DataFrame
records_pseudo = pd.merge(seq, resp, on='id')

# calculate missing number and percentage of nucleotides of each sequence
seq_len_pseudo = len(records_pseudo['sequence'][0])
records_pseudo['missing'] = records_pseudo['sequence'].apply(lambda seq: Counter(seq)['-'])
records_pseudo['missing_percentage'] = records_pseudo['missing'] / seq_len_pseudo * 100

## Load Staphylococcus data
# load responses-staph.csv - the result of antibiotic resistence of staphylococcus
# 125 * 1
resp = pd.read_csv('../data/staph/responses-staph.csv')

# load core_gene_alignment-narsa.aln - the gene sequence of staphylococcus
# 125 * 983088
src = SeqIO.parse('../data/staph/core_gene_alignment-narsa.aln', 'fasta')
data = [(records_staph.id, records_staph.seq._data.upper()) for records_staph in src]
seq = pd.DataFrame(data=data, columns=['ids', 'sequence'])

# merge DataFrames of two files into one DataFrame
records_staph = pd.merge(seq, resp, on='ids')

# calculate missing number and percentage of nucleotides of each sequence
seq_len_staph = len(records_staph['sequence'][0])
records_staph['missing'] = records_staph['sequence'].apply(lambda seq: Counter(seq)['-'])
records_staph['missing_percentage'] = records_staph['missing'] / seq_len_staph * 100