from collections import Counter
import pandas as pd
import numpy as np
from Bio import SeqIO

def load_pseudo():
    ## Load Pseudomonas data
    # load responses-pseudo.csv - the result of antibiotic resistence of pseudomonas
    # 122 * 2 (2 antibiotics)
    resp = pd.read_csv('../data/pseudo/responses-pseudo.csv', names=['id', 'lab-id', 'carb', 'toby'], skiprows=1)

    # load concatenated.fasta - the gene sequence of pseudomonas
    # 122 * (483333 -> 261868)
    src = SeqIO.parse('../data/pseudo/concatenated.fasta', 'fasta')
    data = [(record.id, record.seq._data) for record in src]
    seq = pd.DataFrame(data=data, columns=['id', 'sequence'])
    seq_len_pseudo = len(seq['sequence'][0])
    seq['missing'] = seq['sequence'].apply(lambda seq: Counter(seq)['-'])
    seq['missing_%'] = seq['missing'] / seq_len_pseudo * 100

    # load concatenated_naive_impute.fasta
    src = SeqIO.parse('../data/pseudo/concatenated_naive_impute.fasta', 'fasta')
    data = [(record.id, record.seq._data) for record in src]
    seq_i = pd.DataFrame(data=data, columns=['id', 'sequence_i'])
    seq_i['missing_i'] = seq_i['sequence_i'].apply(lambda seq: Counter(seq)['-'])
    seq_i['missing_%_i'] = seq_i['missing_i'] / seq_len_pseudo * 100

    # combine three files
    records_pseudo = pd.merge(seq, seq_i, on='id')
    records_pseudo = pd.merge(records_pseudo, resp, on='id')
    return records_pseudo

def load_staph():
    ## Load Staphylococcus data
    # load responses-staph.csv - the result of antibiotic resistence of staphylococcus
    # 125 * 1
    resp = pd.read_csv('../data/staph/responses-staph.csv', names=['resp', 'id'], skiprows=1)

    # load core_gene_alignment-narsa.aln - the gene sequence of staphylococcus
    # 125 * 983088
    src = SeqIO.parse('../data/staph/core_gene_alignment-narsa.aln', 'fasta')
    data = [(record.id, record.seq._data.upper()) for record in src]
    seq = pd.DataFrame(data=data, columns=['id', 'sequence'])
    seq_len_staph = len(seq['sequence'][0])
    seq['missing'] = seq['sequence'].apply(lambda seq: Counter(seq)['-'])
    seq['missing_%'] = seq['missing'] / seq_len_staph * 100

    # load core_gene_alignment-narsa_naive_impute.fasta
    src = SeqIO.parse('../data/staph/core_gene_alignment-narsa_naive_impute.fasta', 'fasta')
    data = [(record.id, record.seq._data.upper()) for record in src]
    seq_i = pd.DataFrame(data=data, columns=['id', 'sequence_i'])
    seq_i['missing_i'] = seq_i['sequence_i'].apply(lambda seq: Counter(seq)['-'])
    seq_i['missing_%_i'] = seq_i['missing_i'] / seq_len_staph * 100

    # combine three files
    records_staph = pd.merge(seq, seq_i, on='id')
    records_staph = pd.merge(records_staph, resp, on='id')
    return records_staph

def load_nucleotides(path):
    src = SeqIO.parse(path, 'fasta')
    n = pd.DataFrame(src)

    src = SeqIO.parse(path, 'fasta')
    n.index = [record.id for record in src]
    return n

def load_condons(path):
    src = SeqIO.parse(path, 'fasta')
    c = pd.DataFrame([[seq.seq._data[i:i+3] for i in range(0, len(seq), 3)] for seq in src])

    src = SeqIO.parse(path, 'fasta')
    c.index = [record.id for record in src]
    return c