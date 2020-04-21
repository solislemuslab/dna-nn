from collections import Counter
import pandas as pd
import numpy as np
from Bio import SeqIO

def load_pseudo(numerical=False):
    ## Load Pseudomonas data
    # load responses-pseudo.csv - the result of antibiotic resistence of pseudomonas
    # 122 * 2 (2 antibiotics)
    resp = pd.read_csv('../data/pseudo/responses-pseudo.csv', names=['id', 'lab-id', 'carb', 'toby'], skiprows=1)

    # load concatenated.fasta - the gene sequence of pseudomonas
    # 122 * (483333 -> 261868)
    src = SeqIO.parse('../data/pseudo/concatenated.fasta', 'fasta')
    data = [(record.id, record.seq._data.upper()) for record in src]
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

    # combine everything
    records = pd.merge(seq, seq_i, on='id')
    records = pd.merge(records, resp, on='id')
    if numerical:
        numerical_response = pd.read_csv('../data/pseudo/Perron_phenotype-GSU-training.csv')
        records = records.merge(numerical_response[['strain', 'carb.lag.delta', 'toby.lag.delta']],
                                left_on='lab-id', right_on='strain', how='left')
        records.rename(columns={'carb.lag.delta': 'carb_num', 'toby.lag.delta': 'toby_num'}, inplace=True)
        records.drop(columns=['strain', 'lab-id'], inplace=True)
    
    return records

def load_staph(numerical=False):
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

    # combine everything
    records = pd.merge(seq, seq_i, on='id')
    records = pd.merge(records, resp, on='id')
    if numerical:
        numerical_response = pd.read_csv('../data/staph/nrs_metadata3.txt', delimiter='\t')
        records = records.merge(numerical_response[['sample_tag', 'Total.Area']],
                                left_on='id', right_on='sample_tag', how='left')
        records.drop(columns='sample_tag', inplace=True)

    return records

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