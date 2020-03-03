import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Bio import Phylo

def diff(seq1, seq2):
    '''Count the number of different loci between two sequences'''
    diff_count = 0
    for n1, n2 in zip(seq1, seq2):
        if n1 != n2:
            diff_count += 1
    return diff_count

def dist_mat(records):
    '''Build the distance matrix'''
    dist_mat = np.zeros((records.shape[0], records.shape[0]), dtype='i4')
    for i in range(records.shape[0]):
        for j in range(i, records.shape[0]):
            d = diff(records['sequence'][i], records['sequence'][j])
            dist_mat[i, j] = d
            dist_mat[j, i] = d
    dist_mat = pd.DataFrame(dist_mat, index=records['id'], columns=records['id'])
    dist_mat[dist_mat==0] = np.inf
    return dist_mat

def upgma(df, copy=True):
    '''Build a phylogenetic tree (in newick format) using UPGMA based on a distance matrix'''
    if copy:
        df = df.copy()
    while df.size > 1:
        mask = (df == df.min().min())     # where is the minimum
        row = mask.any(axis=1)            # which row
        pos = df[row].idxmin(axis=1)      # which column
        for i in range(0, pos.size//2):
            a = pos.index[i]
            b = pos[i]
            # combine the row and column of one with the other
            result = (df.loc[a] + df.loc[b]) / 2
            df.loc[a] = result
            df.loc[:, a] = result
            # drop row and coulmn of the other
            df = df.drop(b, axis=0).drop(b, axis=1)
            # rename index and column name of the combined sample
            df.rename(mapper=lambda s: s if s != a else '({},{})'.format(a, b), axis=0, inplace=True)
            df.rename(mapper=lambda s: s if s != a else '({},{})'.format(a, b), axis=1, inplace=True)
    return df.index[0]+';'

def draw_tree(path, title, format='newick'):
    tree = Phylo.read(path, format=format)
    plt.figure(figsize=(23, 23))
    plt.title(title)
    Phylo.draw(tree, axes=plt.gca())