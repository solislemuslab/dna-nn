## Contents

### Jupyter notebook file

CNN-RNN.ipynb: the codes to read and parse the dataset, and define, compile, and train the CNN and RNN models

new-data-process.ipynb: the codes to read and compute the response csv file based on xlsx file received in late November, 2021

binary-classification-description.ipynb: description of process and results of binary classification

regression-description.ipynb: description of process and results of regression



### CHTC-scripts folder

Files to train the model on CHTC servers. Check out the readme file in the folder for description of contents

Possible OOM on some CHTC servers but could run on others.



## Model

Models adapted from https://github.com/solislemuslab/dna-nn-theory/blob/master/cnn/dna_nn/model.py

CNN - cnn_nguyen_conv1d_2_conv2d



## Problem

### Small dataset

- training accuracy did not increase - the model is not being trained
- overfitting after data augmentation

#### Try to fix

1. down-sample for false class and augment the samples for true class with reverse complement to have a balanced training set - not working
1. Reverse complement and bootstrap (not working) for true class
1. Use simulated data generated with IQ-tree



## Methods

### Data Augmentation

- Using reverse complement to augment one more sample to each sample
- [IQ-tree](http://www.iqtree.org/) allows for estimation of tree from sequences in a fast and accurate manner
  - From Claudia: use the sequences of the same class to reconstruct a phylogenetic tree, then simulate new sequences under this phylogenetic tree and use the new simulated sequences as the augmented data
  - simulate sequences on a given tree: http://www.iqtree.org/doc/AliSim
  - beginner’s tutorial to reconstruct trees: http://www.iqtree.org/doc/Tutorial
- Not working or not suitable
  - Bootstrap
    - [Paper](https://www.pnas.org/content/93/23/13429): “A bootstrap data matrix **x*** is formed by randomly selecting 221 columns from the original matrix **x** *with replacement*.”

  - Map the sequence to genome and shift to left or right. 
    - Whole genome is not availabe

  - Extend half of the sequence and crop to use left, middle, and right to keep balance (not used with new data)
    - Our sequences are too long, which would include too many manually generated data that are not similar to the original samples

