## Contents

### Jupyter notebook file

CNN-RNN.ipynb: the codes to read and parse the dataset, and define, compile, and train the CNN and RNN models

### CHTC-scripts folder

Files to train the model on CHTC servers

Possible OOM on some servers but could run on others.



## Models

Models adapted from https://github.com/solislemuslab/dna-nn-theory/blob/master/cnn/dna_nn/model.py

CNN

- cnn_nguyen_conv1d_2_conv2d

RNN

- deepram_conv1d_recurrent_onehot

- deepram_recurrent_onehot



## Problem

### Imbalanced dataset

Carb: ~80% false, Toby: ~95% false. Model will biased to all the inputs as true.

### Try to fix

1. Current - down-sample for false class and augment the samples for true class with reverse complement to have a balanced training set



## TODO

### Data Augmentation

- Currently using reverse complement to augment one more sample to each sample
- Map the sequence to genome and shift to left or right. 
  - Whole genome is not availabe
- Extend half of the sequence and crop to use left, middle, and right
  - Our sequences are too long, which would include too many manually generated data that are not similar to the original samples

Searching for more methods

### Change in training process

- ? Set penalization in trainig process
- ? Change the metric (not accuracy, maybe loss, ROC, or precision?) in optimizer when compile the model