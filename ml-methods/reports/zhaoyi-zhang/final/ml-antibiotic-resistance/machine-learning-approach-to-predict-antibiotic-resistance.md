---
title: "Bacterial Antibiotic Resistance Prediction"
author: Zhaoyi Zhang
output:
  html_document: default
  pdf_document: default
---
## Description of objective

Traditional methods such as Direct Association (DA) can be used to predict antibiotic resistance. In DA, genetic variants in a genome sequence are categorized into two groups (resistance-associated and non-resistance-associated) using a series of rules, and then the susceptibility of the given sample to a particular drug can determined from those groups. However, this approach is not ideal for predicting phenotype, since there is a complex relationship between genotype and phenotype. Instead, we propose a machine learning approach to predict antibiotic resistance based on whole-genome sequences and hope to identify resistance-responsible genetic loci by inspecting the models.

## Description of data

One data set available is from Brown Lab at Georgia Institute of Technology, and contains 122 whole-genome sequences of _Pseudomonas aeruginosa_ (referred as _pseudo_) of length 483333 and its resistance to two particular antibiotics, Tobramycin (referred as toby) and Carbenicillin (referred as carb). The other data set is from Read lab at Emory University, and contains 125 whole-genome sequences of _Staphylococcus aureus_ (referred as _staph_) of length 983088 and its toxicity (delta-toxin)<a href="#data"> [1]</a>. The genome data files of two species of bacteria are both in FASTA alignment format and their resistance and toxicity measures (positive or negative labels as well as numerical values) are in CSV files.

The genome sequences contain missing nucleotides, which are denoted as ‘-’. This occurs due to gene insertion or deletion during the cell replication process, or simply because these loci cannot be identified from gene sequencing.

As shown below (<a href="#classes">Fig. 1</a>), both data sets are very unbalanced, where over 80% of the samples are in negative class (susceptible or non-toxic). Because of the unbalanced class distribution, most machine learning models will perform poorly and need modification to avoid predicting the majority class for all samples.

<a id="classes"></a>

![classes](graph/classes.png)
<div align="center"><b>Fig. 1</b></div><br>

## Data pre-processing steps

Each of the paragraphs below describes a single step in data pre-processing. By choosing different options at different steps, we obtain 96 processed data files for each species of bacteria.

1. Imputation: The missing loci can be imputed by simply replacing the missing nucleotide with the nucleotide which appears most frequently at the same locus in other samples. The two figures below (<a href="#impute">Fig. 2-3</a>) show that naive imputation reduces the missing rate by a certain amount. Note that the scales of x-axis are different on the left and right side. However not all missingness can be naively imputed, since some loci are dominated by missing the symbol (i.e. over 50% of the samples have a missingness at those loci).

<a id="impute"></a>

![missing_pseudo](graph/missing_pseudo.png)
<div align="center"><b>Fig. 2</b></div><br>

![missing_staph](graph/missing_staph.png)
<div align="center"><b>Fig. 3</b></div><br>

2. Grouping into codons: Nucleotides in the sequences can be grouped into codons.

3. Feature selection: A genetic locus or codon is discarded if it is invariant across all samples. Alternatively, features are removed using chi-squared test scores to reduce possibly correlated features <a href="#chi2">[2]</a>.

4. Feature extraction: Features are extracted using Principal Component Analysis (PCA) or t-distributed Stochastic Neighbor Embedding (t-SNE), which are commonly used dimension reduction techniques. Besides these two, we also use string kernel to compute the pair-wise genetic distance and view the distance matrix as the extracted features <a href="#string">[3]</a>.

5. Encoding: Nucleotides or codons are encoded into different numerical values (e.g. ‘-’ to 0, ‘A’ to 1, or ‘AGT’ to 5). Optionally, we use one-hot encoder to further convert numerical values into categorical variables to reduce the bias introduced when assigning nucleotides or codons ranks.

By applying the pre-processing steps described above, we obtain 48 sets of numerical data and 48 sets of one-hot encoded data for each species of bacteria.

We name each data file using a four-letter combination that encodes the pre-processing steps (except encoding step) it goes through. Numerical and one-hot encoded data files are stored in two separate folders to keep track of the encoding decision. The first letter can be `i` (imputed) or `o` (original). The second letter can be `n` (nucleotide feature) or `c` (codon feature). The third letter can be `-` (no feature selection), `v` (variance threshold), or `x` (chi-squared). The fourth letter can be `-` (no feature extraction), `p` (PCA), `t` (t-SNE), or `s` (string kernel). Letters are separated by `_`. The data file has `.npy` file extension, which can be read into a Python Numpy array. An example data file name can be `i_c_-_s.npy` for imputed codon features without feature selection and string kernel feature extraction.

## Results and analysis

We build an Unweighted Pair Group Method with Arithmetic Mean (UPGMA) and a maximum likelihood phylogenetic tree for each species of bacteria, but the trees do not give insights on how antibiotic-resistant samples are clustered, since they do tend to scatter across the trees (figures not shown but could be added upon request).

We also use PCA and t-SNE to visualize all pre-processed data, but all of them don’t show a clear boundary between two classes (positive samples scatter among negative samples). Therefore, it might be hard to train a model that can classify them well (figures not shown but could be added upon request).

We train three types of classification models (logistic regression, random forest, and support vector machine) with fixed hyper-parameters on each data set, so, in total, there are 288 models for each species of bacteria. Those machine learning models are able to classify the negative class very well. However, this can also be achieved by a dummy estimator which just output negative prediction for all samples, because the data sets mainly consist of negative class. If we give positive class high weight in the loss function, models can learn to correctly classify both classes on the training set, but still fail to generalize and correctly classify the positive labels on the test set.

The best result we obtain for _Pseudomonas_ is from a support vector machine model with an balanced accuracy of 0.884 and F1 score of 0.778 (see <a href="#t1">table 1</a>), while the best result we obtain for _Staphylococcus_ is from a logistic regression model with an balanced accuracy of 0.828 and F1 score of 0.522 (see <a href="#t2">table 2</a>). Based on the results, we can see that support vector machine performs well on _Pseudomonas_ data, while logistic regression does well on both. Therefore, we could fine tune the hyper-parameters of those two models to potentially get better performance.

<div align="left" id="t1"><b>Table 1</b><br>5 best classfiers for <i>Pseudomonas</i> according to F1 score</div>

| accuracy <a href="#accuracy">[4]</a> | balanced_accuracy <a href="#baccuracy">[5]</a> | precision <a href="#precision">[6]</a> | recall <a href="#recall">[7]</a> | f1 <a href="#f1">[8]</a> | data           | encode | model    |
| :----------------------------------- | :--------------------------------------------- | :------------------------------------- | :------------------------------- | :----------------------- | :------------- | :----- | -------- |
| 0.889                                | 0.884                                          | 0.700                                  | 0.875                            | 0.778                    | o\_n\_v\_-.npy | False  | svm      |
| 0.889                                | 0.839                                          | 0.750                                  | 0.750                            | 0.750                    | o\_n\_x\_-.npy | False  | logistic |
| 0.861                                | 0.866                                          | 0.636                                  | 0.875                            | 0.737                    | o\_c\_-\_-.npy | True   | svm      |
| 0.861                                | 0.821                                          | 0.667                                  | 0.750                            | 0.706                    | o\_n\_x\_-.npy | True   | logistic |
| 0.861                                | 0.821                                          | 0.667                                  | 0.750                            | 0.706                    | o\_n\_x\_-.npy | True   | svm      |
<br>

<div align="left" id="t2"><b>Table 2</b><br>5 best classfiers for <i>Staphylococcus</i> according to F1 score</div>

| accuracy <a href="#accuracy">[4]</a> | balanced_accuracy <a href="#baccuracy">[5]</a> | precision <a href="#precision">[6]</a> | recall <a href="#recall">[7]</a> | f1 <a href="#f1">[8]</a> | data           | encode | model    |
| :----------------------------------- | :--------------------------------------------- | :------------------------------------- | :------------------------------- | :----------------------- | :------------- | :----- | :------- |
| 0.711                                | 0.828                                          | 0.353                                  | 1.000                            | 0.522                    | i\_c\_-\_t.npy | False  | logistic |
| 0.895                                | 0.667                                          | 1.000                                  | 0.333                            | 0.500                    | o\_n\_-\_-.npy | True   | rf       |
| 0.895                                | 0.667                                          | 1.000                                  | 0.333                            | 0.500                    | o\_n\_-\_-.npy | False  | rf       |
| 0.895                                | 0.667                                          | 1.000                                  | 0.333                            | 0.500                    | i\_c\_-\_-.npy | False  | svm      |
| 0.895                                | 0.667                                          | 1.000                                  | 0.333                            | 0.500                    | o\_n\_-\_-.npy | False  | svm      |

<br>

To avoid the problem of unbalanced classes in the data sets, we also train regression models with toxicity measures. Similarly, we train three types of regression models (linear regression, random forest, and support vector machine) on each data set, and there are 288 models for each bacterium.

In general, regression models perform very poorly on _Pseudomonas_ data for currently unknown reasons (see <a href="#t3">table 3</a>), but this is worth investigating. We could add regularization or tune hyper-parameters to try to improve the result. On the other side, regression models perform slightly better on _Staphylococcus_ data sets, where the best model only has a mean squared error of 0.145 (see <a href="#t4">table 4</a>). However, the R2 score is not high for these models.

<div align="left"  id="t3"><b>Table 3</b><br>5 best regressors for <i>Pseudomonas</i> according to mean squared error</div>

| r2 <a href="#r2">[9]</a> | max_error <a href="#me">[10]</a> | mean_absolute_error <a href="#mae">[11]</a> | mean_squared_error <a href="#mse">[12]</a> | data           | encode | model  |
| :----------------------- | :------------------------------- | :------------------------------------------ | :----------------------------------------- | :------------- | :----- | ------ |
| 0.079                    | 32.932                           | 7.994                                       | 103.135                                    | o\_c\_v\_p.npy | False  | rf     |
| 0.027                    | 29.160                           | 7.833                                       | 108.943                                    | i\_n\_x\_t.npy | False  | rf     |
| 0.021                    | 24.740                           | 7.780                                       | 109.563                                    | o\_c\_x\_s.npy | False  | rf     |
| 0.020                    | 33.059                           | 7.873                                       | 109.688                                    | o\_c\_x\_s.npy | True   | linear |
| 0.020                    | 32.705                           | 7.846                                       | 109.722                                    | i\_n\_x\_s.npy | True   | linear |

<br>

<div align="left" id="t4"><b>Table 4</b><br>5 best regressors for <i>Staphylococcus</i> according to mean squared error</div>

| r2 <a href="#r2">[9]</a> | max_error <a href="#me">[10]</a> | mean_absolute_error <a href="#mae">[11]</a> | mean_squared_error <a href="#mse">[12]</a> | data        | encode | model |
| :----------------------- | :------------------------------- | :------------------------------------------ | :----------------------------------------- | :---------- | :----- | ----- |
| 0.213                    | 0.896                            | 0.251                                       | 0.142                                      | i_c_-_t.npy | False  | rf    |
| 0.182                    | 0.894                            | 0.292                                       | 0.148                                      | i_c_-_-.npy | False  | svm   |
| 0.101                    | 0.903                            | 0.263                                       | 0.162                                      | i_n_x_-.npy | False  | svm   |
| 0.093                    | 0.912                            | 0.260                                       | 0.164                                      | o_c_v_p.npy | False  | rf    |
| 0.090                    | 0.992                            | 0.269                                       | 0.164                                      | i_c_v_p.npy | False  | rf    |

<br>

To see how different metrics vary with each other, we plot the scores of all classifiers or regressors in one figure for each species of bacteria. There are 288 models (represented by x-axis) in each figure. Classifiers are ordered by increasing recall and break ties using precision. Regressors are ordered by increasing mean squared error.

![metric_pseudo_clf](graph/metric_pseudo_clf.png)
<div align="center"><b>Fig. 4</b></div><br>

![metric_staph_clf](graph/metric_staph_clf.png)
<div align="center"><b>Fig. 5</b></div><br>

![metric_pseudo_reg](graph/metric_pseudo_reg.png)
<div align="center"><b>Fig. 6</b></div><br>

![metric_staph_reg](graph/metric_staph_reg.png)
<div align="center"><b>Fig. 7</b></div><br>

To sum up, classifiers can achieve an accuracy of 77% largely due to the unbalanced data sets, but some of them can achieve an accuracy above 80%. In the future, we can use different class weights to take the nature of the data sets into account to potentially gain higher performance. Regressors are not quite suitable for the data sets, since the R2 scores are very close to 0, or even large negative values <a href="neg-r2">[13]</a>.

## Specific biological insights learned

We train decision trees to predict how different combinations of decisions leads to different model performance. All models are listed and then fit into a decision tree to identify those with lowest balanced accuracy or mean squared error (see <a href="#tree">Fig. 8-11</a>).

Each node in this decision tree corresponds to a decision (either a pre-processing step or a model choice) in the data analysis. For example, node #1 in the pseudo classifier decision tree corresponds to `extraction <=0.5`. The variable `extraction` represents whether features were extracted  (`extraction=1`), or not extracted (`extraction=0`). The statement `extraction<=0.5` means that models trained on extracted features go to the right of this node, and models trained on original features go to the left of this node. In this manner, every node in the decision tree identifies a given decision on the data analysis. Each leaf node corresponds to a group of models trained with the same combination of decisions from the top.

The color of the node indicates the predicted performance (balanced accuracy for classifiers, mean squared error for regressors) for the models that follow the same combinations of decisions. Therefore, for the classifiers, the darker the color, the better the predicted performance; for the regressors, the darker the color, the worse the performance.

We identify the best combination for pseudo classifiers is using unimputed data, variance threshold or no feature selection, no feature extraction, and logistic regression model, while the best combination for staph classifiers is using imputed data, variance threshold feature selection, no feature extraction, and no specific model. Regressors are not discussed due to their poor performance. 

However, we are yet to determine the genetic loci that are responsible for antibiotic resistance, since the models are not accurate enough.

<a id="tree"></a>

![decision_tree_pseudo_clf](graph/decision_tree_pseudo_clf.png)
<div align="center"><b>Fig. 8</b></div><br>

![decision_tree_pseudo_reg](graph/decision_tree_pseudo_reg.png)
<div align="center"><b>Fig. 9</b></div><br>

![decision_tree_staph_clf](graph/decision_tree_staph_clf.png)
<div align="center"><b>Fig. 10</b></div><br>

![decision_tree_staph_reg](graph/decision_tree_staph_reg.png)
<div align="center"><b>Fig. 11</div><br>

## Reference
<ol>
<li><a id="data"></a>M. Su, J. Lyles, R. A. Petit III, J. Peterson, M. Hargita, H. Tang, C. Solis-Lemus, C. Quave, and T. D. Read. Genomic analysis of variability in delta-toxin levels between Staphylococcus aureus strains. In review, 2018.</li>
<li><a id="chi2"></a>Feature selection: chi2. Link: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html">https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html</a></li>
<li><a id="string"></a>Feature extraction: string kernel. Link: <a href="https://string-kernel.readthedocs.io/en/latest/">https://string-kernel.readthedocs.io/en/latest/</a></a></li>
<li><a id="accuracy"></a>Mertic: accuracy. Link: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html">https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html</a></li>
<li><a id="baccuracy"></a>Mertic: balanced accuracy. Link: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html">https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html</a></li>
<li><a id="precision"></a>Mertic: precision. Link: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html">https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html</a></li>
<li><a id="recall"></a>Mertic: recall. Link: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html">https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html</a></li>
<li><a id="f1"></a>Mertic: f1. Link: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html">https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html</a></li>
<li><a id="r2"></a>Mertic: r2. Link: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html">https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html</a></li>
<li><a id="me"></a>Mertic: max error. Link: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.max_error.html">https://scikit-learn.org/stable/modules/generated/sklearn.metrics.max_error.html</a></li>
<li><a id="mae"></a>Mertic: mean absolute error. Link: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html">https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html</a></li>
<li><a id="mse"></a>Mertic: mean squared error. Link: <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html">https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html</a></li>
<li><a id="neg-r2"></a>Negative r2 scores. Link: <a href="https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score">https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score</a></li>
</ol>