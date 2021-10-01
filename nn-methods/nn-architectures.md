# NN architectures in literature

It is difficulty to decide NN architecture, so we will use architectures that showed promise in literature already.

## Chen 2019: TB antibiotic resistance (sequences)

- The final model used these 222 total predictors in training and subsequent analyses.
- 3601 isolates: 1379 in training, 2222 in testing

We implemented a WDNN with three hidden layers each with 256 rectified linear units (ReLU) [28], dropout [29], batch normalization [30], and L2 regularization (Fig. 7). Dropout and L2 regularization are used to prevent overfitting of the models to the training data. L2 regularization was applied on the wide model (which is equivalent to the well-known Ridge Regression model) [31], the hidden layers of the deep model, and the output sigmoid layer. The network was trained via stochastic gradient descent using the Adam optimizer for 100 epochs with randomized initial starting weights determined by Xavier uniform initialization.
The MD-WDNN was trained simultaneously on resistance status for all 11 drugs, including ciprofloxacin. Each of the 11 nodes in the final layer represented one drug and its output value was the probability that the MTB isolate was resistant to the corresponding drug. We constructed a single-drug WDNN (SD-WDNN) with the same architecture as the multidrug model except for the structure of the output layer, which predicts for one drug.

The MD-WDNN utilized a loss function that is a variant of traditional binary cross entropy. Due to imbalance between the susceptible and resistant classes within each drug, we adjusted our loss function to upweight the sparser class according to the susceptible-resistant ratio within each drug. Thus, the final loss function was a class- weight binary cross entropy that masked outputs where the resistance status was missing.

### In julia
- Layer 0: Dense(222,256)
- Layer 1: Dense(256,256, relu) + BatchNorm(256, relu) + Dropout(0.5)
- Layer 2: Dense(256, 256, relu) + BatchNorm(256, relu) + Dropout(0.5)
- Layer 3: Dense(256,478, relu) + BatchNorm(478, relu) + Dropout(0.5)
- Layer 4: Dense(478, 11, \sigma)

- loss(x, y) = crossentropy(m(x), y) + sum(norm, params(m)) with crossentropy modified to upweight the sparser class due to the imbalance. We should also include L2 norm in every layer, but not sure how to do this

### From Flux documentation: 

[Regularisation](https://fluxml.ai/Flux.jl/stable/models/regularisation/)

We can regularise this by taking the (L2) norm of the parameters, m.W and m.b.
```
penalty() = norm(m.W) + norm(m.b)
loss(x, y) = crossentropy(softmax(m(x)), y) + penalty()
```
where `m` is the Model.

Here's a larger example with a [multi-layer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron).
```
m = Chain(
  Dense(28^2, 128, relu),
  Dense(128, 32, relu),
  Dense(32, 10), softmax)

loss(x, y) = crossentropy(m(x), y) + sum(norm, params(m))
```

One can also easily add per-layer regularisation via the activations function:
```
julia> c = Chain(Dense(10,5,σ),Dense(5,2),softmax)
Chain(Dense(10, 5, NNlib.σ), Dense(5, 2), NNlib.softmax)

julia> activations(c, rand(10))
3-element Array{Any,1}:
 param([0.71068, 0.831145, 0.751219, 0.227116, 0.553074])
 param([0.0330606, -0.456104])
 param([0.61991, 0.38009])

julia> sum(norm, ans)
2.639678767773633 (tracked)
```

- Note that default activation is identity: `Dense(in::Integer, out::Integer, σ = identity)`
- Note that the `softmax` defined as the last parameter in `Chain()` represents the last normalization to the output to convert to probabilities of each class. See [here](https://en.m.wikipedia.org/wiki/Softmax_function)
- Note that we can learn about Flux batch norm layer and dropout layer [here](https://github.com/FluxML/Flux.jl/blob/master/src/layers/normalise.jl)


## Pu2018: images

The architecture of a CNN implemented in DeepDrug3D is shown in Fig 2. The input voxel (Fig 2A) is followed by two consecutive convolutional layers with leaky ReLu activation functions (Fig 2B). The output from the second convolutional layer, 64 feature maps of size 26 x 26 x 26, are passed through a dropout layer, a maxpooling layer, and another dropout layer before entering the fully connected layer (Fig 2C). Since the network output are different ligand types, softmax is the final activation layer. Note that this architecture is similar to the VGG-network comprising multiple blocks of stacked convolutional layers followed by dropout and maxpooling layers [58]. However, since our voxel size of 32 x 32 x 32 x 14 is significantly larger than the typical image data used in computer vision, such a deep network architecture would be computationally unfeasible. Further, because of a relatively small number of samples, i.e. non-redundant binding sites in the PDB, a simpler architecture helps avoid over-fitting, yet it still captures distinctive features to accurately classify binding pockets.

Deep-Drug3D employs the Adam optimization algorithm, which was shown to outperform other methods in terms of the robustness and stability [59]. The learning rate, the learning rate decay, and b1 and b2 hyperparameters of the Adam are set to 0.00001, 0, 0.9, and 0.999, respectively. We found empirically that different batch sizes of 16, 32, 64, and 128 yield a comparable performance, and 50 epochs are sufficient to reach the convergence.
