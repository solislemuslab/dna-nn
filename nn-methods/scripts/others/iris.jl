## From https://github.com/FluxML/model-zoo/blob/master/other/iris/iris.jl
## We will study to understand Flux.jl
## Claudia July 2019

using Flux
using Flux: crossentropy, normalise, onecold, onehotbatch
using Statistics: mean
using JLD

labels = Flux.Data.Iris.labels() ## vector dimension 150
##features = Flux.Data.Iris.features() ## 4×150 Array{Float64,2}
features = load("features-iris-missing.jld", "features")

## test with missingness
## vc1 = vcat(missing,fill(1.0,size(features,1)-1))
## features = hcat(vc1, features)
## features = features[:,1:(end-1)]


normed_features = features ## no need to normalise just to check
# Subract mean, divide by std dev for normed mean of 0 and std dev of 1.
##normed_features = normalise(features, dims=2)


klasses = sort(unique(labels))
onehot_labels = onehotbatch(labels, klasses) ## 3×150 Flux.OneHotMatrix{Array{Flux.OneHotVector,1}} (true,false,false)=Iris-setosa


# Split into training and test sets, 2/3 for training, 1/3 for test.
train_indices = [1:3:150 ; 2:3:150]

X_train = normed_features[:, train_indices]
y_train = onehot_labels[:, train_indices]

X_test = normed_features[:, 3:3:150]
y_test = onehot_labels[:, 3:3:150]


# Declare model taking 4 features as inputs and outputting 3 probabiltiies,
# one for each species of iris.
model = Chain(
    Dense(4, 3),
    softmax
)

loss(x, y) = crossentropy(model(x), y)

# Gradient descent optimiser with learning rate 0.5.
optimiser = Descent(0.5)


# Create iterator to train model over 110 epochs.
data_iterator = Iterators.repeated((X_train, y_train), 110)

println("Starting training.")
Flux.train!(loss, params(model), data_iterator, optimiser)

# Evaluate trained model against test set.
accuracy(x, y) = mean(onecold(model(x)) .== onecold(y))

accuracy_score = accuracy(X_test, y_test)

println("\nAccuracy: $accuracy_score")

# Sanity check.
@assert accuracy_score > 0.8


## function confusion_matrix(X, y)
##     ŷ = onehotbatch(onecold(model(X)), 1:3)
##     y * ŷ'
## end
## println("\nConfusion Matrix:\n")
## display(confusion_matrix(X_test, y_test))
