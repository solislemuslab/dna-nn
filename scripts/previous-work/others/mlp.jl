## Julia script from
## https://github.com/FluxML/model-zoo/blob/master/vision/mnist/mlp.jl
## to understand how are images read for Flux.jl
## Claudia July 2019

using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated
# using CuArrays

# Classify MNIST digits with a simple multi-layer-perceptron

imgs = MNIST.images()
typeof(imgs)
## Array{Array{ColorTypes.Gray{FixedPointNumbers.Normed{UInt8,8}},2},1}
size(imgs)
## (60000,)
imgs[1]
## 28×28 Array{Gray{N0f8},2} with eltype ColorTypes.Gray{FixedPointNumbers.Normed{UInt8,8}}:
typeof(imgs[1])
## Array{ColorTypes.Gray{FixedPointNumbers.Normed{UInt8,8}},2}

# Stack images into one large batch
X = hcat(float.(reshape.(imgs, :))...)
## 784×60000 Array{Float64,2}:
## Note: 28*28=784
size(X)
## (784, 60000)

labels = MNIST.labels()
## 60000-element Array{Int64,1}:
# One-hot-encode the labels
Y = onehotbatch(labels, 0:9)
## 10×60000 Flux.OneHotMatrix{Array{Flux.OneHotVector,1}}:
## there are 10 labels: 0:9


## Now let's try with my own image
using FileIO, Images, CSV, DataFrames, JLD, ImageTransformations
img = FileIO.load("../../data/pseudomonas/PIL_3dayLBCR-training/PIL-1_3dayLBCR-1.jpg")
img2 = FileIO.load("../../data/pseudomonas/PIL_3dayLBCR-training/PIL-1_3dayLBCR-2.jpg")
## 9840×10560 Array{RGB4{N0f8},2} with eltype RGB4{Normed{UInt8,8}}:
imgs = [Gray.(imresize(img,ratio=0.25)), Gray.(imresize(img2,ratio=0.25))]
## 2-element Array{Array{Gray{Normed{UInt8,8}},2},1}:

X = hcat(float.(reshape.(imgs, :))...)

## so, I can read all images in an array, and then follow the same steps

## original script ----------------------------------------------------------
using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated
# using CuArrays

# Classify MNIST digits with a simple multi-layer-perceptron

imgs = MNIST.images()
# Stack images into one large batch
X = hcat(float.(reshape.(imgs, :))...) |> gpu

labels = MNIST.labels()
# One-hot-encode the labels
Y = onehotbatch(labels, 0:9) |> gpu

m = Chain(
  Dense(28^2, 32, relu),
  Dense(32, 10),
  softmax) |> gpu

loss(x, y) = crossentropy(m(x), y)

accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))

dataset = repeated((X, Y), 200)
evalcb = () -> @show(loss(X, Y))
opt = ADAM()

Flux.train!(loss, params(m), dataset, opt, cb = throttle(evalcb, 10))

accuracy(X, Y)

# Test set accuracy
tX = hcat(float.(reshape.(MNIST.images(:test), :))...) |> gpu
tY = onehotbatch(MNIST.labels(:test), 0:9) |> gpu

accuracy(tX, tY)
