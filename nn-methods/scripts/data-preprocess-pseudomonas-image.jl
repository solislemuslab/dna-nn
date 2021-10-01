## Julia script for further data pre processing for
## pseudomonas image data
## We do this so that we do not save the huge features.jld file
## Claudia August 2019

using FileIO, Images, ImageTransformations
imagestest = load(string(datafolder,"images-testing.jld"), "images")
## 65-element Array{Array{Gray{Normed{UInt8,8}},2},1}
imagestrain = load(string(datafolder,"images-training.jld"), "images")
## 262-element Array{Array{Gray{Normed{UInt8,8}},2},1}

# Stack images into one large batch
Xtest = hcat(float.(reshape.(imagestest, :))...)
## 1623600×65 Array{Float64,2
Xtrain = hcat(float.(reshape.(imagestrain, :))...)
## 1623600×262 Array{Float64,2}

features = hcat(Xtrain,Xtest)
## 1623600×327 Array{Float64,2}

