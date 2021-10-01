## Julia script to understand how BEDFiles stores data
## inspired by runtests.jl in BEDFiles.jl
## Claudia July 2019

using BEDFiles, SparseArrays, Test

const mouse = BEDFile(BEDFiles.datadir("mouse.bed"))
typeof(mouse)
typeof(mouse.data) ##Array{UInt8,2}
mouse.data
mouse.m
