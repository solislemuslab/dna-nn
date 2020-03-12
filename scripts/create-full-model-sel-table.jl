## julia script to create the table with options
## for full model selection in staph and pseudomonas
## Claudia March 2020


## Staph (this could be done more efficiently with a Dict, good enough for now)
impute = ["yes","no"]
inputmatrix = ["codon", "original", "snp"]
featsel = ["no", "varThreshold", "recursive", "chi2"]
featext = ["no", "pca", "tsne"]
response = ["binary", "numerical"]
data = ["original", "appended"]
models = ["regression", "RF", "SVM"]

using DataFrames
df = DataFrame(impute="", inputmatrix="", featsel="", featext="",
               response="", data="", models="")

for i in impute
    for ii in inputmatrix
        for fs in featsel
            for fe in featext
                for r in response
                    for d in data
                        for m in models
                            push!(df, [i,ii,fs,fe,r,d,m])
                        end
                    end
                end
            end
        end
    end
end

df = df[2:end,:] ##remove first row created just to assign type to entries
##864×7 DataFrame

using CSV
CSV.write("../results/staph-FMS.csv", df)




## Pseudomonas (this could be done more efficiently with a Dict, good enough for now)
impute = ["yes","no"]
inputmatrix = ["codon", "original", "snp"]
featsel = ["no", "varThreshold", "recursive", "chi2"]
featext = ["no", "pca", "tsne"]
response = ["binary", "numerical"]
models = ["regression", "RF", "SVM"]

using DataFrames
df = DataFrame(impute="", inputmatrix="", featsel="", featext="",
               response="", models="")

for i in impute
    for ii in inputmatrix
        for fs in featsel
            for fe in featext
                for r in response
                    for m in models
                        push!(df, [i,ii,fs,fe,r,m])
                    end
                end
            end
        end
    end
end

df = df[2:end,:] ##remove first row created just to assign type to entries
##432×7 DataFrame

using CSV
CSV.write("../results/pseudo-FMS.csv", df)
