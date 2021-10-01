## julia script to read the data/staph/responses.csv
## and study if we have an unbalances case
## Claudia July 2019

using DataFrames, CSV, JLD

## -----------------------------
## Which dataset?
## -----------------------------
datafolder = "../data/staph/"
labels = load(string(datafolder,"labels.jld"), "labels")

datafolder = "../data/pseudomonas/images/"
labels = load(string(datafolder,"labels-carb.jld"), "carb") ## here we chose the response: carb or toby
labels = load(string(datafolder,"labels-toby.jld"), "toby") ## here we chose the response: carb or toby
## -----------------------------

df = DataFrame(resp=labels)

prop_resistant = sum(df[:resp][.!ismissing.(df[:resp])]) / sum(.!ismissing.(df[:resp]))
prop_nonresistant = sum(.!df[:resp][.!ismissing.(df[:resp])]) / sum(.!ismissing.(df[:resp]))
