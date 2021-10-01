## julia script to read the data/staph
## and create input files for Flux.jl
## Claudia July 2019

using DataFrames, CSV, BioSequences, FileIO, JLD

## -----------------------------
## Reformatting fasta file
## -----------------------------
datafolder = "../data/staph/"
dat1 = "core_snps.snp_sites.aln" ## input file
io1 = open(string(datafolder,dat1), "r")
out1 = "core_gene_alignment-narsa.csv" ## output file
outIO1 = open(string(datafolder,out1),"w")

lines1 = readlines(io1);
n = sum(occursin.('>',lines1))

io1 = open(string(datafolder,dat1), "r")
## First row:
reader = FASTA.Reader(io1)
record = FASTA.Record()
read!(reader,record)
line = string(FASTA.identifier(record),",",join(collect(FASTA.sequence(record)),','));
line2 = replace(line, "-"=>"");
write(outIO1,line2)
write(outIO1,"\n")

cont = [true]
while cont[1]
    record = FASTA.Record()
    try
        read!(reader,record)
    catch
        cont[1] = false
        continue
    end
    line = string(FASTA.identifier(record),",",join(collect(FASTA.sequence(record)),','));
    line2 = replace(line, "-"=>"");
    write(outIO1,line2)
    write(outIO1,"\n")
end
close(outIO1)


## -----------------------------------
## Reading responses in correct order
## -----------------------------------

## IMPORTANT: First we need to run in the terminal:
## cd Dropbox/Sharing/personal/ClauDan/work/grants/NIH/R01-neural-nets/June2019/preliminary-data/data/staph
## awk -F ',' '{print $1}' core_gene_alignment-narsa.csv > ids.csv

dat1 = "ids.csv"
dat2 = "nrs_metadata3.txt"

df1 = CSV.read(string(datafolder,dat1), header=false)
df2 = CSV.read(string(datafolder,dat2), delim='\t', header=true)

## Total.Area is the delta-toxin production. Based on Michelle Su:
## I used >= 20000 for the Total Area for the binary categories.
## For 4-5 categories:
## 0(can be its own, but I prefer it not to be)
## <=1000
## 1001-7000
## 7001-30000
## >30000

th = 20000
resp = df2[Symbol("Total.Area")] .> th
ids = df2[:sample_tag]

## genomes have one extra strain:
missing_strain = setdiff(df1[:Column1], df2[:sample_tag])
df = DataFrame(resp=vcat(resp,missing),ids=vcat(ids,missing_strain[1]))
sort!(df, [:ids])

sum(df[:ids] .== df1[:Column1]) == length(df1[:Column1])
## same order in predictors (core_gene_alignment-narsa.csv) and responses (df)

out2 = "responses.csv"
CSV.write(string(datafolder,out2), df)

## -----------------------------
## Saving as julia data files
## -----------------------------
datafolder = "../data/staph/"
resp = "responses.csv"
seq = "core_gene_alignment-narsa.csv"  ## subset bc problems reading the full data (see notebook.md)

df_resp = CSV.read(string(datafolder,resp),header=true)
df_seq = CSV.read(string(datafolder,seq), type=String, pool=true, header=false, delim=',');

## aqui voy: hacer subset x stackoverflown
## despues hacer fit-nn general, para poder usar con cualq dataset, y poner model
## en otro archivo tipo model-staph.jl, y include("model-staph.jl") en fit-nn
size(df_seq) ## (125,5003)

## convert to Matrix{UInt8}
mat_seq = convert(Matrix, df_seq)
vc1 = vcat(missing,fill(convert(UInt8,1),size(mat_seq,1)-1))
mat = fill(convert(UInt8,1),size(mat_seq))
mat = hcat(vc1,mat)
mat = mat[:,2:end]
size(mat) == size(mat_seq) || error("sizes do not match!")

for i in 1:size(mat_seq,1)  ##could not make this work without a for loop:(
    for j in 1:size(mat_seq,2)
        if ismissing(mat_seq[i,j])
            mat[i,j] = missing
        elseif mat_seq[i,j] == "A"
            mat[i,j] = convert(UInt8,1)
        elseif mat_seq[i,j] == "C"
            mat[i,j] = convert(UInt8,2)
        elseif mat_seq[i,j] == "G"
            mat[i,j] = convert(UInt8,3)
        elseif mat_seq[i,j] == "T"
            mat[i,j] = convert(UInt8,4)
        end
    end
end

typeof(mat) ## Array{Union{Missing, UInt8},2}

## we need to remove one strain that has missing response
ind = findfirst(ismissing.(df_resp[:resp]))

labels = df_resp[:resp]
deleteat!(labels,ind)
features = mat'
features = features[:,vcat(1:(ind-1),(ind+1):end)]
