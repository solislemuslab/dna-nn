# Neural network models

We have worked done for the preliminary results on the submitted R01
All details of the preliminary work are in the `nn-methods/full-notebook.md` file, but we summarize below only the section corresponding to Pseudomonas and sequences.
Data in folder: `data/data2`.

Because we copied the following commands from the original `full-notebook.md`, paths might not work.

## Data

From [Dropbox folder](https://www.dropbox.com/sh/0clxfxxyaf8k6ds/AAArGoD46V5noKoD3piJQa20a?dl=0), we have the subfolder `CDS-OGs`, which we copy here.
This folder has 372 ortholog groups (OGs), one per file with the aligned sequences for all the taxa: `OG_X.aligned.fasta` with X=1,...,372.

We need to concatenate these sequences, but they are not in the same order of taxa, and not all files have all taxa. We use the script `concatenate-fasta.jl`.

- Input: `OG_X.aligned.fasta` files, one per OG
- Output: `concatenated.fasta`

### Creating input files to NN

This script takes:
- `concatenated.fasta`: 122 strains, 483,333 sites
- `../images/Perron_phenotype-GSU-training.csv` with phenotypes
- `matchingIDs.csv` copied from `Perron_Collection.xlsx` in the folder `Dropbox⁩/Sharing⁩/personal⁩/ClauDan⁩/work⁩/projects⁩/present⁩/sam-collaboration⁩/pseudomonas⁩/data⁩/CDS-PseudomonasLibrary⁩/Phenotype⁩`, and manually changed some IDs in the csv file to match the fasta file:
    - SWPA15J=NSWPA15a -> SWPA15J_NSWPA15a
    - CN573=PSE143 -> CN573_PSE143
    - PER11 grande colonie -> PER11
    - IDEXX Canine1 -> IDEXXCanine1
    - IDEXX Canine4 -> IDEXXCanine4
    - IDEXX Canine6 -> IDEXXCanine6
    - IDEXX Canine8 -> IDEXXCanine8
    - IDEXX Canine12 -> IDEXXCanine12
    - CPHL1999 grande colonie -> CPHL1999
    - J9UH1F grande colonie -> J9UH1F
    - LiA*/* -> LiA* (for example LiA131/2005 -> LiA131)
    - MC178 (LCV) -> MC178
    - PN3529(134)w -> PN3529 (not in fasta, but still changed)
    - Br670 grande colonie -> Br670
    - Br225 grande colonie -> Br225
    - Br231 grande colonie -> Br231
    - Br670 grande colonie -> Br670
    - Lo062 grande colonie -> Lo062
and creates the JLD files:
- `features.jld`
- `labels.jld`
with intermediate files:
- `responses.csv`
- `concatenated.csv`: 122 strains, 483,333 sites
- `concatenated-subsetX.csv`: 122 strain, 30,000 sites (part X=1,...,16)


```julia
using DataFrames, CSV, BioSequences, FileIO, JLD

## -----------------------------
## Reformatting fasta file
## -----------------------------
datafolder = "../data/pseudomonas/sequences/"
dat1 = "concatenated.fasta" ## input file
io1 = open(string(datafolder,dat1), "r")
out1 = "concatenated.csv" ## output file
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
```

We need to extract the IDs from the alignments:
```shell
cd Dropbox/Sharing/personal/ClauDan/work/grants/NIH/R01-neural-nets/Oct2019/preliminary-data/data/pseudomonas/sequences
awk -F ',' '{print $1}' concatenated.csv > ids.csv
```

```julia
## -----------------------------------
## Reading responses in correct order
## -----------------------------------
dat1 = "ids.csv"
dat2 = "../data/pseudomonas/images/Perron_phenotype-GSU-training.csv"
dat3 = "matchingIDs.csv"
dat4 = "../data/pseudomonas/images/Perron_phenotype-GSU-testing.csv"

df1 = CSV.read(string(datafolder,dat1), header=false)
df2 = CSV.read(dat2, header=true)
df4 = CSV.read(dat4, header=true)
df3 = CSV.read(string(datafolder,dat3), header=true)


## resistant:1 (-infinity -> 0) and susceptible:0 (0 -> +infinity)
carb = df2[Symbol("carb.lag.delta")] .< 0
toby = df2[Symbol("toby.lag.delta")] .< 0
carb2 = df4[Symbol("carb.lag.delta")] .< 0
toby2 = df4[Symbol("toby.lag.delta")] .< 0
carb = vcat(carb,carb2)
toby = vcat(toby,toby2)
strain = vcat(df2[:strain],df4[:strain])

pheno0 = DataFrame(LabID = strain, carb =  carb, toby = toby)
unique!(pheno0)

## matching of IDs
pheno = join(pheno0, df3, on = :LabID)
names!(df1,[:OriginalID])
pheno2 = join(df1, pheno, on = :OriginalID, kind=:left)

out2 = "responses.csv"
CSV.write(string(datafolder,out2), pheno2)
```

**Subsets of the data:** These commands are super inefficient!
```shell
cd Dropbox/Sharing/personal/ClauDan/work/grants/NIH/R01-neural-nets/Oct2019/preliminary-data/data/pseudomonas/sequences
awk -F ',' '{for(i=1;i<=30000;i++)printf "%s ",$i;printf "\n"}' concatenated.csv > concatenated-subset1.csv
awk -F ',' '{for(i=30001;i<=60000;i++)printf "%s ",$i;printf "\n"}' concatenated.csv > concatenated-subset2.csv
awk -F ',' '{for(i=60001;i<=90000;i++)printf "%s ",$i;printf "\n"}' concatenated.csv > concatenated-subset3.csv
awk -F ',' '{for(i=90001;i<=120000;i++)printf "%s ",$i;printf "\n"}' concatenated.csv > concatenated-subset4.csv
awk -F ',' '{for(i=120001;i<=150000;i++)printf "%s ",$i;printf "\n"}' concatenated.csv > concatenated-subset5.csv
awk -F ',' '{for(i=150001;i<=180000;i++)printf "%s ",$i;printf "\n"}' concatenated.csv > concatenated-subset6.csv
awk -F ',' '{for(i=180001;i<=210000;i++)printf "%s ",$i;printf "\n"}' concatenated.csv > concatenated-subset7.csv
awk -F ',' '{for(i=210001;i<=240000;i++)printf "%s ",$i;printf "\n"}' concatenated.csv > concatenated-subset8.csv
awk -F ',' '{for(i=240001;i<=270000;i++)printf "%s ",$i;printf "\n"}' concatenated.csv > concatenated-subset9.csv
awk -F ',' '{for(i=270001;i<=300000;i++)printf "%s ",$i;printf "\n"}' concatenated.csv > concatenated-subset10.csv
awk -F ',' '{for(i=300001;i<=330000;i++)printf "%s ",$i;printf "\n"}' concatenated.csv > concatenated-subset11.csv
awk -F ',' '{for(i=330001;i<=360000;i++)printf "%s ",$i;printf "\n"}' concatenated.csv > concatenated-subset12.csv
awk -F ',' '{for(i=360001;i<=390000;i++)printf "%s ",$i;printf "\n"}' concatenated.csv > concatenated-subset13.csv
awk -F ',' '{for(i=390001;i<=420000;i++)printf "%s ",$i;printf "\n"}' concatenated.csv > concatenated-subset14.csv
awk -F ',' '{for(i=420001;i<=450000;i++)printf "%s ",$i;printf "\n"}' concatenated.csv > concatenated-subset15.csv
awk -F ',' '{for(i=450001;i<=NF;i++)printf "%s ",$i;printf "\n"}' concatenated.csv > concatenated-subset16.csv
```

```julia
## -----------------------------
## Saving as julia data files
## -----------------------------
datafolder = "../data/pseudomonas/sequences/"

k=1 ## just to initialize mat0
seq = string("concatenated-subset",k,".csv")
df_seq = CSV.read(string(datafolder,seq), type=String, pool=true, header=false, delim=' ')

## convert to Matrix{UInt8}
mat_seq = convert(Matrix, df_seq)
sum(ismissing.(mat_seq[:,end])) == 122 && @warn "last column is all missing"
mat_seq = mat_seq[:,2:(end-1)] ## removing strain column and last column

mat = fill(convert(UInt8,1),size(mat_seq))
size(mat) == size(mat_seq) || error("sizes do not match!")

for i in 1:size(mat_seq,1)  ##could not make this work without a for loop:(, at least not with missingness
    for j in 1:size(mat_seq,2)
        if ismissing(mat_seq[i,j])
            ## mat[i,j] = missing ## flux cannot handle missingness; ML people love to impute with 0
            mat[i,j] = convert(UInt8,0)
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
typeof(mat) ## Array{UInt8,2}
size(mat) ## (122, 29999)

## Initializing mat0
mat0 = [mat]

for k in 2:16
    @show k
    seq = string("concatenated-subset",k,".csv")
    df_seq = CSV.read(string(datafolder,seq), type=String, pool=true, header=false, delim=' ')

    ## convert to Matrix{UInt8}
    mat_seq = convert(Matrix, df_seq)
    sum(ismissing.(mat_seq[:,end])) == 122 && @warn "last column is all missing"
    mat_seq = mat_seq[:,2:(end-1)] ## removing strain column and last column

    mat = fill(convert(UInt8,1),size(mat_seq))
    size(mat) == size(mat_seq) || error("sizes do not match!")

    for i in 1:size(mat_seq,1)  ##could not make this work without a for loop:(, at least not with missingness
        for j in 1:size(mat_seq,2)
            if ismissing(mat_seq[i,j])
                ## mat[i,j] = missing ## flux cannot handle missingness; ML people love to impute with 0
                mat[i,j] = convert(UInt8,0)
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
    typeof(mat) ## Array{UInt8,2}
    size(mat) ## (122, 29999)

    ## uniting two halves:
    mat0[1] = hcat(mat0[1],mat)
end 

features = mat0[1]' 
## 483318×122 LinearAlgebra.Adjoint{UInt8,Array{UInt8,2}}

resp = "responses.csv"
df_resp = CSV.read(string(datafolder,resp),header=true)
## 122×4 DataFrame
## we need to remove one strain that has missing response
ind = findall(ismissing.(df_resp[:carb])) 
ind = findall(ismissing.(df_resp[:toby]))
## it is the same in both: 4, 39, 115

carb = Array(df_resp[:carb])
toby = Array(df_resp[:toby])

deleteat!(carb,ind) ##119
deleteat!(toby,ind) ##119
features = features[:,vcat(1:(ind[1]-1),(ind[1]+1):end)]
features = features[:,vcat(1:(ind[2]-1),(ind[2]+1):end)]
features = features[:,vcat(1:(ind[3]-1),(ind[3]+1):end)]
## 483318×119 Array{UInt8,2}

## Save to JLD
save(string(datafolder,"features.jld"), "features", features)
save(string(datafolder,"labels-carb.jld"), "carb", carb)
save(string(datafolder,"labels-toby.jld"), "toby", toby)
```

### Neural network fitting: Running Flux.jl

Input files:
- `features.jld`
- `labels-carb.jld`
- `labels-toby.jld`
- Model specification in `model-pseudomonas-sequences.jl`

Different models tried (in `model-pseudomonas-sequences.jl`):
All the models are saved in `results/model-accuracy.csv`.


## Comparing to Random Forest

We will use the Julia package [DecisionTree](https://github.com/bensadeghi/DecisionTree.jl) to fit a random forest to compare the accuracy of NN. See `understand-input-decisiontrees.jl` to understand the type of input.
We will create `fit-randForest.jl` and rerun for all datasets. For some reason this feels better than modifying `fit-nn.jl`.

## Summarizing results in a plot:

R script: `plot-results.r` from the Rmd proposal.


# Machine learning project Fall 2019-2020
People
- Aryan Adhlakha (junior, CS)
- Lareina Liu (senior, stat)
- Zhaoyi (Kevin) Zhang (sophomore, CS+DS)

Folder: `ml-methods`.

## Data

Copied from `NIH/R01-neural-nets/Oct2019/preliminary-data`. We put in `data/data1`.

We have two datasets

- Data input (genomes) of Staph bacteria: `core_gene_alignment-narsa.aln` 
    - rows=bacterial strains (individuals)
    - columns=nucleotide sites (features)
- Labels for Staph bacteria (0=susceptible to antibiotic, 1=resistant to antibiotic): `responses-staph.csv`

- Data input (genomes) of Pseudomonas bacteria: `concatenated.fasta`
    - rows=bacterial strains (individuals)
    - columns=nucleotide sites (features)
- Labels for Pseudomonas bacteria (0=susceptible to antibiotic, 1=resistant to antibiotic): `responses-pseudo.csv`

### Data description (meeting 1/29)
- 2 datasets:
    - Pseudomonas Matrix: 122 by 483,333; missingness rate: 9.70% on average, 17.13% highest
        - Two antibiotics: carb and toby. Each label vector is true/false
            - carb: 79% false, 21% true; highly unbalanced, so we need to take this into account when we penalized wrong predictions. That is, a naive prediction predicting everything to be false will be ~79% accurate
            - toby: 95% false, 5% true
        - 261,868 variant columns out of the 483,333
        - __~125__ different condons in total
    - Staphylococcus Matrix: 125 by 983,088; missingness rate 1.19% on average, 13.03% highest
        - One antibiotic (unnamed). The label vector is 17% true, 83% false
        - 496,218 variant columns out of 983,088
        - __~116__ different condons in total
        - only one sequence contains 10 N nucleotides

Ideas to encode the nucleotides ACGT:
- ASCII code (disadvantage: are they still treated as numbers or categories?)
- One-hot encoding: A->0001, C->0010, G->0100, T->1000, - -> 0000

Questions:
- Complete the description of the data in terms of dimensions, missingness, balance of labels
- How do we treat missingness? Do we try to impute?
- Better to use only variant columns which reduces dimension by half approx, or to convert each codon (3 nucleotides) which reduces dimension by third?
  - grouping nucleotides into condons seems to significantly reduce the number of features in both data sets

Next steps:
- Calculate distance matrix from both datasets (Pseudomonas 122 by 122; Staph 125 by 125). Distances are defined as differences in genome sequence
- With distance matrix, we can cluster the bacteria into groups
- We can see if these clusters match the labels (resistant/susceptible) by plotting them
- Investigate transfer learning, data augmentation, and standard statistics methods like regression

### New data

- File `saureus_core_gene_alignment.tar.gz`
From Jon Moller (Tim Read's lab):
The core-genome alignment contains each set of core genes (common to all 263 strain genomes) for each strain in a FASTA alignment file. There are indeed 263 core genomes total as you stated.

## Analyses

Next steps: We want to fit statistical/machine-learning models to accomplish two tasks:
- feature selection: identify genes associated with antibiotic-resistance
- prediction: for a given new genome, can we predict whether it will be antibiotic-resistant or not

Methods:
- regression (we need to explore penalized regression because we have more features than individuals)
- random forest
- neural networks
- ...

### Main difficulties in this project
- Input data are categories/letters: ACGT, cannot be treated as numbers 1234
- There is correlation among rows and among columns. For example, every three letters in the genome corresponds to one "codon"
- Biologists want to know how many individual bacteria they need to sequence to train the method with high prediction accuracy 

### Previous work

Claudia had fit naive neural networks and random forest in Julia. All scripts in `scripts/previous-work`:
- `notebook.md`: explains pre-processing of the data
- `*.jl`: julia scripts, described in `notebook.md`


## Results and scripts

Everything is in the `ml-methods` folder. Note that we did not fit any NN in this part of the project. We fit logistic regression, SVMs and RFs.