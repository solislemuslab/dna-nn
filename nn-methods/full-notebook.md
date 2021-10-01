# To do
- no missingness with zero (daniel work)
- in the future, predict both antibiotics for pseudomonas: carb, toby like they do in Chen2019
- in future study conv.jl from flux zoo because we might want to keep the images as images
- loss that handles unbalance in phenotype
- fixit fit-nn: better model (based on papers), better loss (account for unbalanced), maybe run in loma

# Software remarks
- Keep in mind that we need master branch of JLD: `(v1.1) pkg> add JLD#master`

# Staph analyses

## Data

From Michelle email 6/5/19:
```
Alignment file for tree with only narsa strains:
https://drive.google.com/open?id=1_L2HzrV5xDiTEoJsS8HAFyQM09L64yFx

Alignment file for tree with argenteus outgroup:
https://drive.google.com/open?id=1Vg38RrshNarsRUOckWgvuUTahS8TpXNa
```

Both downloaded as `core_gene_alignment.aln`, but renamed as:
- `core_gene_alignment-narsa.aln`
- `core_gene_alignment-argenteus.aln`

For the phenotype (delta toxin), I copied it from `staphylococcus/data/data6`.

Then, Michelle also created the file containing only the SNPs, because it was difficult to read in julia the big file with all sites:
- `core_snps.snp_sites.aln`: 125 strains, 54152 sites


## Creating input files

This script takes:
- `core_snps.snp_sites.aln`: 125 strains, 54152 sites
- `"nrs_metadata3.txt"`
and creates the JLD files:
- `features.jld`
- `labels.jld`
with intermediate files:
- `responses.csv`
- `core_gene_alignment-narsa.csv`: 125 strains, 54152 sites
- `core_gene_alignment-narsa-subset.csv`: 125 strain, 27075 sites (part I)
- `core_gene_alignment-narsa-subset2.csv`: 125 strain, 27077 sites (part II)

```julia
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
```

We need to extract the IDs from the alignments:
```shell
cd Dropbox/Sharing/personal/ClauDan/work/grants/NIH/R01-neural-nets/Oct2019/preliminary-data/data/staph
awk -F ',' '{print $1}' core_gene_alignment-narsa.csv > ids.csv
```

```julia
## -----------------------------------
## Reading responses in correct order
## -----------------------------------
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
```

But the problem is that we cannot read easily the file `core_gene_alignment-narsa.csv` (125 rows=strains, ~900k columns=sites ih the original, ~50k in the SNPs one). We tried [CSVFiles.jl](https://github.com/queryverse/CSVFiles.jl) and [JuliaDB.jl](https://juliadb.org/) without success:
```
using CSV
df1 = CSV.read(string(datafolder,dat1), delim=','); 
## takes years, killed it
using CSVFiles
df1 = load(string(datafolder,dat1), nastrings=[ "NA", "NaN", "" ], header_exists=false) |> DataFrame 
## ERROR: StackOverflowError:
using JuliaDB
df1 = loadtable(string(datafolder,dat1), nastrings=[ "NA", "NaN", "" ], header_exists=false) 
## ERROR: StackOverflowError:
```

Professor Bates suggests:
```
df_seq = CSV.read(string(datafolder,seq), type=String, pool=true)
```
since all the julia functions are meant to read narrow tall matrices (fewer columns than rows), so it is difficult to read a fat matrix without specifying the type of the columns.
This got an stack overflown error too.

**Subset of the data:** We will take the first half of the data:
```shell
cd Dropbox/Sharing/personal/ClauDan/work/grants/NIH/R01-neural-nets/Oct2019/preliminary-data/data/staph
awk -F ',' '{for(i=1;i<=27076;i++)printf "%s ",$i;printf "\n"}' core_gene_alignment-narsa.csv > core_gene_alignment-narsa-subset.csv
awk -F ',' '{for(i=27077;i<=NF;i++)printf "%s ",$i;printf "\n"}' core_gene_alignment-narsa.csv > core_gene_alignment-narsa-subset2.csv
```

```julia
## -----------------------------
## Saving as julia data files
## -----------------------------
datafolder = "../data/staph/"
resp = "responses.csv"
seq = "core_gene_alignment-narsa-subset.csv"
seq2 = "core_gene_alignment-narsa-subset2.csv" 

df_resp = CSV.read(string(datafolder,resp),header=true)
## 125×2 DataFrame
df_seq = CSV.read(string(datafolder,seq), type=String, pool=true, header=false, delim=' ')
## 125×27077 DataFrame.
df_seq2 = CSV.read(string(datafolder,seq2), type=String, pool=true, header=false, delim=' ')
## 125×27078 DataFrame

size(df_seq) == (125,27077)
size(df_seq2) == (125,27078)

## convert to Matrix{UInt8}
mat_seq = convert(Matrix, df_seq)
sum(ismissing.(mat_seq[:,end])) == 125 && @warn "last column is all missing"
mat_seq = mat_seq[:,2:(end-1)] ## removing strain column and last column

mat = fill(convert(UInt8,1),size(mat_seq))
## to make type Union{missing, UInt8}; we don't need this if we use 0 instead of missing
##vc1 = vcat(missing,fill(convert(UInt8,1),size(mat_seq,1)-1)) 
##mat = hcat(vc1,mat)
##mat = mat[:,2:end]
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
typeof(mat) ## Array{Union{Missing, UInt8},2} or Array{UInt8,2}
size(mat) ## (125, 27075)

## convert to Matrix{UInt8}
mat_seq2 = convert(Matrix, df_seq2)
sum(ismissing.(mat_seq2[:,end])) == 125 && @warn "last column is all missing"
mat_seq2 = mat_seq2[:,1:(end-1)] ## removing strain column and last column

mat2 = fill(convert(UInt8,1),size(mat_seq2))
## to make type Union{missing, UInt8}; we don't need this if we use 0 instead of missing
##vc1 = vcat(missing,fill(convert(UInt8,1),size(mat_seq2,1)-1))
##mat2 = hcat(vc1,mat2)
##mat2 = mat2[:,2:end]
size(mat2) == size(mat_seq2) || error("sizes do not match!")

for i in 1:size(mat_seq2,1)  ##could not make this work without a for loop:(, at least not with missingness
    for j in 1:size(mat_seq2,2)
        if ismissing(mat_seq2[i,j])
            ## mat2[i,j] = missing ## flux cannot handle missingness; ML people love to impute with 0
            mat2[i,j] = convert(UInt8,0)
        elseif mat_seq2[i,j] == "A"
            mat2[i,j] = convert(UInt8,1)
        elseif mat_seq2[i,j] == "C"
            mat2[i,j] = convert(UInt8,2)
        elseif mat_seq2[i,j] == "G"
            mat2[i,j] = convert(UInt8,3)
        elseif mat_seq2[i,j] == "T"
            mat2[i,j] = convert(UInt8,4)
        end
    end
end
typeof(mat2) ## Array{Union{Missing, UInt8},2} or Array{UInt8,2}
size(mat2) ## (125, 27077)

## uniting two halves:
mat0 = hcat(mat,mat2)
## 125×54152 Array{Union{Missing, UInt8},2}:
## 125×54152 Array{UInt8,2} (when using 0 instead of missing)

## we need to remove one strain that has missing response
ind = findfirst(ismissing.(df_resp[:resp]))

labels = Array(df_resp[:resp])
deleteat!(labels,ind)
features = mat0' 
## 54152×125 LinearAlgebra.Adjoint{Union{Missing, UInt8},Array{Union{Missing, UInt8},2}}
## or
## 54152×125 LinearAlgebra.Adjoint{UInt8,Array{UInt8,2}} with 0 instead of missing
features = features[:,vcat(1:(ind-1),(ind+1):end)]
## 54152×124 Array{Union{Missing, UInt8},2}
## or
## 54152×124 Array{UInt8,2} with 0 instead of missing

## Save to JLD
save(string(datafolder,"features.jld"), "features", features)
save(string(datafolder,"labels.jld"), "labels", labels)
```

## Exploratory checks

We use the script `exploratory-response.jl` to see if we have a highly unbalanced case. For the staph data, we have 17% resistant strains, and 83% non-resistant, so we will need to adjust the loss function.

## Loss is NaN

We found an issue that loss was NaN when trying different types of loss functions.
It turns out that these features (SNPs) are not all variant. I found two sites (ind = [1788, 50737]) that are actually invariant, which is why after normalizing, we got NaN in the `normed_features`.

## Neural network fitting

### Understanding Flux.jl

Sadly, we cannot use `Mocha.jl` anymore as it is deprecated. We will use `Flux.jl` now, but we need to understand it.
For this, we will study the script [`iris.jl`](https://github.com/FluxML/model-zoo/blob/master/other/iris/iris.jl) in `scripts/others`.

### Running Flux.jl

Input files:
- `features.jld`
- `labels.jld`
- Model specification in `model-staph.jl`

Note: We have only one `fit-nn.jl`, and put the specific model in `model-staph.jl`.



# Pseudomonas analyses

## Images

### Data

 Courtesy of Jennifer Rattray. See [here](https://danielpimentel.github.io/teaching.html) and [here](https://www.dropbox.com/sh/1lp3ke597r9f1dt/AADhOtyEVroUKHd9Zw8XRUZXa?dl=0). The link for the testing folder was emailed by Jennifer.

 Folder `PIL_3dayLBCR-training` has:
    - `Perron_phenotype-GSU-training.xlsx` with phenotypes of interest for 234 strains. We will use 
        - `carb.lag.delta`: difference in lag between carbenicillin and control (antibiotic resistance)
        - `toby.lag.delta`: difference in lag between tobramycin and control (antibiotic resistance)
    - Images of the form: `PIL-X_3dayLBCR-Y.jpg` where X=strain (1-374 not consecutive, 69 strains only) and Y=image number

Folder `PIL_3dayLBCR-testing` has:
    - `Perron_phenotype-GSU-testing.xlsx` which we will not use because we will use the training one
    - Images of the form: `PIL-X_3dayLBCR-Y.jpg` where X=strain (7-345 not consecutive, 15 strains only) and Y=image number

### Creating input files

This script takes:
- `Perron_phenotype-GSU-training.xlsx`: 234 strains (1-374 not consecutive), 2 phenotypes of interest: `carb.lag.delta`, `toby.lag.delta`
- Images of the form: `PIL-X_3dayLBCR-1.jpg` in two folders: ``PIL_3dayLBCR-training` (69 strains), `PIL_3dayLBCR-testing` (17 strains)

and creates the JLD files (`xxx`=training or testing):
- `features-xxx.jld` : images
- `labels-carb-xxx.jld`: phenotype carb
- `labels-toby-xxx.jld`: phenotype toby
- `strain-xxx.jld`: strain ID (only for reference, not needed in NN)
- `labels-carb.jld`: phenotype carb concatenating training and testing
- `labels-toby.jld`: phenotype toby concatenating training and testing


#### Comparing strains with image and with phenotypes

We will check which are the strains with image
```shell
cd Dropbox/Sharing/personal/ClauDan/work/grants/NIH/R01-neural-nets/Oct2019/preliminary-data/data/pseudomonas/
ls PIL_3dayLBCR-training/*.jpg > strains-image-files-training.txt
ls PIL_3dayLBCR-testing/*.jpg > strains-image-files-testing.txt
```

And we will save the strain IDs in these files (as there are repeated images per strain):
```julia
using CSV, DataFrames
imgs = CSV.read("strains-image-files-training.txt", header=false)
ids = fill(0,size(imgs,1))
for i in 1:size(imgs,1)
    s = split.(split(imgs[i,1],'_'),'-')
    try 
        ids[i] = parse(Int,s[2][3])
    catch ##special case 55a
        ss = split(s[2][3],'a')
        ids[i] = parse(Int,ss[1])
    end
end

ids = unique(ids)
sort!(ids)
df = DataFrame(strains=ids)
CSV.write("strain-ids-images-training.txt",df)

imgs = CSV.read("strains-image-files-testing.txt", header=false)
ids = fill(0,size(imgs,1))
for i in 1:size(imgs,1)
    s = split.(split(imgs[i,1],'_'),'-')
    try 
        ids[i] = parse(Int,s[2][3])
    catch
        ss = split(s[2][3],'a')
        ids[i] = parse(Int,ss[1])
    end
end

ids = unique(ids)
sort!(ids)
df = DataFrame(strains=ids)
CSV.write("strain-ids-images-testing.txt",df)
```

Now, we need to compare to the strains that we have in the phenotype file. First, we need to save as CSV (manually): `Perron_phenotype-GSU-training.csv`.
```julia
cd("/Users/Clauberry/Dropbox/Sharing/personal/ClauDan/work/grants/NIH/R01-neural-nets/Oct2019/preliminary-data/data/pseudomonas")
pheno = CSV.read("Perron_phenotype-GSU-training.csv", header=true)
ids = unique(pheno[:strain]) ## 234
df = DataFrame(strains=ids)
CSV.write("strain-ids-pheno.txt",df)
```

Thus, we compare the strains:
```julia
using CSV
training = CSV.read("strain-ids-images-training.txt")
testing = CSV.read("strain-ids-images-testing.txt")
pheno = CSV.read("strain-ids-pheno.txt")

strainsImage = union(training[:strains],testing[:strains]) ## 86 strains
strainsImgPheno = intersect(pheno[:strains],strainsImage) ## 85 strains
setdiff(strainsImage, strainsImgPheno) ## strain 25
```
We do not have phenotype for strain 25, but we will remove it from now.

#### Reading in the images and phenotypes

Now, we want to read all the images for the 69 strains, and then the 17 strains. Note that the individual images are really big:
```julia
using FileIO, Images
img = FileIO.load("PIL_3dayLBCR-training/PIL-1_3dayLBCR-1.jpg")
## 9840×10560 Array{RGB4{N0f8},2} with eltype RGB4{Normed{UInt8,8}}
```
so we will use the [`ImageTransformations`](https://juliaimages.org/latest/imagetransformations/) package to reduce the resolution of the images slightly. Otherwise, the resulting input matrix (with all training images) is 60Gb big! Alternatively, we could have used `imagemagick` to manipulate images, but we decided to do everything in julia:
```
brew search imagemagick
convert ${inputfilename} -resize 10% ${outputfilename}
```
We will also store as gray images, instead of color.

We will do the following code **twice**: once for training and once for testing files.
Run for both training and testing:
```julia
cd("/Users/Clauberry/Dropbox/Sharing/personal/ClauDan/work/grants/NIH/R01-neural-nets/Oct2019/preliminary-data/data/pseudomonas")

## IMPORTANT: choose here which folder
folder = "training"
folder = "testing"

using FileIO, Images, CSV, DataFrames, JLD, ImageTransformations
files = CSV.read(string("strains-image-files-",folder,".txt"), header=false)
pheno = CSV.read("Perron_phenotype-GSU-training.csv", header=true)

images = Array{Array{Gray{Normed{UInt8,8}},2},1}(undef,length(files[:Column1]))
carb = Array{Union{Number,Missing},1}(undef, length(files[:Column1]))
toby = Array{Union{Number,Missing},1}(undef, length(files[:Column1]))
strain = Array{Number,1}(undef, length(files[:Column1]))
missInd = Integer[]

##rr = 1/8 ## images-training.jld is 946Mb
##rr = 1/2 ## images-training.jld is 15Gbs
##rr = 1/4 ## images-training.jld is 4Gbs
## instead of rr=1/4, we will use a fixed size because
## images have different sizes (size based on rr=1/4),
## but later we divided by 2, ow features ~13G
ll = div(2460,2)
ul = div(2640,2)
GC.gc()
count = [1]

for l in files[:Column1]
    @show l
    GC.gc()
    s = split.(split(l,'_'),'-')
    id = [0]
    try 
        id[1] = parse(Int,s[2][3])
    catch
        ss = split(s[2][3],'a')
        id[1] = parse(Int,ss[1])
    end
    ind = findall(x->x==id[1],pheno[:strain])
    @show id[1]
    strain[count[1]] = id[1]    
    if(length(ind) == 0)
        @warn "could not find the phenotype for strain $id"
        carb[count[1]] =  missing
        toby[count[1]] =  missing
        push!(missInd, count[1])
        ## we don't read the image in this case
    else
        length(ind) > 1 && error("More than one phenotype for strain $id")
        ii = ind[1]
        carb[count[1]] =  pheno[Symbol("carb.lag.delta")][ii]
        toby[count[1]] =  pheno[Symbol("toby.lag.delta")][ii]
        img = FileIO.load(l)
        images[count[1]] = Gray.(imresize(img,ll,ul))
    end
    count[1] += 1
end
GC.gc()

num = size(images,1) ## 266, 65

## for training only: need to remove strains with missing phenotypes
length(missInd) > 4 && error("there should be only one missing strain")
deleteat!(images,missInd)
deleteat!(carb,missInd)
deleteat!(toby,missInd)
deleteat!(strain,missInd)

## Save to JLD
save(string("images-",folder,".jld"), "images", images)
save(string("labels-carb-",folder,".jld"), "carb", carb)
save(string("labels-toby-",folder,".jld"), "toby", toby)
save(string("strain-",folder,".jld"), "strain", strain)

## Reading labels again to concatenate
carbtest = load("labels-carb-testing.jld", "carb")
## 65-element Array{Union{Missing, Number},1}
carbtrain = load("labels-carb-training.jld", "carb")
## 262-element Array{Union{Missing, Number},1}
tobytest = load("labels-toby-testing.jld", "toby")
## 65-element Array{Union{Missing, Number},1}
tobytrain = load("labels-toby-training.jld", "toby")
## 262-element Array{Union{Missing, Number},1}

## resistant:1 (-infinity -> 0) and susceptible:0 (0 -> +infinity)
carb = vcat(carbtrain, carbtest).<0
##327-element BitArray{1}
toby = vcat(tobytrain, tobytest).<0
##327-element BitArray{1}

## Saving as JLD as concatenated now
save("labels-carb.jld", "carb", carb)
save("labels-toby.jld", "toby", toby)
```

We are not doing creating the `features.jld` file because it would be too heavy. So, we do that in the `data-preprocess-pseudomonas-image.jl`, which will be called inside `fit-nn.jl`.

### Exploratory checks

We use the script `exploratory-response.jl` to see if we have a highly unbalanced case. For the pseudomonas image data: 
- `carb` response, we have 15% resistant strains, and 85% non-resistant
- `toby` response, we have 4% resistant strains, and 96% non-resistant (we will probably not used this response) 
so we will need to adjust the loss function.

### Neural network fitting: Running Flux.jl

Input files:
- `features.jld`
- `labels.jld`
- Model specification in `model-pseudomonas-images.jl`

Different models tried (in `model-pseudomonas-images.jl`):
- Model 1, Accuracy: 0.41284403669724773, Time: ~1hr
All the models are saved in `results/model-accuracy.csv`.

## Sequences

### Data

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


# Comparing to Random Forest

We will use the Julia package [DecisionTree](https://github.com/bensadeghi/DecisionTree.jl) to fit a random forest to compare the accuracy of NN. See `understand-input-decisiontrees.jl` to understand the type of input.
We will create `fit-randForest.jl` and rerun for all datasets. For some reason this feels better than modifying `fit-nn.jl`.