# Welcome and Data Modules

# Load Packages
using BenchmarkTools
using DataFrames
using DelimitedFiles
using CSV
using XLSX
using JLD
using NPZ
using RData
using RCall
using MAT

# Download data set
P = download("https://raw.githubusercontent.com/nassarhuda/easy_data/master/programming_languages.csv","programminglanguages.csv")

# How to read the file as a delimited file
P,H = readdlm("programminglanguages.csv",',';header = true)

# P is the data H is the header

# Write a delimited file as a text file
writedlm("programminglanguages_dlm.txt", P, '-')

# Read data as a csv file
C = CSV.read("programminglanguages.csv")

# Show the first 10 rows and all columns of the data frame
C[1:10,:]

C[:,:year]

C.year

# Line 24 and 26 produce the same output

# Benchmark delimited v CSV
# csv should be faster but its not {on the desktop, check laptop with new specs}
@btime P,H = readdlm("programminglanguages.csv",','; header = true);
@btime C = CSV.read("programminglanguages.csv"); # the ; supresses output

names(C)

describe(C)

# Writing a csv
CSV.write("programminglanguages_CSV.csv",DataFrame(P))

# Reading and writing excel files
T = XLSX.readdata("zillow_data_download_april2020.xlsx", #filename
    "Sale_counts_city", #sheet names
    "A1:F9" #cell range
    )

# If you don't know the cell ranges
G = XLSX.readtable("zillow_data_download_april2020.xlsx", "Sale_counts_city")
G[1]
G[1][1][1:10]
G[2][1:10]

## Dataframes
D = DataFrame(G...)  #Equivalent to DataFrame(G[1]G[2])

# Summary of the sizes of the data frame
by(D,:StateName, size)

# Create vectors to data frames then join them
foods = ["apple","cucumber","tomato","banana"]
calories = [105,47,22,105]
prices = [0.85, 1.6, 0.8, 0.6]
dataframe_calories = DataFrame(item = foods, calories = calories)
dataframe_prices = DataFrame(item = foods, price = prices)
DF = join(dataframe_calories,dataframe_prices, on=:item)

# We can also use the dataframe constructor on a matrix
DataFrame(T)

XLSX.writetable("writefile_using_XLSX2.xlsx",G[1],G[2])

# Need to figure out why I need the full file path. Working directory is the same so files should load without full path.

jld_data = JLD.load("E:\\Docs\\JuliaAcademy_NICK-DESKTOP_Sep-28-100002-2020_Conflict\\data_science\\mytempdata.jld")
save("mywrite.jld","A",jld_data)

npz_data = npzread("E:\\Docs\\JuliaAcademy_NICK-DESKTOP_Sep-28-100002-2020_Conflict\\data_science\\mytempdata.npz")
npzwrite("mywrite.npz",npz_data)

R_data = RData.load("E:\\Docs\\JuliaAcademy_NICK-DESKTOP_Sep-28-100002-2020_Conflict\\data_science\\mytempdata.rda")
@rput R_data
R"save(R_data, file = \"E:\\Docs\\JuliaAcademy_NICK-DESKTOP_Sep-28-100002-2020_Conflict\\data_science\\mywrite.rda\")"

Matlab_data = matread("E:\\Docs\\JuliaAcademy_NICK-DESKTOP_Sep-28-100002-2020_Conflict\\data_science\\mytempdata.mat")
matwrite("E:\\Docs\\JuliaAcademy_NICK-DESKTOP_Sep-28-100002-2020_Conflict\\data_science\\mywrite.mat",Matlab_data)

@show typeof(jld_data)
@show typeof(npz_data)
@show typeof(R_data)
@show typeof(Matlab_data)

jld_data

npz_data

Matlab_data

R_data

P

#Q1: Which year was a given language invented?
function year_created(P, language::String)
    loc = findfirst(P[:,2] .== language)
    return P[loc,1]
end

year_created(P,"Julia")

#example of error for data not in file
year_created(P, "W")

#sme function with error handling
function year_created_handle_error(P, language::String)
    loc = findfirst(P[:,2] .== language)
    !isnothing(loc) && return P[loc,1]
    error("Error: Language not found.")
end

year_created_handle_error(P, "W")

#Q2: How many languages were created in a given year?

function how_many_per_year(P, year::Int64)
    year_count = length(findall(P[:,1] .== year))
    return year_count
end
how_many_per_year(P, 2011)

P_df = C

#Q1 using a data frame
function year_created_2(P_df, language::String)
    loc = findfirst(P_df.language .== language)
    return P_df.year[loc]
end
year_created_2(P_df, "Julia")

function year_created_handle_error_2(P_df, language::String)
    loc = findfirst(P_df.language .== language)
    !isnothing(loc) && return P_df.year[loc]
    error("Error: Language not found.")
end
year_created_handle_error_2(P_df, "W")

#Q2 using a data frame
function how_many_per_year_2(P_df, year::Int64)
    year_count = length(findall(P_df.year .== year))
    return year_count
end

how_many_per_year_2(P_df, 2001)

Dict([("A",1),("B", 2), ("j",[1,2])])

P_dictionary = Dict{Integer, Vector{String}}()

P_dictionary[67] = ["julia","programming"]

P_dictionary["julia"] = 7
# Does not work because we declared the key must be an integer and the value has to be a vector{string}

dict = Dict{Integer, Vector{String}}()
for i = 1:size(P,1)
    year, lang = P[i,:]
    if year in keys(dict)
        dict[year] = push!(dict[year],lang)
    else
        dict[year] = [lang]
    end
end

curyear = P_df.year[1]
P_dictionary[curyear] = [P_df.language[1]]
# A smarter way to do this: #DOES NOT WORK UNLESS ABOVE 2 LINES ARE IN FOR LOOP
for (i, nextyear) in enumerate(P_df.year[2:end])
    if nextyear == curyear
        #same key
        P_dictionary[curyear] = push!(P_dictionary[curyear],P_df.language[i+1])
        # Note that push! is not our favorite but we're focusing on accuracy rather than speed here
    else
        curyear = nextyear
        P_dictionary[curyear] = [P_df.language[i+1]]
    end
end

length(keys(P_dictionary))

# Q1 with a dictionary
# Instead of looking in one long vector ,we'll look in many small vectors
function year_created_3(P_dictionary, language::String)
    keys_vec = collect(keys(P_dictionary))
    lookup = map(keyid -> findfirst(P_dictionary[keyid] .== language), keys_vec)
    # Now the lookup vector has nothing or a numeric value. we want to find the index of the numeric value
    return keys_vec[findfirst((!isnothing).(lookup))]
end

year_created_3(P_dictionary, "Julia")

# Benchmarking the year_created functions
@btime year_created(P, "Julia")
@btime year_created_2(P_df, "Julia")
@btime year_created_3(P_dictionary, "Julia")

#Q2 with a dictionary  DOES NOT WORK!!!
how_many_per_year_3(P_dictionary, year::Int64)= length(P_dictionary[year])

how_many_per_year_3(P_dictionary, 2011)

## Linear Algebra module

# Load Packages
using LinearAlgebra
using SparseArrays
using Images
using MAT

A = rand(10,10)
Atranspose = A'
A = A*Atranspose

# matrices are stored as a single column so you can use a single index insted of a column row pair
@show A[11] == A[1,2]

isposdef(A)

# Solving a system
b = rand(10);
# \ is more efficient that the inverse function
x = A\b;
@show norm(A*x-b) # Should be a very small number

@show typeof(A)
@show typeof(b)
@show typeof(rand(1,10))
@show typeof(Atranspose)

Matrix{Float64} == Array{Float64,2}
Vector{Float64} == Array{Float64,1}

Atranspose.parent # will display A

sizeof(A)

B = copy(Atranspose)
sizeof(B)

# Examples
A = [1 0; 1 -2];
B = [32; -4];

X = A\B
A*X == B

# LU Factorization
luA = lu(A)
norm(luA.L*luA.U - luA.P*A) # Should be a very small number

# QR Factorization
qrA = qr(A)
norm(qrA.Q*qrA.R - A) # Should be a very small number

# Cholesky factorization (note A must be symmetric positive definite)
cholA = cholesky(A)
norm(cholA.L*cholA.U - A) # Should be a very small number
factorize(A) # Returns convenient factorization; since A is symmetric pos def this is Cholesky

# Sparse Arrays
S = sprand(5,5, 2/5)
S.rowval # indeces of nonzero values
Matrix(S) # Makes the matrix dense
S.colptr
S.nzval # Shows nonzero values

S.m

# Images as matrices
X1 = load("khiam-small.jpg")

@show typeof(X1) #Images are arrays of RGB values
X1[1,1] #each entry is a pixel

Xgray = Gray.(X1) #transforms to gray scale

#extract the red, green and blue values from each pixel
R = map(i -> X1[i].r, 1:length(X1));
R = Float64.(reshape(R,size(X1)...));

G = map(i -> X1[i].g, 1:length(X1));
G = Float64.(reshape(G,size(X1)...));

B = map(i -> X1[i].b, 1:length(X1));
B = Float64.(reshape(B,size(X1)...));

Z = zeros(size(R)...); #matrix of zeros the same saize as the image
RGB.(Z,Z,B) #Displays the blue layer of the image
RGB.(Z,G,Z) #Displays the green layer
RGB.(R,Z,Z) #Displays the red layer

#back to the grayscale
Xgrayvalues = Float64.(Xgray)

#downsample using svd decomposition
SVD_V = svd(Xgrayvalues)

# use the top 4 singular vectors/values to form a new Image
u1 = SVD_V.U[:,1];
v1 = SVD_V.V[:,1];
img1 = SVD_V.S[1]*u1*v1';
for i = 2:4
    u1 = SVD_V.U[:,i]
    v1 = SVD_V.V[:,i]
    global img1 += SVD_V.S[i]*u1*v1'
end

Gray.(img1)
# still far off from original. let's do to 100
i = 1:100;
u1 = SVD_V.U[:,i];
v1 = SVD_V.V[:,i];
img1 = u1*spdiagm(0 =>SVD_V.S[i])*v1';

Gray.(img1)
#resulting image is pretty close to the original but much smaller storage (100 columns v 283)

norm(Xgrayvalues-img1)

#lots of rectangular facial images from Yale
M = matread("face_recog_qr.mat")

#Each vector M["V"] is a face image. let's reshape and look at the first one
q = reshape(M["V2"][:,1],192,168);
Gray.(q)

#can we find the closest image to the first one in the data set (facial recognition)
b = q[:]; #creates a queary image

A = M["V2"][:,2:end]; #original date minus first image
X = A\b;
Gray.(reshape(A*X,192,168))

norm(A*X-b)

#Let's complicate it. Add some noise and research
qv = q+rand(size(q,1),size(q,2))*0.5;
qv = qv./maximum(qv);
Gray.(qv)

b = qv[:]

X = A\b;
Gray.(reshape(A*X,192,168))

norm(A*X-b)

## Module on Statistics

using Statistics
using StatsBase
using RDatasets
using Plots
using StatsPlots
using KernelDensity
using Distributions
using LinearAlgebra
using HypothesisTests
using PyCall
using MLBase
# Load data from R on Old Faithful eruptions
D = dataset("datasets","faithful")
@show names(D)
D

describe(D)

# scatter plot of eruptions & wait time
eruptions = D[!, :Eruptions]
scatter(eruptions, label = "Eruptions")
waittime = D[!,:Waiting];
scatter!(waittime, label = "Time Between Eruptions")

# boxplot to see distribution
boxplot(["eruption length"],
    eruptions,
    legend = false,
    size = (200,400),
    whisker_width = 1,
    ylabel = "time in minutes")

# histogram to see distribution
histogram(eruptions,
    label = "eruptions")

# change bin whisker_width
histogram(eruptions,
    label = "eruptions",
    bins = :sqrt)

# kernel density estimates (kde)
p = kde(eruptions)

# histogram with kde overlaid
histogram(eruptions,
    label = "eruptions")
plot!(p.x,
    p.density.*length(eruptions),
    linewidth = 3,
    color = 2,
    label = "kde fit")
# More detailed version
histogram(eruptions,
    label = "eruptions",
    bins = :sqrt)
plot!(p.x,
    p.density.*length(eruptions).*0.2,
    linewidth = 3,
    color = 2,
    label = "kde fit")

# random data from a normal Distribution plus kde
myrandomvector = randn(100_100)
histogram(myrandomvector)
p = kde(myrandomvector)
plot!(p.x,
    p.density.*length(myrandomvector).*0.1,
    linewidth = 3,
    color = 2,
    label = "kde fit")

# prob distribution

# Normal
d = Normal()
myrandomvector = rand(d, 100000)
histogram(myrandomvector)
p = kde(myrandomvector)
plot!(p.x,
    p.density.*length(myrandomvector).*0.1,
    linewidth = 3,
    color = 2,
    label = "kde fit")

# Binomial
b = Binomial(40) # p = 0.5, n = 40
myrandomvector = rand(b, 100000)
histogram(myrandomvector)
p = kde(myrandomvector)
plot!(p.x,
    p.density.*length(myrandomvector).*0.5,
    linewidth = 3,
    color = 2,
    label = "kde fit")

# fit a random set of numbers to a distribution
x = rand(1000)
d = fit(Normal, x)
myrandomvector = rand(d, 1000)
histogram(myrandomvector,
    nbins = 20,
    fillalpha = 0.3,
    label = "fit")
histogram!(x,
    nbins = 20,
    fillalpha = 0.3,
    label = "myvector")

# fit eruptions to a normal data set
x = eruptions
d = fit(Normal, x)
myrandomvector = rand(d, 1000)
histogram(myrandomvector,
    nbins = 20,
    fillalpha = 0.3,
    label = "fit")
histogram!(x,
    nbins = 20,
    fillalpha = 0.3,
    label = "myvector")

# hypothesis testing
myrandomvector = randn(1000)
histogram(myrandomvector)
OneSampleTTest(myrandomvector) # shows normal

OneSampleTTest(eruptions) #shows not normal

# p-value of spearman and pearson correlation via python
scipy_stats = pyimport("scipy.stats")
####ERROR NEED TO INSTALL SCIPY STATS INTO PYTHON FIRST
@show scipy_stats.spearmanr(eruptions, waittime)
@show scipy_stats.pearsonr(eruptions, waittime)

scipy_stats.pearsonr(eruptions, eruptions)

corspearman(eruptions, waittime)

cor(eruptions, waittime)

scatter(eruptions,
    waittime,
    xlabel = "eruption length",
    ylabel = "wait time between eruptions",
    legend = false,
    size = (400,300))

# AUC scores or confusion matrix
gt = [1, 1, 1, 1, 1, 1, 1, 2];
pred = [1,1,2,2,1,1,1,1];
C = confusmat(2, gt, pred); #confusion matrix
C ./ sum(C, dims = 2) #normalize per class
sum(diag(C) / length(gt)) #compute correct rate
correctrate(gt, pred)

gt = [1, 1, 1, 1, 1, 1, 1, 0]; # ground truth
pred = [1, 1, 0, 0, 1, 1, 1, 1]; # prediction
ROC = roc(gt, pred) #from MLBase package
recall(ROC)
precision(ROC)

## Dimensionality Reduction
using XLSX
using VegaDatasets
using MultivariateStats
using RDatasets
using StatsBase
using Statistics
using LinearAlgebra
using Plots
using ScikitLearn
using Makie
using MLBase
using UMAP
using Distances

c = DataFrame(VegaDatasets.dataset("cars"))

dropmissing!(c)
M = Matrix(c[:,2:7])
names(c)

car_origin = c[:, :Origin]
carmap = labelmap(car_origin) #from MLBase
uniqueids = labelencode(carmap, car_origin)
# PCA
# Center and normalize data
data = M;
data = (data .- mean(data, dims = 1))./ std(data, dims = 1)

p = fit(PCA, data', maxoutdim = 2) #from MultivariateStats
P = projection(p) # projection matrix

Yte = MultivariateStats.transform(p, data') #aplies projection matrix to each car
Xr = reconstruct(p, Yte) #reconstruct testing observations - approximately
norm(Xr-data')

#base plot
Plots.scatter(Yte[1,:], Yte[2,:])

#labelled Plot
Plots.scatter(Yte[1,car_origin.== "USA"], Yte[2,car_origin.== "USA"], color = 1, label = "USA");
Plots.xlabel!("pca component 1");
Plots.ylabel!("pca component 2");
Plots.scatter!(Yte[1,car_origin.== "Japan"], Yte[2,car_origin.== "Japan"], color = 2, label = "Japan");
Plots.scatter!(Yte[1,car_origin.== "Europe"], Yte[2,car_origin.== "Europe"], color = 3, label = "Europe")

# 3d Reduction with 3d scatter
p = fit(PCA, data', maxoutdim = 3);
Yte = MultivariateStats.transform(p, data');
scatter3d(Yte[1,:],Yte[2,:],Yte[3,:],color= uniqueids, legend = false)

scene = Makie.scatter(Yte[1,:],Yte[2,:],Yte[3,:],color= uniqueids);
display(scene)

## t-SNE
@sk_import manifold : TSNE;
tfn = TSNE(n_components = 2); # ,perplexity=20.0, early_exaggeration = 50)
Y2 = tfn.fit_transform(data);
Plots.scatter(Y2[:,1],Y2[:,2], color = uniqueids, legend = false, size = (400,300), markersize = 3)

## UMAP
L = cor(data, data, dims =2); #each element is correlation between 2 cars
embedding = umap(L, 2)
Plots.scatter(embedding[1,:], embedding[2,:], color = uniqueids)

L2 = pairwise(Euclidean(), data, data, dims = 1);
embedding2 = umap(-L2, 2);
Plots.scatter(embedding2[1,:],embedding2[2,:], color = uniqueids)

#Interesting finding: for sure 2 clusters mostly US, one cluster US, Japan, Europe
