
using JSON
using DataStructures
using NearestNeighbors
using Serialization
include("SpatialRegionTools.jl")

param  = JSON.parsefile("../hyper-parameters.json")
regionps = param["region"]
cityname = regionps["cityname"]
cellsize = regionps["cellsize"]

if !isfile("../data/$cityname.h5")
    println("Please provide the correct hdf5 file ../data/$cityname.h5")
    exit(1)
end

println("Building spatial region with:
         cityname=$cityname,
         minlon=$(regionps["minlon"]),
         minlat=$(regionps["minlat"]),
         maxlon=$(regionps["maxlon"]),
         maxlat=$(regionps["maxlat"]),
         cellsize=$cellsize,
         minfreq=$(regionps["minfreq"])")

region = SpatialRegion(cityname,
                       regionps["minlon"], regionps["minlat"],
                       regionps["maxlat"], regionps["maxlat"],
                       cellsize, cellsize,
                       regionps["minfreq"], # minfreq
                       40_000, # maxvocab_size
                       10, # k
                       4) # vocab_start

paramfile = "../data/$(region.name)-param-cell$(Int(cellsize))"
if isfile(paramfile)
    println("Reading parameter file from $paramfile")
    #loadregion!(region, paramfile)
    region = deserialize(paramfile)
else
    println("Creating paramter file $paramfile")
    num_out_region = makeVocab!(region, "../data/$cityname.h5")
    #saveregion(region, paramfile)
    serialize(paramfile, region)
end

#println("Making Vocabulary...")
#num_out_region = makeVocab!(region, "../data/$cityname.h5")

println("Vocabulary size $(region.vocab_size) with cell size $cellsize (meters)")
createTrainVal(region, "../data/$cityname.h5", downsamplingDistort, 1_000_000, 10_000)
saveKNearestVocabs(region)

#region.cellcount |> keys |> maximum
#region.cellcount |> keys |> minimum
#region.cellcount |> length
#region.cellcount |> values |> sum
#region.cellcount |> values |> minimum
#region.cellcount |> values |> maximum
#region.vocab2hotcell[1638]
#region.vocab2hotcell |> keys |> maximum
#region.vocab_size
## test cell2gps(), gps2cell()
#for cell in 1:region.numx*region.numy-1
#    gps = cell2gps(region, cell)
#    @assert cell == gps2cell(region, gps...) "cell $cell bad"
#end

#cell = region.hotcell[10]
#kcells, dists = knearestHotcells(region, cell, 10)
#kvocabs = map(x->region.hotcell2vocab[x], kcells)
#[(v,d) for (v, d) in zip(kvocabs, dists)]

#saveKNearestVocabs(region)
#V, D = h5open("$cityname-vocab-dist.h5", "r") do f
#           f["V"]|>read, f["D"]|>read
#end

#seq = " 793 14350  8606 23750 13308" |> split |> x->map(parse, x)
#seq = [region.hotcell2vocab[c] for c in region.hotcell]
#trip = seq2trip(region, seq)
#h5open("/tmp/trip.h5", "w") do f f["trip"] = trip end

#a = sort(collect(region.cellcount), by=last, rev=true)[1:40000]
#region.hotcell
#a[end] |> last
