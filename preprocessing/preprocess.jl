
using JSON
using DataStructures
using NearestNeighbors
using Serialization, ArgParse
include("SpatialRegionTools.jl")

args = let s = ArgParseSettings()
    @add_arg_table s begin
        "--datapath"
            arg_type=String
            default="/home/xiucheng/Github/t2vec/data"
    end
    parse_args(s; as_symbols=true)
end

datapath = args[:datapath]
param  = JSON.parsefile("../hyper-parameters.json")
regionps = param["region"]
cityname = regionps["cityname"]
cellsize = regionps["cellsize"]

if !isfile("$datapath/$cityname.h5")
    println("Please provide the correct hdf5 file $datapath/$cityname.h5")
    exit(1)
end

region = SpatialRegion(cityname,
                       regionps["minlon"], regionps["minlat"],
                       regionps["maxlon"], regionps["maxlat"],
                       cellsize, cellsize,
                       regionps["minfreq"], # minfreq
                       40_000, # maxvocab_size
                       10, # k
                       4) # vocab_start


println("Building spatial region with:
        cityname=$(region.name),
        minlon=$(region.minlon),
        minlat=$(region.minlat),
        maxlon=$(region.maxlon),
        maxlat=$(region.maxlat),
        xstep=$(region.xstep),
        ystep=$(region.ystep),
        minfreq=$(region.minfreq)")

paramfile = "$datapath/$(region.name)-param-cell$(Int(cellsize))"
if isfile(paramfile)
    println("Reading parameter file from $paramfile")
    region = deserialize(paramfile)
else
    println("Creating paramter file $paramfile")
    num_out_region = makeVocab!(region, "$datapath/$cityname.h5")
    serialize(paramfile, region)
end

println("Vocabulary size $(region.vocab_size) with cell size $cellsize (meters)")
println("Creating training and validation datasets...")
createTrainVal(region, "$datapath/$cityname.h5", datapath, downsamplingDistort, 1_000_000, 10_000)
saveKNearestVocabs(region, datapath)
