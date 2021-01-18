#module SpatialRegionTools

using StatsBase:rle
using HDF5
using DataStructures
using NearestNeighbors
using Statistics, Printf
include("utils.jl")

#export SpatialRegion, makeVocab!, gps2vocab, saveKNearestVocabs,
#       trip2seq, seq2trip, createTrainVal

const UNK = 3

"""
example:

region = SpatialRegion("porto",
                       -8.735152, 40.953673,
                       -8.156309, 41.307945,
                       cellsize, cellsize,
                       50, # minfreq
                       50_000, # maxvocab_size
                       5, # k
                       4) # vocab_start
"""
mutable struct SpatialRegion
    name::String
    ## bounding box
    minlon::Float64
    minlat::Float64
    maxlon::Float64
    maxlat::Float64
    minx::Float64
    miny::Float64
    maxx::Float64
    maxy::Float64
    xstep::Float64
    ystep::Float64
    numx::Int
    numy::Int
    minfreq::Int
    maxvocab_size::Int
    k::Int
    ## the number of hitting
    cellcount
    ## hot cells
    hotcell::Vector{Int}
    ## hot cell kdtree
    hotcell_kdtree
    ## map a hot cell into the vocabulary id
    hotcell2vocab::Dict{Int, Int}
    ## map the vocabulary id into its hot cell
    vocab2hotcell::Dict{Int, Int}
    ## vocab start
    vocab_start::Int
    ## vocabulary size
    vocab_size::Int
    ## whether indices have been built
    built::Bool
    function SpatialRegion(name::String,
                           minlon::Float64,
                           minlat::Float64,
                           maxlon::Float64,
                           maxlat::Float64,
                           xstep::Float64,
                           ystep::Float64,
                           minfreq::Int,
                           maxvocab_size::Int,
                           k::Int,
                           vocab_start::Int)
        minx, miny = lonlat2meters(minlon, minlat)
        maxx, maxy = lonlat2meters(maxlon, maxlat)
        numx = round(maxx - minx, digits=6) / xstep
        numx = convert(Int, ceil(numx))
        numy = round(maxy - miny, digits=6) / ystep
        numy = convert(Int, ceil(numy))
        new(name,
            minlon, minlat, maxlon, maxlat,
            minx, miny, maxx, maxy,
            xstep, ystep,
            numx, numy, minfreq, maxvocab_size, k,
            Accumulator{Int, Int}(),
            Int[],
            Any,
            Dict(),
            Dict(),
            vocab_start,
            vocab_start,
            false)
    end
end

#function gps2cell(region::SpatialRegion, lon::Float64, lat::Float64)
#    lonoffset = round(lon - region.minlon, 6) / region.hstep
#    latoffset = round(lat - region.minlat, 6) / region.vstep
#    lonoffset = convert(Int64, floor(lonoffset))
#    latoffset = convert(Int64, floor(latoffset))
#    latoffset * region.numlon + lonoffset
#end
#
#function cell2gps(region::SpatialRegion, cell::Int)
#    latoffset = div(cell, region.numlon)
#    lonoffset = mod(cell, region.numlon)
#    lat = region.minlat + (latoffset + 0.5) * region.vstep
#    lon = region.minlon + (lonoffset + 0.5) * region.hstep
#    lon, lat
#end

#cell2gps(region, 0)
#gps2cell(region, (126.48349, 45.6468)...)

"""
Web Mercator coordinate to cell id
"""
function coord2cell(region::SpatialRegion, x::Float64, y::Float64)
    xoffset = round(x - region.minx, digits=6) / region.xstep
    yoffset = round(y - region.miny, digits=6) / region.ystep
    xoffset = convert(Int, floor(xoffset))
    yoffset = convert(Int, floor(yoffset))
    yoffset * region.numx + xoffset
end

"""
Cell id to Web Mercator coordinate
"""
function cell2coord(region::SpatialRegion, cell::Int)
    yoffset = div(cell, region.numx)
    xoffset = mod(cell, region.numx)
    y = region.miny + (yoffset + 0.5) * region.ystep
    x = region.minx + (xoffset + 0.5) * region.xstep
    x, y
end

function gps2cell(region::SpatialRegion, lon::Float64, lat::Float64)
    x, y = lonlat2meters(lon, lat)
    coord2cell(region, x, y)
end

function cell2gps(region::SpatialRegion, cell::Int)
    x, y = cell2coord(region, cell)
    meters2lonlat(x, y)
end

function gps2offset(region::SpatialRegion, lon::Float64, lat::Float64)
    x, y = lonlat2meters(lon, lat)
    xoffset = round(x - region.minx, digits=6) / region.xstep
    yoffset = round(y - region.miny, digits=6) / region.ystep
    xoffset, yoffset
end



#lon, lat = -8.735152, 40.953673
#x, y = lonlat2meters(lon, lat)
#coord2cell(region, x, y+300)
#cell2coord(region, 10)


function inregion(region::SpatialRegion, lon::Float64, lat::Float64)
    lon >= region.minlon && lon < region.maxlon &&
    lat >= region.minlat && lat < region.maxlat
end

function inregion(region::SpatialRegion, trip::Matrix{Float64})
    for i = 1:size(trip, 2)
        inregion(region, trip[:, i]...) || return false
    end
    true
end

#inregion(region, -8.503, 41.35)
#inregion(region, [-8.503 -8.5; 41.01 41.01])

"""
Build the vocabulary from the raw trajectories stored in the hdf5 file.
For a trip (trajectory), each point lies in a column when reading with Julia
while it lies in a row if reading with Python.
"""
function makeVocab!(region::SpatialRegion, trjfile::String)
    region.cellcount = Accumulator{Int, Int}()
    ## useful for evaluating the size of region bounding box
    num_out_region = 0
    ## scan all trips (trajectories)
    h5open(trjfile, "r") do f
        num = read(attributes(f)["num"])
        for i = 1:num
            trip = read(f["/trips/$i"])
            for p = 1:size(trip, 2)
                lon, lat = trip[:, p]
                if !inregion(region, lon, lat)
                    num_out_region += 1
                else
                    cell = gps2cell(region, lon, lat)
                    push!(region.cellcount, cell)
                end
            end
            i % 100_000 == 0 && println("Processed $i trips")
            #i >= 8_000 && break
        end
    end
    ## filter out all hot cells
    max_num_hotcells = min(region.maxvocab_size, length(region.cellcount))
    topcellcount = sort(collect(region.cellcount), by=last, rev=true)[1:max_num_hotcells]
    println("Cell count at max_num_hotcells:$(max_num_hotcells) is $(last(topcellcount[end]))")
    #region.hotcell =
    #    filter((k,v)->v >= region.minfreq, Dict(topcellcount))|>keys|>collect
    region.hotcell = filter(p -> last(p) >= region.minfreq, topcellcount) .|> first
    ## build the map between cell and vocab id
    region.hotcell2vocab = Dict([(cell, i-1+region.vocab_start)
        for (i, cell) in enumerate(region.hotcell)])
    #region.vocab2hotcell = map(reverse, region.hotcell2vocab)
    region.vocab2hotcell = Dict(last(p) => first(p) for p in region.hotcell2vocab)
    ## vocabulary size
    region.vocab_size = region.vocab_start + length(region.hotcell)
    ## build the hot cell kdtree to facilitate search
    coord = hcat(map(x->collect(cell2coord(region, x)), region.hotcell)...)
    region.hotcell_kdtree = KDTree(coord)
    region.built = true

    num_out_region
end

function knearestHotcells(region::SpatialRegion, cell::Int, k::Int)
    @assert region.built == true "Build index for region first"
    coord = cell2coord(region, cell) |> collect
    idxs, dists = knn(region.hotcell_kdtree, coord, k)
    region.hotcell[idxs], dists
end

function nearestHotcell(region::SpatialRegion, cell::Int)
    @assert region.built == true "Build index for region first"
    hotcell, _ = knearestHotcells(region, cell, 1)
    hotcell[1]
end

"""
k-nearest vocabs and corresponding distances for each vocab.

This is used in training for KLDiv loss.
"""
function saveKNearestVocabs(region::SpatialRegion, datapath::String)
    V = zeros(Int, region.k, region.vocab_size)
    D = zeros(Float64, region.k, region.vocab_size)
    for vocab in 0:region.vocab_start-1
        V[:, vocab+1] .= vocab
        D[:, vocab+1] .= 0.0
    end
    for vocab in region.vocab_start:region.vocab_size-1
        cell = region.vocab2hotcell[vocab]
        kcells, dists = knearestHotcells(region, cell, region.k)
        kvocabs = map(x->region.hotcell2vocab[x], kcells)
        V[:, vocab+1] .= kvocabs
        D[:, vocab+1] .= dists
    end
    cellsize = Int(region.xstep)
    file = joinpath(datapath, region.name * "-vocab-dist-cell$(cellsize).h5")
    h5open(file, "w") do f
        f["V"], f["D"] = V, D
    end
    println("Saved cell distance into $file")
    nothing
end

"""
Return the vocab id for a cell in the region.
If the cell is not hot cell, the function will first search its nearest
hotcell and return the corresponding vocab id.
"""
function cell2vocab(region::SpatialRegion, cell::Int)
    @assert region.built == true "Build index for region first"
    if haskey(region.hotcell2vocab, cell)
        return region.hotcell2vocab[cell]
    else
        hotcell = nearestHotcell(region, cell)
        return region.hotcell2vocab[hotcell]
    end
end


"""
Mapping a gps point to the vocab id in the vocabulary consists of hot cells,
each hot cell has an unique vocab id (hotcell2vocab).

If the point falls out of the region, `UNK` will be returned.
If the point falls into the region, but out of the hot cells its nearest hot cell
will be used.
"""
function gps2vocab(region::SpatialRegion, lon::Float64, lat::Float64)
    inregion(region, lon, lat) || return UNK
    cell2vocab(region, gps2cell(region, lon, lat))
end

function trip2seq(region::SpatialRegion, trip::Matrix{Float64})
    seq = Int[]
    for i in 1:size(trip, 2)
        lon, lat = trip[:, i]
        push!(seq, gps2vocab(region, lon, lat))
    end
    seq |> rle |> first
end

function seq2trip(region::SpatialRegion, seq::Vector{Int})
    trip = zeros(Float64, 2, length(seq))
    for i = 1:length(seq)
        cell = get(region.vocab2hotcell, seq[i], -1)
        cell == -1 && error("i=$i is out of vocabulary")
        lon, lat = cell2gps(region, cell)
        trip[:, i] = [lon, lat]
    end
    trip
end

function tripmeta(region::SpatialRegion, trip::Matrix{Float64})
    # centroid(x) = minimum(x) + (maximum(x) - minimum(x)) / 2
    # xs = Vector{Float64}(undef, size(trip, 2))
    # ys = Vector{Float64}(undef, size(trip, 2))
    # for i in 1:size(trip, 2)
    #     lon, lat = trip[:, i]
    #     xs[i], ys[i] = gps2coarseoffset(region, lon, lat)
    # end
    # centroid(xs) / region.xstep, centroid(ys) / region.ystep
    mins, maxs = minimum(trip, dims=2), maximum(trip, dims=2)
    centroids = mins + (maxs - mins) / 2
    gps2offset(region, centroids...)
end

function seqmeta(region::SpatialRegion, seq::Vector{Int})
    trip = seq2trip(region, seq)
    tripmeta(region, trip)
end

"""
Create training and validation dataset

createTrainVal(region, "porto.h5", downsampling, 50, 10)
"""
function createTrainVal(region::SpatialRegion,
                        trjfile::String,
                        datapath::String,
                        injectnoise::Function,
                        ntrain::Int,
                        nval::Int;
                        nsplit=5,
                        min_length=20,
                        max_length=100)
    seq2str(seq) = join(map(string, seq), " ") * "\n"
    h5open(trjfile, "r") do f
        trainsrc = open("$datapath/train.src", "w")
        traintrg = open("$datapath/train.trg", "w")
        trainmta = open("$datapath/train.mta", "w")

        validsrc = open("$datapath/val.src", "w")
        validtrg = open("$datapath/val.trg", "w")
        validmta = open("$datapath/val.mta", "w")
        for i = 1:ntrain+nval
            trip = f["/trips/$i"] |> read
            min_length <= size(trip, 2) <= max_length || continue
            trg = trip2seq(region, trip) |> seq2str
            meta = tripmeta(region, trip)
            mta = @sprintf "%.2f %.2f\n" meta[1] meta[2]

            noisetrips = injectnoise(trip, nsplit)
            srcio, trgio, mtaio = i <= ntrain ? (trainsrc, traintrg, trainmta) : (validsrc, validtrg, validmta)
            for noisetrip in noisetrips
                ## here: feel weird
                #src = noisetrip |> trip2seq |> seq2str
                src = trip2seq(region, noisetrip) |> seq2str
                write(srcio, src)
                write(trgio, trg)
                write(mtaio, mta)
            end
            i % 100_000 == 0 && println("Scaned $i trips...")
            #i >= 8_000 && break
        end
        close(trainsrc), close(traintrg), close(trainmta), close(validsrc), close(validtrg), close(validmta)
    end
    nothing
end

function saveregion(region::SpatialRegion, paramfile::String)
    save(paramfile,
          # JLD cannot handle Accumulator correctly
          #"cellcount", region.cellcount.map,
          "hotcell", region.hotcell,
          "hotcell2vocab", region.hotcell2vocab,
          "vocab2hotcell", region.vocab2hotcell,
          "hotcell_kdtree", region.hotcell_kdtree,
          "vocab_size", region.vocab_size)
    nothing
end

function loadregion!(region::SpatialRegion, paramfile::String)
    jldopen(paramfile, "r") do f
        #region.cellcount = read(f, "cellcount")
        region.hotcell = read(f, "hotcell")
        region.hotcell2vocab = read(f, "hotcell2vocab")
        region.vocab2hotcell = read(f, "vocab2hotcell")
        region.hotcell_kdtree = read(f, "hotcell_kdtree")
        region.vocab_size = read(f, "vocab_size")
        region.built = true
    end
    nothing
end

#end # module
#createTrainVal(region, "porto.h5", downsampling, 50, 10)
