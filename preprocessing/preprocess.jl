
include("SpatialRegionTools.jl")

isfile("porto.h5") || porto2h5("../data/porto.csv")

cellsize = 100.0
cityname = "porto"
region = SpatialRegion(cityname,
                       -8.735152, 40.953673,
                       -8.156309, 41.307945,
                       cellsize, cellsize,
                       100, # minfreq
                       40_000, # maxvocab_size
                       10, # k
                       4) # vocab_start
#region.numx * region.numy
#region.maxy - region.miny
#region.maxx - region.minx
paramfile = "$(region.name)-param-cell$(Int(cellsize)).jld"
if isfile(paramfile)
    println("Reading parameter file from $paramfile")
    loadregion!(region, paramfile)
else
    println("Creating paramter file $paramfile")
    num_out_region = makeVocab!(region, "$cityname.h5")
    saveregion(region, paramfile)
end
#num_out_region
println("Vocabulary size $(region.vocab_size) with cell size $cellsize (meters)")
createTrainVal(region, "$cityname.h5", downsamplingDistort, 1_000_000, 10_000)
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
