include("utils.jl")
include("baselines.jl")


cellsize = 100.0
cityname = "porto"
region = SpatialRegion(cityname,
                       -8.735152, 40.953673,
                       -8.156309, 41.307945,
                       cellsize, cellsize,
                       100, # minfreq
                       60_000, # maxvocab_size
                       5, # k
                       4) # vocab_start
paramfile = "$(region.name)-param-cell$(Int(cellsize)).jld"
loadregion!(region, joinpath("../preprocessing", paramfile))

rate = 0.5
do_split = true
querydbfile = "querydb.h5"
createQueryDB("../preprocessing/porto.h5", 1_000_000+20_000, 1000, 100_000,
              (x,y)->downsampling(x, y, rate),(x,y)->downsampling(x, y, rate);
              do_split=do_split,
              querydbfile=querydbfile)
createTLabel(region, querydbfile; tfile="trj.t",labelfile="trj.label")
