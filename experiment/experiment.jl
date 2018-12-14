cd("/Users/fineday/Github/t2vec/experiment/")
include("utils.jl")
include("baselines.jl")


cellsize = 25.0
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
## no dowsampling or distorting
#createQueryDB("../preprocessing/porto.h5", 1_000_000+20_000, 1000, 100_000,
#              (x,y)->(x,y), (x,y)->(x,y))
#createTLabel(region, "querydb.h5")
#createEDLCSSInput("querydb.h5")

# downsampling query with rate 0.x
rate, suffix = 0.5,  "-cell$(Int(cellsize))-down05"
do_split = true
#rate, suffix = 0.6, "-cell$(Int(cellsize))"
querydbfile = "querydb$suffix.h5"
createQueryDB("../preprocessing/porto.h5", 1_000_000+20_000, 1000, 100_000,
              #(x,y)->(x,y), (x,y)->(x,y);
              (x,y)->downsampling(x, y, rate),(x,y)->downsampling(x, y, rate);
              #(x,y)->distort(x, y, rate),(x,y)->distort(x, y, rate);
              do_split=do_split,
              querydbfile=querydbfile)
createTLabel(region, querydbfile; tfile="trj$suffix.t",labelfile="trj$suffix.label")
#createEDLCSSInput(querydbfile; tfile="edlcss$suffix.t",labelfile="edlcss$suffix.label")

##
vecfile, labelfile = "trj$suffix.h5", "trj$suffix.label"
#joinpath("most-similar-search", vecfile)
vecs = h5open(joinpath("", vecfile), "r") do f
    read(f["layer3"])
end
label = readdlm(labelfile, Int)
querysize, dbsize = 1000, 100000
query, db = vecs[:, 1:querysize], vecs[:, querysize+1:querysize+dbsize]
queryLabel, dbLabel = label[1:querysize], label[querysize+1:querysize+dbsize]
#ranks = ranksearch(query, queryLabel, db, dbLabel)
query, db = [query[:, i] for i in 1:size(query, 2)], [db[:, i] for i in 1:size(db, 2)]
ranks = ranksearch(query,
                   queryLabel,
                   db,
                   dbLabel, euclidean)
maximum(ranks)
mean(ranks)


##
trjfile, labelfile = "trj$suffix.t", "trj$suffix.label"
trjs = open(trjfile, "r") do f
    [readdlm(IOBuffer(line), Int)[:] for line in eachline(f)]
end
label = readdlm(labelfile, Int)
querysize, dbsize = 1000, 100000
query, db = trjs[1:querysize], trjs[querysize+1:querysize+dbsize]
queryLabel, dbLabel = label[1:querysize], label[querysize+1:querysize+dbsize]
ranks = ranksearch(query, queryLabel, db, dbLabel, commonset)
maximum(ranks)
mean(ranks)

## self-similarity
Tb = h5open("trj.h5", "r") do f
    read(f["layer3"])
end
Ta = h5open("trj-distort06.h5", "r") do f
    read(f["layer3"])
end
#d = colwise(Euclidean(), Ta, Tb)
#mean(d)
### cross-similarity
#da = colwise(Euclidean(), Ta, Ta[:, end:-1:1])
#db = colwise(Euclidean(), Tb, Tb[:, end:-1:1])
#mean(abs(db - da) ./ db)
## knn queries
querysize, dbsize = 1000, 10000
query1, db1 = Tb[:, 1:querysize], Tb[:, querysize+1:querysize+dbsize]
query2, db2 = Ta[:, 1:querysize], Ta[:, querysize+1:querysize+dbsize]
knnPrecision(query1, db1, query2, db2)


# scp experiment/trj.t xiucheng@155.69.144.85:t2vec/experiment/
# scp xiucheng@155.69.144.85:t2vec/experiment/trj.h5 ./experiment/

#  python t2vec.py -data "experiment" -vocab_size 18866 -checkpoint "best_model.pt" -mode 2
#  python t2vec.py -data "experiment" -vocab_size 35335 -checkpoint "best_model.pt" -mode 2
