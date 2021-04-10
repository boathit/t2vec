using HDF5
using NearestNeighbors
using Distances
using DelimitedFiles
#include("../preprocessing/utils.jl")
include("../preprocessing/SpatialRegionTools.jl")

function uniformsplit(trip::Matrix{Float64}, timestamp::Vector{Float64})
    n = size(trip, 2)

    #idx1 = append!([1], 2:2:n-1)
    #idx2 = append!([1], 3:2:n-1)
    #push!(idx1, n), push!(idx2, n)
    idx1, idx2 = 1:2:n, 2:2:n
    copy(trip[:, idx1]), copy(timestamp[idx1]), copy(trip[:, idx2]), copy(timestamp[idx2])
end

"""
rate is dropping rate
"""
function downsampling(trip::Matrix{Float64}, timestamp::Vector{T}, rate::Float64) where {T<:Real}
    keep_idx = Int[1]
    for i = 2:size(trip, 2)-1
        rand() > rate && push!(keep_idx, i)
    end
    push!(keep_idx, size(trip, 2))
    trip[:, keep_idx], timestamp[keep_idx]
end

"""
rate is noise rate
"""
function distort(trip::Matrix{Float64}, timestamp::Vector{T}, rate::Float64; radius=50.0) where {T<:Real}
    noisetrip = copy(trip)
    for i = 1:size(noisetrip, 2)
        if rand() <= rate
            x, y = lonlat2meters(noisetrip[:, i]...)
            xnoise, ynoise = 2rand()-1, 2rand()-1
            normz = hypot(xnoise, ynoise)
            xnoise, ynoise = xnoise*radius/normz, ynoise*radius/normz
            noisetrip[:, i] = meters2lonlat(x + xnoise, y + ynoise) |> collect
        end
    end
    noisetrip, timestamp
end

"""
createQueryDB("../preprocessing/porto.h5", 1_000_000+20_000, 1000, 100_000,
              (x,y)->(x,y), (x,y)->(x,y))
"""
function createQueryDB(trjfile::String, start::Int,
                       querysize::Int, dbsize::Int,
                       querynoise::Function, dbnoise::Function;
                       do_split=true,
                       querydbfile="querydb.h5",
                       min_length=30,
                       max_length=100)
    nquery, ndb = 0, 0
    h5open(trjfile, "r") do f
        querydbf = h5open(querydbfile, "w")
        num = read(attributes(f)["num"])
        for i = start:num
            trip = read(f["/trips/$i"])
            timestamp = read(f["/timestamps/$i"])
            if nquery < querysize
                if 2min_length <= size(trip, 2) <= 2max_length
                    if do_split
                        nquery += 1
                        ndb += 1
                        trip1, timestamp1, trip2, timestamp2 = uniformsplit(trip, timestamp)
                        querydbf["/query/trips/$nquery"], querydbf["/query/timestamps/$nquery"] = querynoise(trip1, timestamp1)
                        querydbf["/query/names/$nquery"] = i
                        querydbf["/db/trips/$ndb"], querydbf["/db/timestamps/$ndb"] = dbnoise(trip2, timestamp2)
                        querydbf["/db/names/$ndb"] = i
                    else
                        nquery += 1
                        querydbf["/query/trips/$nquery"], querydbf["/query/timestamps/$nquery"] = querynoise(trip, timestamp)
                        querydbf["/query/names/$nquery"] = i
                    end
                end
            elseif ndb < dbsize
                if 2min_length <= size(trip, 2) <= 2max_length
                    if do_split
                        ndb += 1
                        trip1, timestamp1, _, _ = uniformsplit(trip, timestamp)
                        querydbf["/db/trips/$ndb"], querydbf["/db/timestamps/$ndb"] = dbnoise(trip1, timestamp1)
                        #querydbf["/db/timestamps/$ndb"] = timestamp
                        querydbf["/db/names/$ndb"] = i
                    else
                        ndb += 1
                        querydbf["/db/trips/$ndb"], querydbf["/db/timestamps/$ndb"] = dbnoise(trip, timestamp)
                        querydbf["/db/names/$ndb"] = i
                    end
                end
            else
                break
            end
        end
        querydbf["/query/num"], querydbf["/db/num"] = nquery, ndb
        close(querydbf)
    end
    nquery, ndb
end


"""
Creating trj.t trj.label for t2vec()
"""
function createTLabel(region::SpatialRegion, querydbfile::String;
                      tfile="trj.t", labelfile="trj.label")
    seq2str(seq) = join(map(string, seq), " ") * "\n"

    querydbf = h5open(querydbfile, "r")
    label = Int[]
    open(tfile, "w") do f
        num_query, num_db = read(querydbf["/query/num"]), read(querydbf["/db/num"])
        for i = 1:num_query+num_db
            location, idx = i <= num_query ? ("query", i) : ("db", i-num_query)
            trip = read(querydbf["/$location/trips/$idx"])
            name = read(querydbf["/$location/names/$idx"])
            seq = trip2seq(region, trip)
            write(f, seq2str(seq))
            push!(label, name)
        end
    end
    writedlm(labelfile, label)
    close(querydbf)
    length(label)
end

function createEDLCSSInput(querydbfile::String; tfile="edlcss.t", labelfile="edlcss.label")
    querydbf = h5open(querydbfile, "r")
    label = Int[]
    open(tfile, "w") do f
        num_query, num_db = read(querydbf["/query/num"]), read(querydbf["/db/num"])
        for i = 1:num_query+num_db
            location, idx = i <= num_query ? ("query", i) : ("db", i-num_query)
            trip = read(querydbf["/$location/trips/$idx"])
            name = read(querydbf["/$location/names/$idx"])
            timestamp = read(querydbf["/$location/timestamps/$idx"])

            tripInMeter = mapslices(v->lonlat2meters(v...), trip, 1)
            toInt = x -> Int(round(x))
            tripInMeter = map(xy->(toInt(first(xy)), toInt(last(xy))), tripInMeter)
            seq = map((xy, t)->(first(xy), last(xy), t), tripInMeter[:], timestamp)
            seq = "[" * join(map(string, seq), ";") * ";]"
            seq = replace(seq, " ", "")
            prefix = "$name 24/11/2000 11:30:41 "
            write(f, prefix * seq * "\n")
            push!(label, name)
        end
    end
    writedlm(labelfile, label)
    close(querydbf)
    length(label)
end


"""
For each trj in query, computing the rank of its twin trj in db.
This function is applicable to vector representations.

query[:, i] (db[:, i]) represent a trajectory.
"""
function ranksearch(query::Matrix{T},
                    queryLabel::Vector{Int},
                    db::Matrix{T},
                    dbLabel::Vector{Int}) where T
    @assert size(query, 2) == length(queryLabel) "unmatched query and label"
    @assert size(db, 2) == length(dbLabel) "unmatched db and label"
    println("Building KDTree...")
    kdtree = KDTree(db)
    dbsize = length(dbLabel)
    function rank(x::Vector{T}, xLabel::Int)
        k = 5
        @assert k < dbsize "The database size is too small"
        while k < dbsize
            idxs, _ = knn(kdtree, x, k, true)
            r = findfirst(t->t==xLabel, dbLabel[idxs])
            r == nothing ? k = 2k : return r
        end
        dbsize
    end
    ranks = Int[]
    for i = 1:length(queryLabel)
        push!(ranks, rank(query[:,i], queryLabel[i]))
    end
    ranks
end

"""
For each trj in query, computing the rank of its twin trj in db.
This function is applicable to sequence representations.

query[i] (db[i]) represents a trajectory.
"""
function ranksearch(query::Vector{Vector{T}},
                    queryLabel::Vector{Int},
                    db::Vector{Vector{T}},
                    dbLabel::Vector{Int},
                    distance) where T<:Real
    @assert length(query) == length(queryLabel) "unmatched query and label"
    @assert length(db) == length(dbLabel) "unmatched db and label"
    function rank(x::Vector{T}, xLabel::Int)
        dists = map(t -> distance(x, t), db)
        idxs = sortperm(dists)
        findfirst(t -> t == xLabel, dbLabel[idxs])
    end
    ranks = Vector{Int}(undef, length(queryLabel))
    for i = 1:length(queryLabel)
        ranks[i] = rank(query[i], queryLabel[i])
    end
    ranks
end

function changeRate(x)
    (x[2:end] - x[1]) / x[1]
end

function knnsearch(query, db)
    nquery = size(query, 2)
    nn = zeros(Int, 50, nquery)
    dist = Euclidean()
    for i = 1:nquery
        #dists = mapslices(v -> euclidean(query[:, i], v), db, 1)
        dists = pairwise(dist, reshape(query[:, i], :, 1), db)
        idxs = sortperm(dists[:])
        nn[:, i] = idxs[1:50]
    end
    nn
end

function topkPrecision(nn1, nn2, k)
    num = size(nn1, 2)
    prec = Vector{Float64}(num)
    for i = 1:num
        prec[i] = length(intersect(nn1[1:k, i], nn2[1:k, i])) / k
    end
    mean(prec)
end

function knnPrecision(query1, db1, query2, db2)
    nn1 = knnsearch(query1, db1)
    nn2 = knnsearch(query2, db2)
    precs = Float64[]
    for k in [20, 30, 40, 50]
        push!(precs, topkPrecision(nn1, nn2, k))
    end
    precs
end

#query, db = rand(10, 20), rand(10, 100)
#nn = knnsearch(query, db)
#topkPrecision(nn, nn, 10)
#knnPrecision(query, db, query, db)

#x = rand(2,3)
#y = rand(2,3)
#colwise(Euclidean(), x, y)

#timestamp = h5open("querydb.h5", "r") do f
#    f["/query/timestamps/1"] |> read
#end
#
#tripInMeter = mapslices(v->lonlat2meters(v...), trip, 1)
#seq = map((xy, t)->(first(xy), last(xy), t), tripInMeter[:], timestamp)
#"[" * join(map(string, seq), ";") * ";]"
#lonlat2meters(trip[1,1], trip[2,1])
#changeRate([0.672, 1.12, 1.93, 2.85])
#
#changeRate([7.13, 16.23, 43.34, 99.025])
#
#changeRate([8.28, 16.58, 33.16, 48.96])
#
#changeRate([35.20, 51.78, 66.37, 76.67])
#
#changeRate([0.975, 1.52, 2.22, 2.65])
#
#changeRate([6.62, 13.5, 25.48, 37.44])
#
#changeRate([4.42, 8.77, 17.25, 25.52])
#
#changeRate([62.16, 69.29, 76.77, 80.75])
