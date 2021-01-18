
using CSV, DataFrames, HDF5

"""
This function read the orginal porto taxi csv file and save all trips into
hdf5 file, each trip will be a 2xn matrix where n is the length of the trip.
The first row is longitude and second row is latitude.

The output hdf5 file will be used to construct vocabulary by `makeVocab!()` in
`SpatialRegionTools.jl`.
"""
function porto2h5(csvfile::String)
    df = CSV.File(csvfile) |> DataFrame
    df = df[df.MISSING_DATA .== false, :]
    sort!(df, [:TIMESTAMP])
    println("Processing $(size(df, 1)) trips...")
    ## writing in numeric matrix with hdf5
    h5open("../data/porto.h5", "w") do f
        num, num_incompleted = 0, 0
        for trip in df.POLYLINE
            try
                trip = Meta.parse(trip) |> eval
            catch e
                num_incompleted += 1
                continue
            end
            tripLength = length(trip)
            tripLength == 0 && continue
            trip = hcat(trip...)
            num += 1
            f["/trips/$num"] = trip
            f["/timestamps/$num"] = collect(0:tripLength-1) * 15.0
            num % 100_000 == 0 && println("$num")
        end
        attributes(f)["num"] = num
        println("Incompleted trip: $num_incompleted.\nSaved $num trips.")
    end
end

function porto2standardcsv(portocsv::String)
    df = CSV.read(portocsv)
    df = df[df.MISSING_DATA .== "False", :]
    #sort!(df, [:TIMESTAMP])
    trips = DataFrame[]
    for (i, row) in enumerate(eachrow(df))
        gps = Meta.parse(row.POLYLINE) |> eval
        tripLength = length(gps)
        tripLength == 0 && continue
        gps = hcat(gps...)
        trip = DataFrame(tripid=fill(row.TRIP_ID, tripLength),
                         timestamps=row.TIMESTAMP .+ collect(0:tripLength-1) * 15,
                         lon=gps[1, :],
                         lat=gps[2, :])
        push!(trips, trip)
        i % 100_000 == 0 && println("Processed $i rows.")
    end
    #CSV.write("../data/porto-trips.csv", trips, header=[:tripid, :timestamps, :lon, :lat])
    trips
end

"""
Distorting a trip using Gaussian noise
"""
function distort(trip::Matrix{Float64}, rate::Float64; radius=50.0)
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
    noisetrip
end

"""
Accepting one trip and producing its 10 different noise rate distorted variants
"""
function distort(trip::Matrix{Float64}, nsplit::Int)
    noisetrips = Matrix{Float64}[]

    for rate in 0:0.1:0.9
        noisetrip = distort(trip, rate)
        push!(noisetrips, noisetrip)
    end
    noisetrips
end

"""
Downsampling one trip, rate is dropping rate
"""
function downsampling(trip::Matrix{Float64}, rate::Float64)
    keep_idx = Int[1]
    for i = 2:size(trip, 2)-1
        rand() > rate && push!(keep_idx, i)
    end
    push!(keep_idx, size(trip, 2))
    copy(trip[:, keep_idx])
end

"""
Accepting one trip and producing its 9 different lowsampling rate variants
"""
function downsampling(trip::Matrix{Float64}, nsplit::Int)
    noisetrips = Matrix{Float64}[]
    dropping_rates = 0:0.1:0.8
    for rate in dropping_rates
        noisetrip = downsampling(trip, rate)
        push!(noisetrips, noisetrip)
    end
    noisetrips
end

"""
First downsampling and then distorting the trip, producing its 20 different variants
"""
function downsamplingDistort(trip::Matrix, nsplit::Int)
    noisetrips = Matrix{Float64}[]
    dropping_rates = [0, 0.2, 0.4, 0.5, 0.6]
    distorting_rates = [0, 0.2, 0.4, 0.6]
    for dropping_rate in dropping_rates
        noisetrip1 = downsampling(trip, dropping_rate)
        for distorting_rate in distorting_rates
            noisetrip2 = distort(noisetrip1, distorting_rate)
            push!(noisetrips, noisetrip2)
        end
    end
    noisetrips
end


"""
longitude, latitude to Web Mercator coordinate
"""
function lonlat2meters(lon::Float64, lat::Float64)
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = sin(north)
    semimajoraxis * east, 3189068.5 * log((1 + t) / (1 - t))
end

function meters2lonlat(x::Float64, y::Float64)
    semimajoraxis = 6378137.0
    lon = x / semimajoraxis / 0.017453292519943295
    t = exp(y / 3189068.5)
    lat = asin((t - 1) / (t + 1)) / 0.017453292519943295
    lon, lat
end

#x, y = lonlat2meters(-8.7351,40.95)
#meters2lonlat(x, y)

#h5open("/tmp/test.h5", "w") do f
#    f["/trips/1"] = rand(10,2)
#    f["/trips/2"] = rand(30,2)
#    attributes(f)["num"] = 2
#    attributes(f)["/trips/1"] = 10
#    attributes(f)["/trips/2"] = 11
#end
#
#h5open("/tmp/test.h5", "r") do f
#    read(f["/trips/1"])
#    attributes(f)["/trips/1"] |> read
#end

#porto2h5("/Users/fineday/Downloads/train.csv")


#tic()
#tripLengths = Int[]
#h5open("preprocessing/porto.h5", "r") do f
#    num = read(attributes(f)["num"])
#    for i = 1:num
#        trip = read(f["/trips/$i"])
#        push!(tripLengths, size(trip, 2))
#        #i % 100_000 == 0 && println(i)
#    end
#end
#toc()
#h5open("porto.h5", "r") do fread
#    num = attributes(fread)["num"] |> read
#    h5open("/tmp/porto.h5", "w") do fwrite
#        attributes(fwrite)["num"] = num
#        for i = 1:num
#            trip = read(fread["/trips/$i"])
#            fwrite["/trips/$i"] = trip
#            tripLength = size(trip, 2)
#            fwrite["/timestamps/$i"] = collect(0:tripLength-1) * 15.0
#            i % 100_000 == 0 && println(i)
#        end
#    end
#end
#toc()

#filter(x-> x >= 20 && x <= 100, tripLengths) |> length
#sum(tripLengths)
#
#sum(tripLengths[1:800_000])

#h5open("porto.h5", "r") do fread
#    h5open("/tmp/porto.h5", "w") do fwrite
#        num = attributes(fread)["num"] |> read
#        attributes(fwrite)["num"] = num
#        for i = 1:num
#            fwrite["/trips/$i"] = fread["/trips/$i"] |> read |> transpose
#            i % 100_000 == 0 && println(i)
#        end
#    end
#end

#tripLengths = Int[]
#h5open("../preprocessing/porto.h5", "r") do f
#    num = read(attributes(f)["num"])
#    for i = 1:num
#        trip = read(f["/trips/$i"])
#        tripLength = size(trip, 2)
#        push!(tripLengths, tripLength)
#        #tripLength >= 50 && push!(tripLengths, tripLength)
#        #i % 100_000 == 0 && println(i)
#    end
#end
#t = filter(x -> 30<=x, tripLengths)
#filter(x -> 60 <= x, t[800000:end])
#sum(tripLengths)
#mean(tripLengths)
#tripLengths
#maximum(tripLengths) #3881
#129_535_632 / 1527348
#trip = rand(40:50, 2, 40)*1.0
#distort(trip, 10)
#downsampling(trip, 10)
#sdownsamplingDistort(trip, 5)

#lonlat2meters(-8.735152,40.953673)
