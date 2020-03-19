using ArgParse

include("utils.jl")

args = let s = ArgParseSettings()
    @add_arg_table s begin
        "--datapath"
            arg_type=String
            default="/home/xiucheng/Github/t2vec/data"
    end
    parse_args(s; as_symbols=true)
end

datapath = args[:datapath]

porto2h5("$datapath/porto.csv")
