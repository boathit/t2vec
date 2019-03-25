
This repository contains the code used in our **ICDE-18** paper [Deep Representation Learning for Trajectory Similarity Computation](http://www.ntu.edu.sg/home/lixiucheng/pdfs/icde18-t2vec.pdf).

## Requirements

* Ubuntu OS
* [Julia 0.6.4](https://julialang.org/downloads/oldreleases.html) (**Julia 0.7+ is untested**)
* Python >= 3.5 (Anaconda3 is recommended)
* PyTorch 0.1.2 (You may want to use `virtualenv` to avoid being conflict with your current version)

Please refer to the source code to install all required packages in Julia and Python.

In Julia, you can install a package in REPL like

```
               _
   _       _ _(_)_     |  A fresh approach to technical computing
  (_)     | (_) (_)    |  Documentation: https://docs.julialang.org
   _ _   _| |_  __ _   |  Type "?help" for help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 0.6.4 (2018-07-09 19:09 UTC)
 _/ |\__'_|_|_|\__'_|  |  Official http://julialang.org/ release
|__/                   |  x86_64-pc-linux-gnu


julia> Pkg.add("StatsBase")
```

## Preprocessing

The preprocessing step will generate all data required in the training stage. 

Two parameters can be set at this step, `cellsize` in `preprocessing/preprocess.jl` and denoising `radius` in `preprocessing/utils.jl:disort()`, you can leave them as their default values which are the ones used in our paper.

```shell
$ curl http://archive.ics.uci.edu/ml/machine-learning-databases/00339/train.csv.zip -o data/porto.csv.zip
$ unzip data/porto.csv.zip
$ mv train.csv data/porto.csv
$ cd preprocessing
$ julia preprocess.jl
```

The generated files for training are saved in `data/`.

## Training

```shell
$ python t2vec.py -data data -vocab_size 18866 -criterion_name "KLDIV" -knearestvocabs "data/porto-vocab-dist-cell100.h5"
```

The training produces two model `checkpoint.pt` and `best_model.pt`, `checkpoint.pt` contains the latest trained model and `best_model.pt` saves the model which has the best performance on the validation data. You can find our saved `best_model.pt` [here](https://drive.google.com/open?id=1uxZUmvFHhpY8tOXvCDHuEd7KFOTYp109).

In our original experiment, the model was trained with a Tesla K40 GPU about 14 hours so you can just terminate the training after 14 hours if you use a GPU that is as good as or better than K40, the above two models will be saved automatically.


## Encoding

### Create test files

```bash
cd experiment

julia createTest.jl

head -5 trj.t # the first 5 trajectories
head -5 trj.label # trajectory ids
```

It will produce two files `trj.t` and `trj.label`. Each row of `trj.t` (`trj.label`) is a token representation of the orginal trajectory (trajectory ID).

### Encode trajectories into vectors

```shell
$ python t2vec.py -data experiment -vocab_size 18866 -checkpoint "best_model.pt" -mode 2
```

It will encode the trajectories in file `experiment/trj.t` into vectors which will be saved into file `experiment/trj.h5`.

### Vector representation

In our experiment we train a three-layers model and the last layer outputs are used as the trajectory representations, see the code in `experiment/experiment.jl`:

```julia
vecs = h5open(joinpath("", "trj.h5"), "r") do f
    read(f["layer3"])
end

vecs[i] # the vector representation of i-th trajectory
```

## Reference

```
@inproceedings{DBLP:conf/icde/LiZCJW18,
  author    = {Xiucheng Li and
               Kaiqi Zhao and
               Gao Cong and
               Christian S. Jensen and
               Wei Wei},
  title     = {Deep Representation Learning for Trajectory Similarity Computation},
  booktitle = {34th {IEEE} International Conference on Data Engineering, {ICDE} 2018,
               Paris, France, April 16-19, 2018},
  pages     = {617--628},
  year      = {2018},
  crossref  = {DBLP:conf/icde/2018},
  url       = {https://doi.org/10.1109/ICDE.2018.00062},
  doi       = {10.1109/ICDE.2018.00062},
  timestamp = {Tue, 20 Nov 2018 10:20:00 +0100},
  biburl    = {https://dblp.org/rec/bib/conf/icde/LiZCJW18},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
