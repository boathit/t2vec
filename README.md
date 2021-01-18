
This repository contains the code used in our **ICDE-18** paper [Deep Representation Learning for Trajectory Similarity Computation](https://xiucheng.org/pdfs/icde18-t2vec.pdf).

## Requirements

* Ubuntu OS
* [Julia 1.5+](https://julialang.org/downloads/) (tested on 1.5)
* Python >= 3.8 (Anaconda3 is recommended)
* PyTorch 1.0+ (tested on 1.0, 1.3, 1.5, 1.7)

Please refer to the source code to install all required packages in Julia and Python.

You can install all packages involved for Julia by running,

```shell
$ julia pkg-install.jl
```


## Preprocessing

The preprocessing step will generate all data required in the training stage.

1. For the Porto dataset, you can do as follows.
    ```shell
    $ curl http://archive.ics.uci.edu/ml/machine-learning-databases/00339/train.csv.zip -o data/porto.csv.zip
    $ unzip data/porto.csv.zip
    $ mv train.csv data/porto.csv
    $ cd preprocessing
    $ julia porto2h5.jl
    $ julia preprocess.jl
    ```

2. If you want to work on another city, you are supposed to provide the expected hdf5 input `t2vec/data/cityname.h5` as well as set proper hyperparameters in `t2vec/hyper-parameters.json`. The expected hdf5 input requires the following format,

   ```julia
   attributes(f)["num"] = number of trajectories

   f["/trips/i"] = matrix (2xn)
   f["/timestamps/i"] = vector (n,)
   ```

   where `attributes(f)["num"]` stores the number of trajectories in total; `f["/trips/i"]` is the gps matrix for i-th trajectory, the first row is the longitude sequence and the second row is the latitude sequence, `f["/timestamps/i"]` is the corresponding timestamp sequence. Please refer to [`porto2h5`](https://github.com/boathit/t2vec/blob/master/preprocessing/utils.jl#L12) to see how to generate it.



The generated files for training are saved in `t2vec/data/`.

## Training

```shell
$ python t2vec.py -vocab_size 18864 -criterion_name "KLDIV" -knearestvocabs "data/porto-vocab-dist-cell100.h5"
```

where 18866 is the output of last stage.

The training produces two model `checkpoint.pt` and `best_model.pt`, `checkpoint.pt` contains the latest trained model and `best_model.pt` saves the model which has the best performance on the validation data. You can find our saved `best_model.pt` [here](https://drive.google.com/open?id=1uxZUmvFHhpY8tOXvCDHuEd7KFOTYp109).

In our original experiment, the model was trained with a Tesla K40 GPU about 14 hours so you can just terminate the training after 14 hours if you use a GPU that is as good as or better than K40, the above two models will be saved automatically.


## Encoding and Experiments

### Vector representation

In our experiments we train a three-layers model and the last layer outputs are used as the trajectory representations:

```julia
vecs = h5open(joinpath("", "trj.h5"), "r") do f
    read(f["layer3"])
end

vecs[i] # the vector representation of i-th trajectory
```

### Experiments

Please refer to the [jupyter-notebook](https://github.com/boathit/t2vec/blob/master/experiment/t2vec.ipynb) for the experiments once you obtain the trained model (install [Ijulia](https://github.com/JuliaLang/IJulia.jl) for the Jupyter notebook first).

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
