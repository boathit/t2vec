
import argparse
from train import train
from evaluate import evaluator, t2vec
import torch

## toy_data
## python t2vec.py -data="toy_data" -hidden_size=64 -embedding_size=64 -save_freq=100 -vocab_size=43 -epochs 20
## python t2vec.py -data="toy_data" -hidden_size=64 -embedding_size=64 -save_freq=100 -vocab_size=43 -epochs 20 -criterion_name="KLDIV" -knearestvocabs="toy_data/toy-vocab-dist.h5"

## server
## cell100
## python t2vec.py -vocab_size 18864 -criterion_name "KLDIV" -knearestvocabs "/home/xiucheng/Github/t2vec/data/porto-vocab-dist-cell100.h5"
## python t2vec.py -mode 2 -vocab_size 18864 -checkpoint "/home/xiucheng/Github/t2vec/data/best_model_gen.pt" -prefix "exp1"

parser = argparse.ArgumentParser(description="train.py")

parser.add_argument("-data", default="/home/xiucheng/Github/t2vec/data",
    help="Path to training and validating data")

parser.add_argument("-checkpoint", default="/home/xiucheng/Github/t2vec/data/checkpoint.pt",
    help="The saved checkpoint")

parser.add_argument("-prefix", default="exp", help="Prefix of trjfile")

parser.add_argument("-pretrained_embedding", default=None,
    help="Path to the pretrained word (cell) embedding")

parser.add_argument("-num_layers", type=int, default=3,
    help="Number of layers in the RNN cell")

parser.add_argument("-bidirectional", type=bool, default=True,
    help="True if use bidirectional rnn in encoder")

parser.add_argument("-hidden_size", type=int, default=256,
    help="The hidden state size in the RNN cell")

parser.add_argument("-embedding_size", type=int, default=256,
    help="The word (cell) embedding size")

parser.add_argument("-dropout", type=float, default=0.2,
    help="The dropout probability")

parser.add_argument("-max_grad_norm", type=float, default=5.0,
    help="The maximum gradient norm")

parser.add_argument("-learning_rate", type=float, default=0.001)

parser.add_argument("-batch", type=int, default=128,
    help="The batch size")

parser.add_argument("-generator_batch", type=int, default=32,
    help="""The maximum number of words to generate each time.
    The higher value, the more memory requires.""")

parser.add_argument("-t2vec_batch", type=int, default=256,
    help="""The maximum number of trajs we encode each time in t2vec""")

parser.add_argument("-start_iteration", type=int, default=0)

parser.add_argument("-epochs", type=int, default=15,
    help="The number of training epochs")

parser.add_argument("-print_freq", type=int, default=50,
    help="Print frequency")

parser.add_argument("-save_freq", type=int, default=1000,
    help="Save frequency")

parser.add_argument("-cuda", type=bool, default=True,
    help="True if we use GPU to train the model")

parser.add_argument("-use_discriminative", action="store_true",
    help="Use the discriminative loss if the argument is given")

parser.add_argument("-discriminative_w", type=float, default=0.1,
    help="discriminative loss weight")

parser.add_argument("-criterion_name", default="NLL",
    help="NLL (Negative Log Likelihood) or KLDIV (KL Divergence)")

parser.add_argument("-knearestvocabs", default=None,
    help="""The file of k nearest cells and distances used in KLDIVLoss,
    produced by preprocessing, necessary if KLDIVLoss is used""")

parser.add_argument("-dist_decay_speed", type=float, default=0.8,
    help="""How fast the distance decays in dist2weight, a small value will
    give high weights for cells far away""")

parser.add_argument("-max_num_line", type=int, default=20000000)

parser.add_argument("-max_length", default=200,
    help="The maximum length of the target sequence")

parser.add_argument("-mode", type=int, default=0,
    help="Running mode (0: train, 1:evaluate, 2:t2vec)")

parser.add_argument("-vocab_size", type=int, default=0,
    help="Vocabulary Size")

parser.add_argument("-bucketsize", default=[(20,30),(30,30),(30,50),(50,50),(50,70),(70,70),(70,100),(100,100)],
    help="Bucket size for training")



args = parser.parse_args()

print(args)

## __main__
#args.bucketsize = [(20,30),(30,30),(30,50),(50,50),(50,70),(70,70),(70,100),(100,100)]
#args.bucketsize = [(10, 10), (20, 20), (20, 30)]
#args.vocab_size = 43

if args.mode == 1:
    evaluator(args)
elif args.mode == 2:
    with torch.no_grad():
        t2vec(args)
else:
    train(args)
