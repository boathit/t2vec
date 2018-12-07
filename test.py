from models import *
from train import *
import numpy as np

vocab_size = 100
embedding_size = 32
hidden_size = 64
num_layers = 2
dropout = 0
batch = 32
seq_len = 100
embedding = nn.Embedding(vocab_size, embedding_size)

## test StackingGRUCell
rnn = StackingGRUCell(embedding_size, hidden_size,
                      num_layers, dropout)
input = Variable(torch.randn(batch, embedding_size))
h0 = Variable(torch.randn(num_layers, batch, hidden_size))
output, hn = rnn(input, h0)

## test GlobalAttention
attention = GlobalAttention(hidden_size)
q = Variable(torch.randn(batch, hidden_size))
H = Variable(torch.randn(batch, seq_len, hidden_size))
c = attention(q, H)

## test Encoder
encoder = Encoder(embedding_size, hidden_size, num_layers, dropout,
                  True, embedding)

input = np.random.randint(0, vocab_size, size=(seq_len, batch))
input = Variable(torch.LongTensor(input))

lengths = [20 for _ in range(batch // 2)] + [15 for _ in range(batch // 2)]
lengths = Variable(torch.LongTensor(lengths).view(1, -1))
hn, output = encoder(input, lengths)


## test Decoder
decoder = Decoder(embedding_size, hidden_size, num_layers, dropout,
                  embedding)

use_attention = False
h0 = Variable(torch.randn(num_layers, batch, hidden_size))
H = Variable(torch.randn(seq_len, batch, hidden_size))
output1, hn = decoder(input, h0, H, use_attention)
output1

## feed word one by one
h = h0
output2 = []
for x in input.split(1):
    o, h = decoder(x, h, H, use_attention)
    output2.append(o.squeeze(0))
output2 = torch.stack(output2)
output2

torch.equal(output1.data, output2.data)
torch.equal(hn.data, h.data)

## test EncoderDecoder
encdecoder = EncoderDecoder(vocab_size, embedding_size,
    hidden_size, num_layers, dropout)
output = encdecoder(input, lengths, input)
output

torch.equal(encdecoder.encoder.embedding.weight.data,
            encdecoder.decoder.embedding.weight.data)

## --- copy from data_utils.py

A = [np.r_[2, 3, 4], np.r_[2], np.r_[2, 1], np.r_[1, 2, 3, 4, 5]]
B = [np.r_[2], np.r_[1], np.r_[1, 2, 0], np.r_[2, 3, 4]]

pad_arrays(A)
pad_arrays(B)
pad_arrays_pair(A, B)
np.concatenate(([constants.BOS], A[1], [constants.EOS]))

srcfile, trgfile = "data/src", "data/trg"
data = DataLoader(srcfile, trgfile, 10, True)
data.load(1000)

data.p
input, lengths, target = data.getbatch()
data.start
np.concatenate((data.srcdata[2][0:5], data.srcdata[2][5:10]))

srcfile = "data/train.src"
data = DataOrderScaner(srcfile, 10)
data.load(13)
data.srcdata
src, lengths, invp = data.getbatch()
src
lengths
src.t()[invp]

## ---- copy from train.py

## test code here:
vocab_size = 43
batch = 64
hidden_size = 64
embedding_size = 64
num_layers = 1
criterion_name = "nll"
criterion = nllcriterion(vocab_size)
train_srcfile, train_trgfile = "data/src", "data/trg"
trainData = DataLoader(train_srcfile, train_trgfile, batch)

encoderdecoder = EncoderDecoder(vocab_size=vocab_size,
                                embedding_size=embedding_size,
                                hidden_size=hidden_size, num_layers=num_layers,
                                dropout=0.1, bidirectional=True)
generator = nn.Sequential(nn.Linear(hidden_size, vocab_size),
                          nn.LogSoftmax())

encoderdecoder_optimizer = torch.optim.Adam(encoderdecoder.parameters(),
                                            lr=0.001)
generator_optimizer = torch.optim.Adam(generator.parameters(),
                                       lr=0.001)

trainData.load(1000)

## iteration here
input, lengths, target = trainData.getbatch()
input, lengths, target = Variable(input), Variable(lengths), Variable(target)
output = encoderdecoder(input, lengths, target)
## --- test feed forward
#output = output.view(-1, output.size(2))
#output = generator(output)
## --- test batchloss()
#loss = batchloss(output, target, generator, criterion, criterion_name,
#                 generator_batch=32)
#loss, input.size(0), target.size(0)

## --- copy from evaluate.py

checkpoint = torch.load("checkpoint.pt")
encoderdecoder.load_state_dict(checkpoint["encoderdecoder"])
generator.load_state_dict(checkpoint["generator"])

src = [9, 11, 14]
trg = evaluate(src, (encoderdecoder, generator), 20)
trg

class FakeArgs():
    def __init__(self, vocab_size=43,
                       embedding_size=64,
                       hidden_size=64,
                       num_layers=1,
                       dropout=0.1,
                       bidirectional=True,
                       checkpoint="checkpoint.pt",
                       data="toy_data",
                       batch=64,
                       bucketsize=[(10, 10), (20, 20), (20, 30)],
                       cuda=True,
                       generator_batch=32,
                       criterion_name="NLL",
                       t2vec_batch=128,
                       knearestvocabs="toy_data/toy-vocab-dist.h5",
                       dist_decay_speed=0.8):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.checkpoint = checkpoint
        self.data = data
        self.batch = batch
        self.bucketsize = bucketsize
        self.cuda = cuda
        self.generator_batch = generator_batch
        self.criterion_name = criterion_name
        self.t2vec_batch = t2vec_batch
        self.knearestvocabs = knearestvocabs
        self.dist_decay_speed = dist_decay_speed
args = FakeArgs()

## --- test validate()

valsrc = os.path.join(args.data, "val.src")
valtrg = os.path.join(args.data, "val.trg")
args.batch = 3
valData = DataLoader(valsrc, valtrg, args.batch, args.bucketsize, True)
valData.load()
encoderdecoder = EncoderDecoder(args.vocab_size, args.embedding_size,
                                args.hidden_size, args.num_layers,
                                args.dropout, args.bidirectional)
generator = nn.Sequential(nn.Linear(args.hidden_size, args.vocab_size),
                          nn.LogSoftmax())
criterion_name = "NLL"
criterion = nllcriterion(args.vocab_size)



import h5py
with h5py.File("toy_data/toy-vocab-dist.h5", "r") as f:
    V, D = f["V"][...], f["D"][...]
    V, D = Variable(torch.LongTensor(V)), Variable(torch.FloatTensor(D))



V = [[0, 0, 0],
     [1, 1, 1],
     [2, 2, 2],
     [3, 3, 3],
     [4, 5, 6],
     [5, 4, 6],
     [6, 5, 7],
     [7, 6, 8],
     [8, 7, 9],
     [9, 8, 34],
     [10, 9, 11],
     [11, 10, 12],
     [12, 11, 13],
     [13, 12, 14],
     [14, 13, 15],
     [15, 14, 16],
     [16, 15, 17],
     [17, 16, 18],
     [18, 17, 19],
     [19, 18, 20],
     [20, 19, 21],
     [21, 20, 22],
     [22, 21, 23],
     [23, 22, 24],
     [24, 23, 25],
     [25, 24, 26],
     [26, 25, 27],
     [27, 26, 28],
     [28, 42, 29],
     [29, 28, 30],
     [30, 29, 31],
     [31, 30, 32],
     [32, 31, 33],
     [33, 32, 31],
     [34, 9, 35],
     [35, 34, 36],
     [36, 35, 37],
     [37, 36, 38],
     [38, 37, 39],
     [39, 38, 40],
     [40, 39, 41],
     [41, 40, 42],
     [42, 41, 28]]
D = [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0],
     [0, 0, 0],
     [0, 100, 200],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 200, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100],
     [0, 100, 100]]

V = np.array(V, dtype=np.int64)
D = np.array(D, dtype=np.float64)

with h5py.File("toy_data/toy-vocab-dist.h5", "w") as f:
    f["V"], f["D"] = V, D

with h5py.File("toy_data/toy-vocab-dist.h5", "r") as f:
    V, D = f["V"][...], f["D"][...]


## --- test lossF
cd ..

from models import *
from train import *
class FakeArgs():
    def __init__(self, vocab_size=43,
                       embedding_size=64,
                       hidden_size=64,
                       num_layers=1,
                       dropout=0.1,
                       bidirectional=True,
                       checkpoint="checkpoint.pt",
                       data="toy_data",
                       batch=64,
                       bucketsize=[(10, 10), (20, 20), (20, 30)],
                       cuda=True,
                       generator_batch=32,
                       criterion_name="NLL",
                       t2vec_batch=128,
                       knearestvocabs="toy_data/toy-vocab-dist.h5",
                       dist_decay_speed=0.8):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.checkpoint = checkpoint
        self.data = data
        self.batch = batch
        self.bucketsize = bucketsize
        self.cuda = cuda
        self.generator_batch = generator_batch
        self.criterion_name = criterion_name
        self.t2vec_batch = t2vec_batch
        self.knearestvocabs = knearestvocabs
        self.dist_decay_speed = dist_decay_speed
args = FakeArgs()

train_srcfile = os.path.join(args.data, "train.src")
train_trgfile = os.path.join(args.data, "train.trg")
trainData = DataLoader(train_srcfile, train_trgfile, args.batch, args.bucketsize)

encoderdecoder = EncoderDecoder(args.vocab_size, args.embedding_size,
                                args.hidden_size, args.num_layers,
                                args.dropout, args.bidirectional)
generator = nn.Sequential(nn.Linear(args.hidden_size, args.vocab_size),
                          nn.LogSoftmax())
trainData.load(1000)

input, lengths, target = trainData.getbatch()
input, lengths, target = Variable(input), Variable(lengths), Variable(target)
output = encoderdecoder(input, lengths, target)
#output = output.view(-1, output.size(2))
#output = generator(output)

args.criterion_name = "KLDIV"
if args.criterion_name == "NLL":
    criterion = NLLcriterion(args.vocab_size)
    lossF = lambda o, t: criterion(o, t)
else:
    assert os.path.isfile(args.knearestvocabs),\
        "{} does not exist".format(args.knearestvocabs)
    print("Loading vocab distance file {}...".format(args.knearestvocabs))
    with h5py.File(args.knearestvocabs) as f:
        V, D = f["V"][...], f["D"][...]
        V, D = Variable(torch.LongTensor(V)), Variable(torch.FloatTensor(D))
    D = dist2weight(D, args.dist_decay_speed)
    criterion = KLDIVcriterion(args.vocab_size)
    lossF = lambda o, t: KLDIVloss(o, t, criterion, V, D)

batchloss(output, target, generator, lossF, args.generator_batch)

o = output.view(-1, output.size(2))
o = generator(o)
## (seq_len*generator_batch,)
t = target[1:].view(-1)

D.sum(dim=1)
