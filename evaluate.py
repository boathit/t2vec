
import torch
import torch.nn as nn
from models import EncoderDecoder
from data_utils import DataOrderScaner
import os, h5py
import constants

def evaluate(src, model, max_length):
    """
    evaluate one source sequence
    """
    m0, m1 = model
    length = len(src)
    src = torch.LongTensor(src)
    ## (seq_len, batch)
    src = src.view(-1, 1)
    length = torch.LongTensor([[length]])

    encoder_hn, H = m0.encoder(src, length)
    h = m0.encoder_hn2decoder_h0(encoder_hn)
    ## running the decoder step by step with BOS as input
    input = torch.LongTensor([[constants.BOS]])
    trg = []
    for _ in range(max_length):
        ## `h` is updated for next iteration
        o, h = m0.decoder(input, h, H)
        o = o.view(-1, o.size(2)) ## => (1, hidden_size)
        o = m1(o) ## => (1, vocab_size)
        ## the most likely word
        _, word_id = o.data.topk(1)
        word_id = word_id[0][0]
        if word_id == constants.EOS:
            break
        trg.append(word_id)
        ## update `input` for next iteration
        input = torch.LongTensor([[word_id]])
    return trg

#checkpoint = torch.load("checkpoint.pt")
#m0.load_state_dict(checkpoint["m0"])
#m1.load_state_dict(checkpoint["m1"])
#
#src = [9, 11, 14]
#trg = evaluate(src, (m0, m1), 20)
#trg

def evaluator(args):
    """
    do evaluation interactively
    """
    m0 = EncoderDecoder(args.vocab_size, args.embedding_size,
                        args.hidden_size, args.num_layers,
                        args.dropout, args.bidirectional)
    m1 = nn.Sequential(nn.Linear(args.hidden_size, args.vocab_size),
                       nn.LogSoftmax())
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        m0.load_state_dict(checkpoint["m0"])
        m1.load_state_dict(checkpoint["m1"])
        while True:
            try:
                print("> ", end="")
                src = input()
                src = [int(x) for x in src.split()]
                trg = evaluate(src, (m0, m1), args.max_length)
                print(" ".join(map(str, trg)))
            except KeyboardInterrupt:
                break
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))

def t2vec(args):
    "read source sequences from trj.t and write the tensor into file trj.h5"
    m0 = EncoderDecoder(args.vocab_size, args.embedding_size,
                        args.hidden_size, args.num_layers,
                        args.dropout, args.bidirectional)
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        m0.load_state_dict(checkpoint["m0"])
        if torch.cuda.is_available():
            m0.cuda()
        m0.eval()
        vecs = []
        scaner = DataOrderScaner(os.path.join(args.data, "{}-trj.t".format(args.prefix)), args.t2vec_batch)
        scaner.load()
        i = 0
        while True:
            if i % 100 == 0:
                print("{}: Encoding {} trjs...".format(i, args.t2vec_batch))
            i = i + 1
            src, lengths, invp = scaner.getbatch()
            if src is None: break
            if torch.cuda.is_available():
                src, lengths, invp = src.cuda(), lengths.cuda(), invp.cuda()
            h, _ = m0.encoder(src, lengths)
            ## (num_layers, batch, hidden_size * num_directions)
            h = m0.encoder_hn2decoder_h0(h)
            ## (batch, num_layers, hidden_size * num_directions)
            h = h.transpose(0, 1).contiguous()
            ## (batch, *)
            #h = h.view(h.size(0), -1)
            vecs.append(h[invp].cpu().data)
        ## (num_seqs, num_layers, hidden_size * num_directions)
        vecs = torch.cat(vecs)
        ## (num_layers, num_seqs, hidden_size * num_directions)
        vecs = vecs.transpose(0, 1).contiguous()
        path = os.path.join(args.data, "{}-trj.h5".format(args.prefix))
        print("=> saving vectors into {}".format(path))
        with h5py.File(path, "w") as f:
            for i in range(m0.num_layers):
                f["layer"+str(i+1)] = vecs[i].squeeze(0).numpy()
        #torch.save(vecs.data, path)
        #return vecs.data
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))

#args = FakeArgs()
#args.t2vec_batch = 128
#args.num_layers = 2
#args.hidden_size = 64
#vecs = t2vec(args)
#vecs
