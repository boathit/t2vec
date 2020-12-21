
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from models import EncoderDecoder
from data_utils import DataLoader
import constants, time, os, shutil, logging, h5py



def NLLcriterion(vocab_size):
    "construct NLL criterion"
    weight = torch.ones(vocab_size)
    weight[constants.PAD] = 0
    ## The first dimension is not batch, thus we need
    ## to average over the batch manually
    #criterion = nn.NLLLoss(weight, size_average=False)
    criterion = nn.NLLLoss(weight, reduction='sum')
    return criterion

def KLDIVcriterion(vocab_size):
    "construct KLDIV criterion"
    # weight = torch.ones(vocab_size)
    # weight[constants.PAD] = 0
    # ## The first dimension is not batch, thus we need
    # ## to average over the batch manually
    # criterion = nn.KLDivLoss(weight, size_average=False)
    criterion = nn.KLDivLoss(reduction='sum')
    return criterion

def KLDIVloss(output, target, criterion, V, D):
    """
    output (batch, vocab_size)
    target (batch,)
    criterion (nn.KLDIVLoss)
    V (vocab_size, k)
    D (vocab_size, k)
    """
    ## (batch, k) index in vocab_size dimension
    ## k-nearest neighbors for target
    indices = torch.index_select(V, 0, target)
    ## (batch, k) gather along vocab_size dimension
    outputk = torch.gather(output, 1, indices)
    ## (batch, k) index in vocab_size dimension
    targetk = torch.index_select(D, 0, target)
    return criterion(outputk, targetk)

def KLDIVloss2(output, target, criterion, V, D):
    """
    constructing full target distribution, expensive!
    """
    indices = torch.index_select(V, 0, target)
    targetk = torch.index_select(D, 0, target)
    fulltarget = torch.zeros(output.size()).scatter_(1, indices, targetk)
    ## here: need Variable(fulltarget).cuda() if use gpu
    fulltarget = fulltarget.cuda()
    return criterion(output, fulltarget)

def dist2weight(D, dist_decay_speed=0.8):
    D = D.div(100)
    D = torch.exp(-D * dist_decay_speed)
    s = D.sum(dim=1, keepdim=True)
    D = D / s
    ## The PAD should not contribute to the decoding loss
    D[constants.PAD, :] = 0.0
    return D


def genLoss(gendata, m0, m1, lossF, args):
    """
    One batch loss

    Input:
    gendata: a named tuple contains
        gendata.src (seq_len1, batch): input tensor
        gendata.lengths (1, batch): lengths of source sequences
        gendata.trg (seq_len2, batch): target tensor.
    m0: map input to output.
    m1: map the output of EncoderDecoder into the vocabulary space and do
        log transform.
    lossF: loss function.
    ---
    Output:
    loss
    """
    input, lengths, target = gendata.src, gendata.lengths, gendata.trg
    if args.cuda and torch.cuda.is_available():
        input, lengths, target = input.cuda(), lengths.cuda(), target.cuda()
    ## (seq_len2, batch, hidden_size)
    output = m0(input, lengths, target)

    batch = output.size(1)
    loss = 0
    ## we want to decode target in range [BOS+1:EOS]
    target = target[1:]
    for o, t in zip(output.split(args.generator_batch),
                    target.split(args.generator_batch)):
        ## (seq_len, generator_batch, hidden_size) =>
        ## (seq_len*generator_batch, hidden_size)
        o = o.view(-1, o.size(2))
        o = m1(o)
        ## (seq_len*generator_batch,)
        t = t.view(-1)
        loss += lossF(o, t)

    return loss.div(batch)

def disLoss(a, p, n, m0, triplet_loss, args):
    """
    a (named tuple): anchor data
    p (named tuple): positive data
    n (named tuple): negative data
    """
    a_src, a_lengths, a_invp = a.src, a.lengths, a.invp
    p_src, p_lengths, p_invp = p.src, p.lengths, p.invp
    n_src, n_lengths, n_invp = n.src, n.lengths, n.invp
    if args.cuda and torch.cuda.is_available():
        a_src, a_lengths, a_invp = a_src.cuda(), a_lengths.cuda(), a_invp.cuda()
        p_src, p_lengths, p_invp = p_src.cuda(), p_lengths.cuda(), p_invp.cuda()
        n_src, n_lengths, n_invp = n_src.cuda(), n_lengths.cuda(), n_invp.cuda()
    ## (num_layers * num_directions, batch, hidden_size)
    a_h, _ = m0.encoder(a_src, a_lengths)
    p_h, _ = m0.encoder(p_src, p_lengths)
    n_h, _ = m0.encoder(n_src, n_lengths)
    ## (num_layers, batch, hidden_size * num_directions)
    a_h = m0.encoder_hn2decoder_h0(a_h)
    p_h = m0.encoder_hn2decoder_h0(p_h)
    n_h = m0.encoder_hn2decoder_h0(n_h)
    ## take the last layer as representations (batch, hidden_size * num_directions)
    a_h, p_h, n_h = a_h[-1], p_h[-1], n_h[-1]

    return triplet_loss(a_h[a_invp], p_h[p_invp], n_h[n_invp])



def init_parameters(model):
    for p in model.parameters():
        p.data.uniform_(-0.1, 0.1)

def savecheckpoint(state, is_best, args):
    torch.save(state, args.checkpoint)
    if is_best:
        shutil.copyfile(args.checkpoint, os.path.join(args.data, 'best_model.pt'))

def validate(valData, model, lossF, args):
    """
    valData (DataLoader)
    """
    m0, m1 = model
    ## switch to evaluation mode
    m0.eval()
    m1.eval()

    num_iteration = valData.size // args.batch
    if valData.size % args.batch > 0: num_iteration += 1

    total_genloss = 0
    for iteration in range(num_iteration):
        gendata = valData.getbatch_generative()
        with torch.no_grad():
            genloss = genLoss(gendata, m0, m1, lossF, args)
            total_genloss += genloss.item() * gendata.trg.size(1)
    ## switch back to training mode
    m0.train()
    m1.train()
    return total_genloss / valData.size


def train(args):
    logging.basicConfig(filename=os.path.join(args.data, "training.log"), level=logging.INFO)

    trainsrc = os.path.join(args.data, "train.src")
    traintrg = os.path.join(args.data, "train.trg")
    trainmta = os.path.join(args.data, "train.mta")
    trainData = DataLoader(trainsrc, traintrg, trainmta, args.batch, args.bucketsize)
    print("Reading training data...")
    trainData.load(args.max_num_line)
    print("Allocation: {}".format(trainData.allocation))
    print("Percent: {}".format(trainData.p))

    valsrc = os.path.join(args.data, "val.src")
    valtrg = os.path.join(args.data, "val.trg")
    valmta = os.path.join(args.data, "val.mta")
    if os.path.isfile(valsrc) and os.path.isfile(valtrg):
        valData = DataLoader(valsrc, valtrg, valmta, args.batch, args.bucketsize, True)
        print("Reading validation data...")
        valData.load()
        assert valData.size > 0, "Validation data size must be greater than 0"
        print("Loaded validation data size {}".format(valData.size))
    else:
        print("No validation data found, training without validating...")

    ## create criterion, model, optimizer
    if args.criterion_name == "NLL":
        criterion = NLLcriterion(args.vocab_size)
        lossF = lambda o, t: criterion(o, t)
    else:
        assert os.path.isfile(args.knearestvocabs),\
            "{} does not exist".format(args.knearestvocabs)
        print("Loading vocab distance file {}...".format(args.knearestvocabs))
        with h5py.File(args.knearestvocabs, "r") as f:
            V, D = f["V"][...], f["D"][...]
            V, D = torch.LongTensor(V), torch.FloatTensor(D)
        D = dist2weight(D, args.dist_decay_speed)
        if args.cuda and torch.cuda.is_available():
            V, D = V.cuda(), D.cuda()
        criterion = KLDIVcriterion(args.vocab_size)
        lossF = lambda o, t: KLDIVloss(o, t, criterion, V, D)
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    m0 = EncoderDecoder(args.vocab_size,
                        args.embedding_size,
                        args.hidden_size,
                        args.num_layers,
                        args.dropout,
                        args.bidirectional)
    m1 = nn.Sequential(nn.Linear(args.hidden_size, args.vocab_size),
                       nn.LogSoftmax(dim=1))
    if args.cuda and torch.cuda.is_available():
        print("=> training with GPU")
        m0.cuda()
        m1.cuda()
        criterion.cuda()
        #m0 = nn.DataParallel(m0, dim=1)
    else:
        print("=> training with CPU")

    m0_optimizer = torch.optim.Adam(m0.parameters(), lr=args.learning_rate)
    m1_optimizer = torch.optim.Adam(m1.parameters(), lr=args.learning_rate)

    ## load model state and optmizer state
    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        logging.info("Restore training @ {}".format(time.ctime()))
        checkpoint = torch.load(args.checkpoint)
        args.start_iteration = checkpoint["iteration"]
        best_prec_loss = checkpoint["best_prec_loss"]
        m0.load_state_dict(checkpoint["m0"])
        m1.load_state_dict(checkpoint["m1"])
        m0_optimizer.load_state_dict(checkpoint["m0_optimizer"])
        m1_optimizer.load_state_dict(checkpoint["m1_optimizer"])
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))
        logging.info("Start training @ {}".format(time.ctime()))
        best_prec_loss = float('inf')
        #print("=> initializing the parameters...")
        #init_parameters(m0)
        #init_parameters(m1)
        ## here: load pretrained wrod (cell) embedding

    num_iteration = 67000*128 // args.batch
    print("Iteration starts at {} "
          "and will end at {}".format(args.start_iteration, num_iteration-1))
    ## training
    for iteration in range(args.start_iteration, num_iteration):
        try:
            m0_optimizer.zero_grad()
            m1_optimizer.zero_grad()
            ## generative loss
            gendata = trainData.getbatch_generative()
            genloss = genLoss(gendata, m0, m1, lossF, args)
            ## discriminative loss
            disloss_cross, disloss_inner = 0, 0
            if args.use_discriminative and iteration % 10 == 0:
                a, p, n = trainData.getbatch_discriminative_cross()
                disloss_cross = disLoss(a, p, n, m0, triplet_loss, args)
                a, p, n = trainData.getbatch_discriminative_inner()
                disloss_inner = disLoss(a, p, n, m0, triplet_loss, args)
            loss = genloss + args.discriminative_w * (disloss_cross + disloss_inner)
            ## compute the gradients
            loss.backward()
            ## clip the gradients
            clip_grad_norm_(m0.parameters(), args.max_grad_norm)
            clip_grad_norm_(m1.parameters(), args.max_grad_norm)
            ## one step optimization
            m0_optimizer.step()
            m1_optimizer.step()
            ## average loss for one word
            avg_genloss = genloss.item() / gendata.trg.size(0)
            if iteration % args.print_freq == 0:
                print("Iteration: {0:}\tGenerative Loss: {1:.3f}\t"\
                      "Discriminative Cross Loss: {2:.3f}\tDiscriminative Inner Loss: {3:.3f}"\
                      .format(iteration, avg_genloss, disloss_cross, disloss_inner))
            if iteration % args.save_freq == 0 and iteration > 0:
                prec_loss = validate(valData, (m0, m1), lossF, args)
                if prec_loss < best_prec_loss:
                    best_prec_loss = prec_loss
                    logging.info("Best model with loss {} at iteration {} @ {}"\
                                 .format(best_prec_loss, iteration, time.ctime()))
                    is_best = True
                else:
                    is_best = False
                print("Saving the model at iteration {} validation loss {}"\
                      .format(iteration, prec_loss))
                savecheckpoint({
                    "iteration": iteration,
                    "best_prec_loss": best_prec_loss,
                    "m0": m0.state_dict(),
                    "m1": m1.state_dict(),
                    "m0_optimizer": m0_optimizer.state_dict(),
                    "m1_optimizer": m1_optimizer.state_dict()
                }, is_best, args)
        except KeyboardInterrupt:
            break
