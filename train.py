
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from models import EncoderDecoder
from data_utils import DataLoader
import constants, time, os, shutil, logging, h5py

def NLLcriterion(vocab_size):
    "construct NLL criterion"
    weight = torch.ones(vocab_size)
    weight[constants.PAD] = 0
    ## The first dimension is not batch, thus we need
    ## to average over the batch manually
    criterion = nn.NLLLoss(weight, size_average=False)
    return criterion

def KLDIVcriterion(vocab_size):
    "construct KLDIV criterion"
    weight = torch.ones(vocab_size)
    weight[constants.PAD] = 0
    ## The first dimension is not batch, thus we need
    ## to average over the batch manually
    criterion = nn.KLDivLoss(weight, size_average=False)
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
    indices = torch.index_select(V, 0, target)
    ## (batch, k) gather along vocab_size dimension
    outputk = torch.gather(output, 1, indices)
    ## (batch, k) index in vocab_size dimension
    targetk = torch.index_select(D, 0, target)
    return criterion(outputk, targetk)

def KLDIVloss2(output, target, criterion, V, D):
    """
    constructing full target distribution, expensive
    """
    indices = torch.index_select(V, 0, target)
    targetk = torch.index_select(D, 0, target)
    fulltarget = Variable(torch.zeros(output.size())).scatter_(1, indices, targetk)
    ## here: need Variable(fulltarget).cuda() if use gpu
    fulltarget = Variable(fulltarget)
    return criterion(output, fulltarget)

def dist2weight(D, dist_decay_speed=0.8):
    D = D.div(100)
    D = torch.exp(-D * dist_decay_speed)
    s = D.sum(dim=1)
    return D.div(s.expand_as(D))


def batchloss(output, target, generator, lossF, generator_batch):
    """
    One batch loss

    Input:
    output (seq_len, batch, hidden_size): the output of EncoderDecoder
    target (seq_len, batch): target tensor
    generator: map the output of EncoderDecoder into the vocabulary space and do
        log transform
    lossF: loss function
    generator_batch: the maximum number of words to generate each step
    ---
    Output:
    loss
    """
    batch = output.size(1)
    loss = 0
    ## we want to decode target in range [BOS+1:EOS]
    target = target[1:]
    for o, t in zip(output.split(generator_batch),
                    target.split(generator_batch)):
        ## (seq_len, generator_batch, hidden_size) =>
        ## (seq_len*generator_batch, hidden_size)
        o = o.view(-1, o.size(2))
        o = generator(o)
        ## (seq_len*generator_batch,)
        t = t.view(-1)
        loss += lossF(o, t)

    return loss.div(batch)

def init_parameters(model):
    for p in model.parameters():
        p.data.uniform_(-0.1, 0.1)

def savecheckpoint(state, is_best, filename="checkpoint.pt"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'best_model.pt')

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

    total_loss = 0
    for iteration in range(num_iteration):
        input, lengths, target = valData.getbatch()
        input = Variable(input, volatile=True)
        lengths = Variable(lengths, volatile=True)
        target = Variable(target, volatile=True)
        if args.cuda and torch.cuda.is_available():
            input, lengths, target = input.cuda(), lengths.cuda(), target.cuda()
        output = m0(input, lengths, target)
        loss = batchloss(output, target, m1, lossF, args.generator_batch)
        total_loss += loss * output.size(1)
    ## switch back to training mode
    m0.train()
    m1.train()
    return total_loss.data[0] / valData.size


def train(args):
    logging.basicConfig(filename="training.log", level=logging.INFO)

    trainsrc = os.path.join(args.data, "train.src")
    traintrg = os.path.join(args.data, "train.trg")
    trainData = DataLoader(trainsrc, traintrg, args.batch, args.bucketsize)
    print("Reading training data...")
    trainData.load(args.max_num_line)
    print("Allocation: {}".format(trainData.allocation))
    print("Percent: {}".format(trainData.p))

    valsrc = os.path.join(args.data, "val.src")
    valtrg = os.path.join(args.data, "val.trg")
    if os.path.isfile(valsrc) and os.path.isfile(valtrg):
        valData = DataLoader(valsrc, valtrg, args.batch, args.bucketsize, True)
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
        with h5py.File(args.knearestvocabs) as f:
            V, D = f["V"][...], f["D"][...]
            V, D = Variable(torch.LongTensor(V)), Variable(torch.FloatTensor(D))
        D = dist2weight(D, args.dist_decay_speed)
        if args.cuda and torch.cuda.is_available():
            V, D = V.cuda(), D.cuda()
        criterion = KLDIVcriterion(args.vocab_size)
        lossF = lambda o, t: KLDIVloss(o, t, criterion, V, D)

    m0 = EncoderDecoder(args.vocab_size,
                        args.embedding_size,
                        args.hidden_size,
                        args.num_layers,
                        args.dropout,
                        args.bidirectional)
    m1 = nn.Sequential(nn.Linear(args.hidden_size, args.vocab_size),
                       nn.LogSoftmax())
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

    num_iteration = args.epochs * sum(trainData.allocation) // args.batch
    print("Iteration starts at {} "
          "and will end at {}".format(args.start_iteration, num_iteration-1))
    ## training
    for iteration in range(args.start_iteration, num_iteration):
        try:
            input, lengths, target = trainData.getbatch()
            input, lengths, target = Variable(input), Variable(lengths), Variable(target)
            if args.cuda and torch.cuda.is_available():
                input, lengths, target = input.cuda(), lengths.cuda(), target.cuda()

            m0_optimizer.zero_grad()
            m1_optimizer.zero_grad()
            ## forward computation
            output = m0(input, lengths, target)
            loss = batchloss(output, target, m1, lossF, args.generator_batch)
            ## compute the gradients
            loss.backward()
            ## clip the gradients
            clip_grad_norm(m0.parameters(), args.max_grad_norm)
            clip_grad_norm(m1.parameters(), args.max_grad_norm)
            ## one step optimization
            m0_optimizer.step()
            m1_optimizer.step()
            ## average loss for one word
            avg_loss = loss.data[0] / target.size(0)
            if iteration % args.print_freq == 0:
                print("Iteration: {}\tLoss: {}".format(iteration, avg_loss))
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
                }, is_best)
        except KeyboardInterrupt:
            break
