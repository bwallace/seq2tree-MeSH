import time
import random
import math 

import numpy as np

import torch 
from torch import nn 
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

import load_data
import model 
import preprocess 


USE_CUDA = False

def prep_data():
    input_texts, mesh_outputs = load_data.assemble_pairs()
    abstract_p = preprocess.Preprocessor()
    
    # preprocess and encode texts (inputs)
    abstract_p.preprocess(input_texts)
    X = abstract_p.encode_texts(input_texts)

    labels_p = preprocess.Preprocessor(vocab_size=None, split_char=".", normalize=False)
    labels_p.preprocess(mesh_outputs)
    Y = labels_p.encode_texts(mesh_outputs)

    return (input_texts, abstract_p, mesh_outputs, labels_p, list(zip(X,Y)))

def train(input_variable, target_variable, encoder, decoder, 
            encoder_optimizer, decoder_optimizer, criterion, 
            teacher_forcing_ratio=0.5, clip=5.0):
    '''
    train on a single pair. 
    '''

    # zero out gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # reset loss

    # get size of input and target sentences
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    # run (input) words through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
    
    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([[preprocess.SOS_token]]))
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    # choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:
        
        # teacher forcing: Use the ground-truth target as the next input
        for di in range(target_length):
           
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(
                             decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di] # next target is next input

    else:
        # without teacher forcing: use network's own prediction as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            
            # get most likely word index (highest value) from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            
            decoder_input = Variable(torch.LongTensor([[ni]])) # Chosen word is next input
            if USE_CUDA: 
                decoder_input = decoder_input.cuda()

            # Stop at end of sentence (not necessary when using known targets)
            if ni == preprocess.EOS_token: 
                break

    # backprop
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.data[0] / target_length

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def torch_variable_from_indices(indices):
    var = Variable(torch.LongTensor(indices).view(-1, 1))
    if USE_CUDA: 
        var = var.cuda()
    return var

def torch_variables_from_pair(pair):
    input_variable = torch_variable_from_indices(pair[0])
    target_variable = torch_variable_from_indices(pair[1])
    return (input_variable, target_variable)

def run():
    input_texts, abstract_p, mesh_outputs, labels_p, pairs = prep_data()

    attn_model = 'general'
    hidden_size = 500
    n_layers = 1
    dropout_p = 0.05

    # initialize models
    encoder = model.Encoder(abstract_p.get_dim(), hidden_size)
    decoder = model.AttnDecoderRNN(attn_model, hidden_size, labels_p.get_dim(), n_layers, dropout_p=dropout_p)

    # move models to GPU
    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()

    # Initialize optimizers and criterion
    learning_rate = 0.0001
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    # training details
    n_epochs = 50000
    plot_every = 10
    print_every = 10

    # keep track of time elapsed and running averages
    start = time.time()
    plot_losses = []
    print_loss_total = 0 # Reset every print_every
    plot_loss_total = 0 # Reset every plot_every

    # finally...
    for epoch in range(1, n_epochs + 1):
        
        training_pair = torch_variables_from_pair(random.choice(pairs))
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        # run the train function
        loss = train(input_variable, target_variable, encoder, decoder, 
                        encoder_optimizer, decoder_optimizer, criterion)

        # keep track of loss
        print_loss_total += loss
        plot_loss_total += loss

        if epoch == 0: continue

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
            print(print_summary)

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


