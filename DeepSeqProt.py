##WRITE FILES!!!!!

import os
import time
import torch
import random
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
import argparse
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.metrics.cluster import adjusted_mutual_info_score


## PARAMETERS & MODEL ##
start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#encoder
arch = [2000, 1500, 1000, 500, 1]

#vector quantizer
num_embeddings = 1000
commitment_cost = 10
decay = 0.9

#training
batch_size = 32
learning_rate = 1e-5
max_training_updates = 100000

#command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, help="input fasta file [required]")
parser.add_argument("-o", "--output", type=str, help="output prefix [required]")
parser.add_argument("-t", "--training", type=str, help="either a fasta file used to train the model or pre-trained pytorch model file [required]")
parser.add_argument("-e", "--encoder", type=int, nargs='+', help="list of integers representing the length and width of the encoder architecture [2000, 1500, 1000, 500, 1]")
parser.add_argument("-n", "--num_embeddings", type=int, help="number of discrete embeddings to use in the vector quantized latent space [1000]")
parser.add_argument("-c", "--commitment", type=int, help="commitment cost [10]")
parser.add_argument("-d", "--decay", type=int, help="decay [0.9]")
parser.add_argument("-a", "--num_channels", type=int, help="if a different number of channels are desired for the latent space (for example 2 for plotting purposes) this parameter introduces a simple convolutional layer to transform the number of default channels (20) into -a channels")
parser.add_argument("-b", "--batch", type=int, help="batch size [32]")
parser.add_argument("-l", "--learning", type=int, help="learning rate [1e-5]")
parser.add_argument("-m", "--max", type=int, help="maximum number of training updates [100000]")
args = parser.parse_args()

#required arguments
input = args.input
output_prefix = args.output
train_file = args.training

#update default parameters with optional command line arguments
if args.encoder:
  arch = args.encoder
if args.num_embeddings:
  num_embeddings = args.num_embeddings
if args.commitment:
  commitment_cost = args.commitment
if args.decay:
  decay = args.decay
if args.batch:
  batch_size = args.batch
if args.learning:
  learning_rate = args.learning
if args.max:
  max_training_updates = args.max
if args.num_channels:
  use_conv = True
  num_channels = args.num_channels
else:
  use_conv = False
  
#write log
os.mkdir(output_prefix)
output_file = output_prefix + "/" + output_prefix
log = open(output_file + "_log.txt", "w")
log.write("PARAMETERS\n\n")
log.write("Required Arguments\n")
log.write("input file = " + str(input) + "\n")
log.write("training file = " + str(train_file) + "\n")
log.write("output prefix = " + str(output_prefix) + "\n")
log.write("Encoder\n")
log.write("encoder architecture = " + str(arch) + "\n")
log.write("Vector Quantizer\n")
log.write("number of embeddings = " + str(num_embeddings) + "\n")
log.write("commitment cost = " + str(commitment_cost) + "\n")
log.write("decay = " + str(decay) + "\n")
if args.num_channels:
  log.write("number of channels = " + str(num_channels) + "\n")
log.write("Training\n")
log.write("batch size = " + str(batch_size) + "\n")
log.write("learning rate = " + str(learning_rate) + "\n")
log.write("maximum training updates = " + str(max_training_updates) + "\n\n")
log.close()

#one hot encode protein sequence
def one_hot_seq(seq):
    aa_dict = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
           'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}
    out = []
    #replace 20 aa with integers
    for aa in seq:
        out.append(aa_dict.get(aa, 0))
    #one hot conversion
    enumerate(out)
    for i,v in enumerate(out):
        z = np.zeros(20, dtype=int)
        if v != 0:
            z[v-1] = 1
        out[i] = z
    #convert to array
    out_array = np.asanyarray(out)
    #transpose
    out_array = np.transpose(out_array)
    return(out_array)

#interpolate one hot sequences to standard length
def seq_inter(array, length):
  #linearly spaced vector along original sequence length
  lin = np.linspace(0, 1, len(array[0]))
  #interpolation along first dim
  inter_out = interp1d(lin, array, axis=1)
  #linearly spaced vector along new uniform sequence length
  new_len = np.linspace(0, 1, length)
  return torch.tensor(inter_out(new_len),dtype=torch.float)

#dataset for fasta files, interpolates one-hot sequences to standard length
class fasta_data(Dataset):
  def __init__(self, fasta_file, length):
    #read fasta file
    self.fasta_file = list(SeqIO.parse(fasta_file, "fasta"))
    self.length = length
  def __len__(self):
    return len(self.fasta_file)
  def __getitem__(self, idx):
    #sort fasta header and sequence into dictionary
    if torch.is_tensor(idx):
      idx = idx.tolist()
    ids = self.fasta_file[idx].id
    seqs = seq_inter(one_hot_seq(self.fasta_file[idx].seq), self.length)
    sample = {'id':ids, 'seq':seqs}
    return sample

#encoder
class encoder(nn.Module):
    def __init__(self, arch=arch, *args, **kwargs):
        super().__init__()

        self.blocks = nn.ModuleList([
                      nn.Linear(arch[0], arch[1]),
                      *[nn.Linear(input, output)
                      for (input, output) in zip(arch[1:], arch[2:])]
        ])

        if use_conv:
          self.conv = nn.Conv1d(20,num_channels,1,1)

    def forward(self, x):
        for i, block in enumerate(self.blocks) :
          x = block(x)
        if use_conv:
          x = self.conv(x)
        return x

#vector quantizer
class vector_quantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(vector_quantizer, self).__init__()

        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        #convert inputs from BCL -> BLC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        #flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        #calculate distances
        distances = (torch.sum(flat_input ** 2, dim = 1, keepdim = True)
                    + torch.sum(self._embedding.weight ** 2, dim = 1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        #encoding
        embedding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        embeddings = torch.zeros(embedding_indices.shape[0], self._num_embeddings, device = inputs.device)
        embeddings.scatter_(1, embedding_indices, 1)
        #quantize and unflatten
        quantized = torch.matmul(embeddings, self._embedding.weight).view(input_shape)
        #use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
            (1 - self._decay) * torch.sum(embeddings, 0)
            #laplace smoothing of cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            dw = torch.matmul(embeddings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        #loss
        e_latent_loss = nn.functional.mse_loss(quantized.detach(), inputs)
        q_latent_loss = nn.functional.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        quantized = inputs + (quantized - inputs).detach()
        embedding_usage = len(torch.unique(embedding_indices))        
        #convert quantized from BLC -> BCL 
        return quantized.permute(0,2,1).contiguous(), loss, embedding_usage, embedding_indices

#decoder
class decoder(nn.Module):
    def __init__(self, arch=arch, *args, **kwargs):
        super().__init__()

        self.arch = arch[::-1]

        if use_conv:
          self.deconv = nn.ConvTranspose1d(20,num_channels,1,1)

        self.blocks = nn.ModuleList([
                      nn.Linear(self.arch[0], self.arch[1]),
                      *[nn.Linear(input, output)
                      for (input, output) in zip(self.arch[1:], self.arch[2:])]
        ])

    def forward(self, x):
        if use_conv:
          x = self.deconv(x)
        for i, block in enumerate(self.blocks) :
            x = block(x)
        return x

#model
class model(nn.Module):
    def __init__(self, arch,
                 num_embeddings, embedding_dim,
                 commitment_cost, decay,
                 epsilon=1e-5):
        super(model, self).__init__()

        self._encoder = encoder(arch = arch)

        self._vq = vector_quantizer(num_embeddings = num_embeddings,
                                    embedding_dim = embedding_dim,
                                    commitment_cost = commitment_cost,
                                    decay = decay,
                                    epsilon = epsilon)

        self._decoder = decoder(arch = arch)


    def forward(self, x):
        encoded = self._encoder(x)
        quantized, loss, embedding_usage, embeddings = self._vq(encoded)
        x_recon = self._decoder(quantized)

        return x_recon, loss, embedding_usage, embeddings

## TRAINING ##
def train(train_file):

  #load training data and model
  data = fasta_data(train_file, arch[0])
  training_loader = DataLoader(data, batch_size = batch_size, shuffle = True)
  data_var = 0.032 #average variance per sequence 
  embedding_dim = 20 * arch[-1]
  vae = model(arch, num_embeddings, embedding_dim, commitment_cost, decay)
  vae.to(device)
  optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate, amsgrad=False)

  vae.train()
  
  train_res_loss = []
  train_res_embedding_usage = []

  for i in range(max_training_updates):
      batch = next(iter(training_loader))
      batch_data = batch['seq']
      batch_data = batch_data.to(device)

      optimizer.zero_grad()

      batch_recon, vq_loss, embedding_usage, embeddings = vae(batch_data)
      recon_error = nn.functional.mse_loss(batch_recon, batch_data) / data_var
      loss = recon_error + vq_loss
      loss.backward()

      optimizer.step()

      train_res_loss.append(loss.item())
      train_res_embedding_usage.append(embedding_usage)

      if (i+1) % 100 == 0:
          log = open(output_file + "_log.txt", "a")
          log.write(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)) + "\n")
          log.write('%d iterations' % (i+1) + "\n")
          log.write('loss: %.3f' % np.mean(train_res_loss[-100:]) + "\n")
          log.write('embedding_usage: %.3f' % np.mean(train_res_embedding_usage[-100:])+ "\n\n")
          log.close()

      if (i+1) % 1000 == 0:
          torch.save(vae, output_file + ".pt")

      if (i+1) > 2000:
          if np.mean(train_res_loss[-2000:-1000]) <= np.mean(train_res_loss[-1000:]):
            torch.save(vae, output_file + ".pt")
            return
            
  torch.save(vae, output_file + ".pt")

if os.path.splitext(train_file)[-1] in ['.fasta','.fa','.fna','.faa','.frn','.fnn','.fas','.pep','.cds']:
  log = open(output_file + "_log.txt", "a")
  log.write("BEGIN TRAINING on fasta file " + str(train_file) + " " + time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)) + "\n\n")
  log.close()
  train(train_file)
  
## TESTING ##
def test(fasta, model):
  batch_size = 100

  model = torch.load(model, map_location=device)
  model.eval()

  validation_data = fasta_data(fasta, arch[0])
  validation_loader = DataLoader(validation_data, batch_size = batch_size, shuffle = False)

  output = []
  header = []

  for i, batch in enumerate(validation_loader):

      validation_id = batch['id']

      validation_seqs = batch['seq']
      validation_seqs = validation_seqs.to(device)
      
      vq_output_eval = model._encoder(validation_seqs)
      valid_quantize, loss, embedding_usage, embeddings = model._vq(vq_output_eval)

      encoding = embeddings.detach().cpu().numpy().flatten()
      embeds = valid_quantize.view(len(encoding), -1).detach().cpu().numpy().squeeze()

      output.append(np.column_stack([validation_id, encoding, embeds]))

      if i == 0:
        dims = []
        for i in range(np.shape(embeds)[1]):
          dims.append("Dim_" + str(i))
          header = ['Entry', 'Encoding'] + dims

      if (i+1) % 1000 == 0:
        log = open(output_file + "_log.txt", "a")
        log.write(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)) + "\n")
        log.write("%d sequences processed" % (i+1*batch_size)+ "\n\n")

  return pd.DataFrame(np.concatenate(output), columns=header)

if os.path.splitext(train_file)[-1] in ['.fasta','.fa','.fna','.faa','.frn','.fnn','.fas','.pep','.cds']:
  model_file = output_file + ".pt"
  log = open(output_file + "_log.txt", "a")
  log.write("BEGIN TESTING on model " + str(model_file) + " " + time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)) + "\n\n")
  log.close()
  coords = test(input, model_file)
else:
  log = open(output_file + "_log.txt", "a")
  log.write("BEGIN TESTING on pretrained model " + str(train_file) + " " + time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)) + "\n\n")
  log.close()
  coords = test(input, train_file)

#write output table
coords.to_csv(output_file + "_coordinates.txt", sep='\t', index=False)

#split test fasta by encoding
os.mkdir(output_prefix + "/fastas")
for e in np.unique(coords["Encoding"].tolist()):

    input_seq_iterator = SeqIO.parse(input, "fasta")
    entries = coords[coords["Encoding"] == e]["Entry"].tolist()

    subfasta = [record for record in input_seq_iterator if record.id in entries]
    SeqIO.write(subfasta, output_prefix + "/fastas/subfasta" + str(e) + ".fa", "fasta")