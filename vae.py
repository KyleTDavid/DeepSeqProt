import os
import time
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from math import ceil
from functools import partial
import matplotlib.pyplot as plt
import pickle
import resnet


## PARAMETERS & MODEL ##
start_time = time.time()

#basic blocks
kernel_size = 1
dilation = 1
conv = partial(resnet.conv_auto, kernel_size=kernel_size, dilation=dilation, bias=False)
trans = partial(resnet.trans_auto, kernel_size=kernel_size, dilation=dilation, bias=False)

#encoder
in_channels = 20
lin_arc = [2000, 1500, 1000, 500, 1]
res_arc = [128, 32, 8, 2]

#vector quantizer        
num_embeddings = 100
commitment_cost = 0.1
decay = 0.9

#training
batch_size = 32
learning_rate = 1e-5
max_training_updates = 100000

#inputs and outputs
test_file = "scerevisiae_test.fa"
train_file = "scerevisiae_train.fa"
output_prefix = "test"

#write log
os.mkdir(output_prefix)
os.mkdir(output_prefix + "/models")
output_file = output_prefix + "/" + output_prefix
log = open(output_file + "_log.txt", "w")
log.write("PARAMETERS\n\n")
log.write("Encoder\n")
log.write("encoder architecture = " + str(lin_arc) + "\n")
log.write("Vector Quantizer\n")
log.write("number of embeddings = " + str(num_embeddings) + "\n")
log.write("commitment cost = " + str(commitment_cost) + "\n")
log.write("decay = " + str(decay) + "\n\n")
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
    self.fasta_file = fasta_file
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
    def __init__(self, lin_arc = lin_arc, res_arc = res_arc, depths =([1]*len(res_arc)),
                 block = resnet.basic_block, *args, **kwargs):
        super().__init__()
        
        self.linear = nn.ModuleList([
                      nn.Linear(lin_arc[0], lin_arc[1]),
                      *[nn.Linear(input, output)
                      for (input, output) in zip(lin_arc[1:], lin_arc[2:])]
        ])

        self.in_out_block_sizes = list(zip(res_arc, res_arc[1:]))
        self.conv = nn.ModuleList([ 
            resnet.layer(in_channels, res_arc[0], n=depths[0], 
                        block=block, conv=conv,
                         sampling=1, *args, **kwargs),
            *[resnet.layer(in_channels * block.expansion, 
                          out_channels, n=n, conv=conv,
                          sampling=1,
                          block=block, *args, **kwargs) 
            for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])],   
        ])

    def forward(self, x):
        #print(x.shape)

        for i, res in enumerate(self.conv) :            
            x = res(x)
            #print(x.shape)

        for i, fc in enumerate(self.linear) :            
            x = fc(x)
            #print(x.shape)
        
        mu, logvar = torch.chunk(x, 2, dim=1)
        return mu, logvar

#bottleneck
#ugly
class bottleneck(nn.Module):
      def __init__(self, *args, **kwargs):
        super().__init__()
        
      def forward(self, mu, logvar):
        std = torch.exp(logvar / 2).to(device)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return z.to(device)
        
        #convert quantized from BLC -> BCL #clean this
        #return quantized.permute(0,2,1).contiguous(), loss, perplexity, self._embedding.weight.data, indices_list

#decoder 
class decoder(nn.Module):
    def __init__(self, lin_arc = lin_arc, res_arc = res_arc, depths = ([1]*(len(res_arc)+1)),
                 block = resnet.basic_block, *args, **kwargs):
        super().__init__()
        
        self.lin_arc = lin_arc[::-1]
        self.linear = nn.ModuleList([
                      nn.Linear(self.lin_arc[0], self.lin_arc[1]),
                      *[nn.Linear(input, output)
                      for (input, output) in zip(self.lin_arc[1:], self.lin_arc[2:])]
        ])

        self.res_arc = [int(res_arc[-1] / 2)] + res_arc[::-1]
        self.in_out_block_sizes = list(zip(self.res_arc, self.res_arc[1:]))
        self.conv = nn.ModuleList([ 
            resnet.layer(self.res_arc[0], self.res_arc[0], n=depths[0], 
                        block=block, conv=conv,
                        sampling=1, *args, **kwargs),
            *[resnet.layer(in_channels * block.expansion, 
                          out_channels, n=n, conv=conv,
                          block=block,
                          sampling=1, *args, **kwargs) 
            for (in_channels, out_channels), n in zip(self.in_out_block_sizes, depths[1:])],
        ])
        
        self.gate = nn.Sequential(
        nn.Conv1d(self.res_arc[-1], 20, kernel_size=1, stride=1, padding=0, bias=False), 
        nn.ReLU(),
        )

    def forward(self, x):
        #print(x.shape)

        for i, fc in enumerate(self.linear) :            
            x = fc(x)
            #print(x.shape)

        for i, res in enumerate(self.conv) :            
            x = res(x)
            #print(x.shape)

        x = self.gate(x)

        return x

#model
class model(nn.Module):
    def __init__(self):
      
        super(model, self).__init__()

        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        
        self._encoder = encoder()
        
        self._bottleneck = bottleneck()
        
        self._decoder = decoder()
        
        
    def forward(self, x):
        mu, logvar = self._encoder(x)
        bottlenecked = self._bottleneck(mu, logvar)
        x_recon = self._decoder(bottlenecked)

        #print(mu.shape)
        #print(bottlenecked.shape)
            
        return x_recon, mu, logvar, self.log_scale, bottlenecked
        
## LOAD MODEL ##
vae = model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae.to(device)

optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate, amsgrad=False)

training_fasta = list(SeqIO.parse(test_file, "fasta"))

def gaussian_likelihood(x_hat, logscale, x):
      scale = torch.exp(logscale)
      mean = x_hat
      dist = torch.distributions.Normal(mean, scale)

      # measure prob of seeing image under p(x|z)
      log_pxz = dist.log_prob(x)
      return log_pxz.sum(dim=(2, 1))

def kl_divergence(z, mu, std):
      # --------------------------
      # Monte carlo KL divergence
      # --------------------------
      # 1. define the first two probabilities (in this case Normal for both)
      p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
      q = torch.distributions.Normal(mu, std)

      # 2. get the probabilities from the equation
      log_qzx = q.log_prob(z)
      log_pz = p.log_prob(z)

      # kl
      kl = (log_qzx - log_pz)
      kl = kl.sum(-1)
      return kl
      
#def train(input):

vae.train()

train_res_loss = []
train_res_recon_error = []
train_res_perplexity = []

log = open(output_file + "_log.txt", "a")
log.write("BEGIN TRAINING " + time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)) + "\n\n")
log.close()

data = fasta_data(training_fasta, lin_arc[0])
training_loader = DataLoader(data, batch_size = batch_size, shuffle = True)

for i in range(max_training_updates):
    batch = next(iter(training_loader))
    batch_data = batch['seq']
    batch_data = batch_data.to(device)
    
    optimizer.zero_grad()

    x_recon, mu, logvar, ls, bn = vae(batch_data)

    recon_loss = gaussian_likelihood(x_recon, vae.log_scale, batch_data)
    #print(recon_loss.shape)

    kl = kl_divergence(bn, mu, torch.exp(logvar / 2))
    #print(kl.shape)

    elbo = (kl - recon_loss)

    elbo = elbo.mean()

    elbo.backward()

    optimizer.step()
    
    train_res_loss.append(elbo.item())

    if (i+1) % 100 == 0:
        log = open(output_file + "_log.txt", "a")
        log.write(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)) + "\n")
        log.write('%d iterations' % (i+1) + "\n")
        log.write('loss: %.3f' % np.mean(train_res_loss[-100:]) + "\n")
        log.close()
        
torch.save(vae, output_file + ".pt")
f = plt.figure(figsize=(16,8))
ax = f.add_subplot(1,2,1)
ax.plot(train_res_loss)
ax.set_yscale('log')
ax.set_title('NMSE.')
ax.set_xlabel('iteration')
f.savefig(output_file + '.png')

#get encoding for each sequence in test fasta
def gen_embed(fasta, model, batch_size):
  
  log = open(output_file + "_log.txt", "w")
  log.write("BEGIN TESTING " + time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)) + "\n\n")
  log.close()

  #model = torch.load(model, map_location=device)
  model.eval()

  validation_data = fasta_data(fasta, lin_arc[0])
  validation_loader = DataLoader(validation_data, batch_size = batch_size, shuffle = False)

  output = []
  header = ['Entry', 'mu', 'mu2', 'logvar', 'logvar2', 'out', 'out2']

  for i, batch in enumerate(validation_loader):

      validation_id = batch['id']

      validation_seqs = batch['seq']
      validation_seqs = validation_seqs.to(device)

      test_mu, test_logvar = model._encoder(validation_seqs)
      bottlenecked = model._bottleneck(test_mu, test_logvar)
      
      
      output.append(np.column_stack([validation_id, test_mu.detach().cpu().numpy().squeeze(), test_logvar.detach().cpu().numpy().squeeze(), bottlenecked.detach().cpu().numpy().squeeze()]))
    

  #return pd.DataFrame(output, columns=header)
  df = pd.DataFrame(np.concatenate(output), columns=header)
  #df = df.sort_values(by='Encoding').reset_index(drop=True)

  out = {}

  return df

out = gen_embed(list(SeqIO.parse('scerevisiae_test.fa', "fasta")), vae, 100)

out.to_csv(output_file + "_df.txt", sep='\t', header=True, index=False)

