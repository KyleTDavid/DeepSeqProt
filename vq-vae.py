#KL divergence for latent space loss?

import resnet
import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
from math import ceil
from functools import partial

##parameters##

#basic blocks
kernel_size = 3
dilation = 1
conv = partial(resnet.conv_auto, kernel_size=kernel_size, dilation=dilation, bias=False)
trans = partial(resnet.trans_auto, kernel_size=kernel_size, dilation=dilation, bias=False)

#encoder
in_channels = 20
e_arch = [128, 2]
e_depth = [6, 1]

#vector quantizer        
num_embeddings = 12
embedding_dim = 2
commitment_cost = 0.25
decay = 0.99

#decoder
d_arch = [2, 128]
d_depth = [1, 6]

#dynamic sampling to ensure one 1 datum per channel at quantizer
batch_size = 32
learning_rate = 1e-3
num_training_updates = 3000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_file = "saccharomyces_cerevisiae_proteome.fa"
output_file = "test"

log = open(output_file + "_log.txt", "w")
log.write("PARAMETERS\n\n")
log.write("Basic Block\n")
log.write("kernel size = " + str(kernel_size) + "\n")
log.write("dilation factor = " + str(dilation) + "\n\n")
log.write("Encoder\n")
log.write("encoder architecture = " + str(e_arch) + "\n")
log.write("encoder depths = " + str(e_depth) + "\n\n")
log.write("Vector Quantizer\n")
log.write("number of embeddings = " + str(num_embeddings) + "\n")
log.write("embedding dimenstions = " + str(embedding_dim) + "\n")
log.write("commitment cost = " + str(commitment_cost) + "\n")
log.write("decay = " + str(decay) + "\n\n")
log.write("Decoder\n")
log.write("decoder architecture = " + str(d_arch) + "\n")
log.write("decoder depths = " + str(d_depth) + "\n\n")
log.write("Learning\n")
log.write("batch size = " + str(batch_size) + "\n")
log.write("learning rate = " + str(learning_rate) + "\n")
log.write("number of training updates = " + str(num_training_updates) + "\n\n")
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

#dataset for fasta files
class fasta_data(Dataset):
  def __init__(self, fasta_file):
    #read fasta file and (don't) sort by length
    self.fasta_file = list(SeqIO.parse(fasta_file, "fasta"))
    '''self.fasta_file.sort(key=lambda r: len(r))'''
    #get average sequence length for variance estimation and interpolation
    seq_lengths = [len(i) for i in self.fasta_file]
    self.avgseqlen = sum(seq_lengths) / len(seq_lengths)
  def __len__(self):
    return len(self.fasta_file)
  def __getitem__(self, idx):
    #sort fasta header and sequence into dictionary
    if torch.is_tensor(idx):
      idx = idx.tolist()
    ids = self.fasta_file[idx].id
    seqs = seq_inter(one_hot_seq(self.fasta_file[idx].seq), int(self.avgseqlen))
    sample = {'id':ids, 'seq':seqs}
    return sample

#encoder architecture
class encoder(nn.Module):
    def __init__(self, conv, in_channels=20, block_arch=[1024, 256, 64, 3], deepths=[2, 2, 2, 2], 
                 block = resnet.basic_block, *args, **kwargs):
        super().__init__()
        self.block_arch = block_arch
        
        self.gate = nn.Sequential(
            nn.Conv1d(in_channels, self.block_arch[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(self.block_arch[0]),
            nn.ReLU()
            #nn.Conv1d(in_channels, self.block_arch[0], kernel_size=7, stride=2, padding=3, bias=False),
            #nn.BatchNorm1d(self.block_arch[0]),
            #nn.ReLU(),
            #nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        self.in_out_block_sizes = list(zip(block_arch, block_arch[1:]))
        self.blocks = nn.ModuleList([ 
            resnet.layer(block_arch[0], block_arch[0], n=deepths[0], 
                        block=block, conv=conv, *args, **kwargs),
            *[resnet.layer(in_channels * block.expansion, 
                          out_channels, n=n, conv=conv,
                          block=block, *args, **kwargs) 
            for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:-1])],
            resnet.layer(block_arch[-2], block_arch[-1], n=deepths[-1], 
                        block=block, conv=conv, sampling=515)       
        ])
        
        
    def forward(self, x):
        x = self.gate(x)
        for i, block in enumerate(self.blocks) :
            x = block(x)
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
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device = inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        #quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        #use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
            (1 - self._decay) * torch.sum(encodings, 0)
            
            #laplace smoothing of cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        #loss
        e_latent_loss = nn.functional.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim = 0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        #convert quantized from BLC -> BCL
        return loss, quantized.permute(0,2,1).contiguous(), perplexity, encodings, self._embedding.weight.data, encoding_indices

#decoder architecture
class decoder(nn.Module):
    def __init__(self, conv, block_arch=[3, 64, 256, 1024], deepths=[2, 2, 2, 2], 
                 block=resnet.basic_block, *args, **kwargs):
        super().__init__()

        self.block_arch = block_arch
        
        self.linit = nn.Linear(1, 2)

        self.in_out_block_sizes = list(zip(block_arch[1:], block_arch[2:]))

        self.blocks = nn.ModuleList([ 
            resnet.layer(block_arch[0], block_arch[1], n=deepths[0], 
                        block=block, conv=conv, sampling=82),
            *[resnet.layer(in_channels * block.expansion, 
                          out_channels, n=n, conv=conv,
                          block=block, *args, **kwargs) 
            for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]       
        ])
        
        self.gate = nn.Sequential(
            nn.Conv1d(embedding_dim, 20, kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm1d(20),
            nn.ReLU(),
            #nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Linear(1, int(data.avgseqlen))
        )
        
    def forward(self, x):
        #x = self.linit(x)
        x = self.gate(x)
        return x

#model
class model(nn.Module):
    def __init__(self, conv, in_channels, e_arch,
                 e_depth, num_embeddings, embedding_dim,
                 commitment_cost, decay, trans, 
                 d_arch, d_depth, sampling, epsilon=1e-5):
        super(model, self).__init__()
        
        self._encoder = encoder(conv = conv,
                                    in_channels = in_channels,
                                    block_arch = e_arch,
                                    deepths = e_depth,
                                    sampling = sampling)
        
        self._vq = vector_quantizer(num_embeddings = num_embeddings,
                                    embedding_dim = embedding_dim,
                                    commitment_cost = commitment_cost,
                                    decay = decay,
                                    epsilon = epsilon)
        
        self._decoder = decoder(conv = trans,
                                    block_arch = d_arch,
                                    deepths = d_depth,
                                    sampling = sampling)
        
    def forward(self, x):
        encoded = self._encoder(x)
        loss, quantized, perplexity, _, embeddings, encodings = self._vq(encoded)
        x_recon = self._decoder(quantized)
            
        return loss, x_recon, perplexity, quantized, encoded, embeddings, encodings #encoded is redundant can be removed

#load data and specify model
data = fasta_data(input_file)

#variance (this might be wrong)
data_var = np.mean([torch.var(batch['seq']) for batch in data])

training_loader = DataLoader(data, batch_size = batch_size, shuffle = True)

sampling = ceil(data.avgseqlen**(1/len(e_arch))) - 1 #dynamic sampling
#sampling = 1

vae = model(conv, in_channels, e_arch, e_depth, num_embeddings,
              embedding_dim, commitment_cost, decay, trans, d_arch,
              d_depth, sampling).to(device)


optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate, amsgrad=False)

#training
vae.train()
train_res_loss = []
train_res_recon_error = []
train_res_perplexity = []

start_time = time.time()
log = open(output_file + "_log.txt", "a")
log.write("BEGIN TRAINING " + time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)) + "\n\n")
log.close()

#list to store embeddings to observe learning
embeddings_list = list()

for i in range(num_training_updates):
    batch = next(iter(training_loader))
    batch_data = batch['seq']
    batch_data = batch_data.to(device)
    
    optimizer.zero_grad()

    vq_loss, batch_recon, perplexity, quantized, encoded, embeddings, encodings = vae(batch_data)
    recon_error = nn.functional.mse_loss(batch_recon, batch_data) / data_var
    loss = recon_error + vq_loss
    loss.backward()

    optimizer.step()
    
    train_res_loss.append(loss.item())
    train_res_recon_error.append(recon_error.item())
    train_res_perplexity.append(perplexity.item())

    if (i+1) % 100 == 0:
        log = open(output_file + "_log.txt", "a")
        log.write(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)) + "\n")
        log.write('%d iterations' % (i+1) + "\n")
        log.write('loss: %.3f' % np.mean(train_res_loss[-100:]) + "\n")
        log.write('recon_error: %.3f' % np.mean(train_res_recon_error[-100:])+ "\n")
        log.write('perplexity: %.3f' % np.mean(train_res_perplexity[-100:])+ "\n\n")
        log.close()
        torch.save(vae, output_file + ".pt")
                       
        embeddings_list.append(pd.DataFrame(embeddings).astype("float"))

log = open(output_file + "_log.txt", "a")
log.write("done! " + time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)) + "\n\n")
log.close()


#get encoding and embedding coordinates for test dataset

batch_size = 1
def gen_embed(fasta, model):

  log = open(output_file + "_log.txt", "a")
  log.write("BEGIN TESTING " + time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)) + "\n\n")
  log.close()

  model = torch.load(model, map_location=device)
  model.eval()

  validation_data = fasta_data(fasta)
  validation_loader = DataLoader(validation_data, batch_size = batch_size, shuffle = False)

  panda_output = pd.DataFrame()

  for i, batch in enumerate(validation_loader):

      validation_ids = validation_data[i]['id']

      validation_seqs = batch['seq']
      validation_seqs = validation_seqs.to(device)
      vq_output_eval = model._encoder(validation_seqs)
      _, valid_quantize, _, e, embeddings, encodings = model._vq(vq_output_eval)

      encoding = encodings.detach().cpu().numpy().flatten()
      embeds = valid_quantize.view(batch_size, embedding_dim).detach().cpu().numpy().flatten()

      panda = pd.DataFrame({'ID': validation_ids, 'Encoding': encoding,
                            'x': embeds[0], 'y': embeds[1]})
      
      panda_output = panda_output.append(panda)  

      if (i+1) % 1000 == 0:
        log = open(output_file + "_log.txt", "a")
        log.write("%d sequences processed" % (i+1*batch_size)+ "\n")
    
  return panda_output
  
model_file = output_file + ".pt"

out_directory = output_file + "_output"
os.mkdir(out_directory)

test_embeddings = gen_embed("non_singleton_fams.fa", model_file)
final_embeddings = embeddings_list[-1]
final_embeddings['Count'] = test_embeddings['Encoding'].value_counts()
final_embeddings['Encoding'] = final_embeddings.index

import matplotlib.pyplot as plt
plt.scatter(final_embeddings[0], final_embeddings[1], s=final_embeddings['Count'], alpha=0.5)

for i, txt in enumerate(final_embeddings['Encoding']):
    plt.annotate(txt, (final_embeddings[0][i], final_embeddings[1][i]))

plt.savefig(out_directory + "/Encodings.png")

os.mkdir(out_directory + "/learnings")
i=0
for e in embeddings_list:
  plt.figure()
  
  plt.scatter(e[0], e[1])

  plt.savefig("/content/" + out_directory + "/learnings/fig"+str(i)+".png")
  i+=100
  
  os.mkdir(out_directory + "/clusters")

for e in np.unique(test_embeddings["Encoding"].tolist()):
    input_seq_iterator = SeqIO.parse(input_file, "fasta")
    encoding = test_embeddings[test_embeddings["Encoding"] == e]["ID"].tolist()
    subfasta = [record for record in input_seq_iterator if record.id in encoding]
    SeqIO.write(subfasta, "/content/" + out_directory + "/clusters/subfasta" + str(e) + ".fa", "fasta")
