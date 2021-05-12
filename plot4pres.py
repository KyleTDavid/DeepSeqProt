import resnet
import pickle
import os
import time
import torch
import random

import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from functools import partial

import plotly.express as px
from plotly.graph_objs import *
import plotly.graph_objects as go

## PARAMETERS & MODEL ##
start_time = time.time()

#vector quantized or just regular variational autoencoder or just regular autoencoder?
model_type = 'vae'

#basic blocks
kernel_size = 1
dilation = 1
conv = partial(resnet.conv_auto, kernel_size=kernel_size, dilation=dilation, bias=False)
trans = partial(resnet.trans_auto, kernel_size=kernel_size, dilation=dilation, bias=False)

#encoder
in_channels = 20
res_arc = [12, 4]
lin_arc = [2000,  1]

#vector quantizer
embedding_dim = res_arc[-1] * lin_arc[-1]
num_embeddings = 100
commitment_cost = 0.1
decay = 0.9

#training
batch_size = 32
learning_rate = 1e-3
max_training_updates = 100

#inputs and outputs
test_file = "data/scerevisiae/scerevisiae_test.fa"
train_file = "data/scerevisiae/scerevisiae_train.fa"
output_prefix = "test"

#write log
os.mkdir(output_prefix)
output_file = output_prefix + "/" + output_prefix
log = open(output_file + "_log.txt", "w")
log.write("PARAMETERS\n\n")
log.write("Encoder\n")
log.write("linear architecture = " + str(lin_arc) + "\n")
log.write("convolutional architecture = " + str(res_arc) + "\n\n")
if model_type == 'vq':
  log.write("Vector Quantizer\n")
  log.write("number of embeddings = " + str(num_embeddings) + "\n")
  log.write("commitment cost = " + str(commitment_cost) + "\n")
  log.write("decay = " + str(decay) + "\n\n")
log.write("Training\n")
log.write("batch size = " + str(batch_size) + "\n")
log.write("learning rate = " + str(learning_rate) + "\n")
log.write("maximum training updates = " + str(max_training_updates) + "\n\n")
log.close()

os.mkdir(output_prefix + "/models")
os.mkdir(output_prefix + "/figs")


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
  new_len = np.linspace(0, 1, lin_arc[0])
  return torch.tensor(inter_out(new_len),dtype=torch.float)

#truncate sequences or pad with zeroes to standard length
def seq_pad(array, length):
  if len(array[0]) > length:
    array = array[0:, 0:length]
  if len(array[0]) < length:
    pad_length = length - len(array[0])
    array = np.pad(array, ((0,0),(0,pad_length)), 'constant', constant_values=0)
  return torch.tensor(array,dtype=torch.float)

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
    def __init__(self, lin_arc = lin_arc, res_arc = res_arc,
                 block = resnet.basic_block, *args, **kwargs):
        super().__init__()

        #convolutional layers
        self.conv = nn.ModuleList([ 
            resnet.layer(in_channels, res_arc[0], 
                        block=block, conv=conv,
                         sampling=1, *args, **kwargs),
            *[resnet.layer(in_channels * block.expansion, 
                          out_channels, conv=conv,
                          sampling=1,
                          block=block, *args, **kwargs) 
            for (in_channels, out_channels) in zip(res_arc[0:], res_arc[1:])]   
        ])
        
        #linear layers
        self.linear = nn.ModuleList([
                      nn.Linear(lin_arc[0], lin_arc[1]),
                      *[nn.Linear(input, output)
                      for (input, output) in zip(lin_arc[1:], lin_arc[2:])]
        ])

        #self._sm = nn.Sigmoid()

    def forward(self, x):
          
        for i, res in enumerate(self.conv) :            
            x = res(x)
            #print("Encoder Convolutional Layer%d" % i)
            #print(x.shape)
            #print("")

        for i, fc in enumerate(self.linear) :            
            x = fc(x)
            #print("Encoder Linear Layer%d" % i)
            #print(x.shape)
            #print("")
                    
        #return self._sm(x)
        return x

#decoder 
class decoder(nn.Module):
    def __init__(self, lin_arc = lin_arc, res_arc = res_arc,
                 block = resnet.basic_block, *args, **kwargs):
        super().__init__()
        
        #linear layers
        self.lin_arc = lin_arc[::-1]
        self.linear = nn.ModuleList([
                      nn.Linear(self.lin_arc[0], self.lin_arc[1]),
                      *[nn.Linear(input, output)
                      for (input, output) in zip(self.lin_arc[1:], self.lin_arc[2:])]
        ])

        #convolutional layers
        self.res_arc = res_arc[::-1] + [in_channels]
        self.conv = nn.ModuleList( 
            [resnet.layer(in_channels * block.expansion, 
                          out_channels, conv=conv,
                          sampling=1,
                          block=block, *args, **kwargs) 
            for (in_channels, out_channels) in zip(self.res_arc[0:], self.res_arc[1:])]
            + [resnet.layer(self.res_arc[-1], in_channels,
                        block=block, conv=conv,
                        sampling=1, *args, **kwargs)]  
        )
        
    def forward(self, x):
        #print(x.shape)

        for i, fc in enumerate(self.linear) :            
            x = fc(x)
            #print("Decoder Linear Layer%d" % i)
            #print(x.shape)
            #print("")

        for i, res in enumerate(self.conv) :            
            x = res(x)
            #print("Decoder Convolutional Layer%d" % i)
            #print(x.shape)
            #print("")
            
        return x

#vector quantized variational autoencoder
class vector_quantizer(nn.Module):
    def __init__(self, num_embeddings = num_embeddings, embedding_dim = embedding_dim,
                 commitment_cost = commitment_cost):
        super(vector_quantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = nn.functional.mse_loss(quantized.detach(), inputs)
        q_latent_loss = nn.functional.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        quantized = quantized.permute(0,2,1).contiguous()
        avg_probs = torch.mean(encodings, dim=0)
        #perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        perplexity = len(torch.unique(encoding_indices))
        
        # convert quantized from BHWC -> BCHW
        return quantized, loss, perplexity, self._embedding.weight.data, encoding_indices
      
#variational autoencoder
class variational_bottleneck(nn.Module):
      def __init__(self, *args, **kwargs):
        super().__init__()

        self._pool = nn.AdaptiveMaxPool1d(1, return_indices=True)
        self._unpool = nn.MaxUnpool1d(lin_arc[0])
        
      def forward(self, x):
        #x, indices = self._pool(x)
        mu, logvar = torch.chunk(x, 2, dim=1)
        std = logvar.mul(0.5).exp_()
        eps1 = torch.autograd.Variable(std.data.new(std.size()).normal_())
        eps2 = torch.autograd.Variable(std.data.new(std.size()).normal_())
        out1 = eps1.mul(std).add_(mu)
        out2 = eps2.mul(std).add_(mu)
        out = torch.cat([out1, out2], dim=1)
        return out, mu, logvar
        #return self._unpool(out, indices), mu, logvar

#model
class model(nn.Module):
    def __init__(self, model_type, epsilon=1e-5):
        super(model, self).__init__()

        self._model_type = model_type

        self._encoder = encoder()
        self._v = variational_bottleneck()
        self._vq = vector_quantizer()
        self._decoder = decoder()
        
    def forward(self, x):
        encoded = self._encoder(x)
        
        if self._model_type == 'ae':
          recon = self._decoder(encoded)
          return recon, encoded

        elif self._model_type == 'vae':
          v_out, mu, logvar = self._v(encoded)
          recon = self._decoder(v_out)
          return recon, mu, logvar
        
        elif self._model_type == 'vqvae':
          quantized, loss, perplexity, embeddings, encodings = self._vq(encoded)
          recon = self._decoder(quantized)
          return recon, loss, perplexity, embeddings, encodings

## LOAD DATA ##
training_fasta = list(SeqIO.parse(train_file, "fasta"))
training_data = fasta_data(training_fasta, lin_arc[0])
training_loader = DataLoader(training_data, batch_size = batch_size, shuffle = True)

## LOAD MODEL ##

data_var = 0.032 #*32/20 #average variance per sequence? hardcoded for now because I'm impatient
sampling = 1

vae = model(model_type = model_type)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae.to(device)

optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate, amsgrad=False)

## TRAINING ##

'''def checkpoint():
  torch.save(vae, output_file + ".pt")

  f = plt.figure(figsize=(16,8))
  ax = f.add_subplot(1,2,1)
  ax.plot(train_res_recon_error_smooth)
  ax.set_yscale('log')
  ax.set_title('Smoothed NMSE.')
  ax.set_xlabel('iteration')

  ax = f.add_subplot(1,2,2)
  ax.plot(train_res_perplexity_smooth)
  ax.set_title('Smoothed Average codebook usage (perplexity).')
  ax.set_xlabel('iteration')

  f.savefig(output_file + "_loss.png")'''


def train(loader):

  vae.train()

  train_res_loss = []
  train_res_recon_error = []
  train_res_perplexity = []

  log = open(output_file + "_log.txt", "a")
  log.write("BEGIN TRAINING " + time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)) + "\n\n")
  log.close()


  for i in range(max_training_updates):
      batch = next(iter(loader))
      batch_data = batch['seq']
      batch_data = batch_data.to(device)
      
      optimizer.zero_grad()
      
      if model_type == 'ae':
        recon, encoder = vae(batch_data)
        loss = nn.functional.mse_loss(recon, batch_data)
        loss.backward

        optimizer.step()
        train_res_loss.append(loss.item())

      elif model_type == 'vae':
        recon, mu, logvar = vae(batch_data)
        MSE = nn.functional.mse_loss(recon, batch_data)
        #BCE = nn.functional.binary_cross_entropy(recon, batch_data)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD / (32 * 20 * 2000)
        loss = MSE + KLD
        loss.backward()
        optimizer.step()
        train_res_loss.append(loss.item())

      elif model_type == 'vqvae':
        batch_recon, vq_loss, perplexity, embeddings, encodings = vae(batch_data)
        recon_error = nn.functional.mse_loss(batch_recon, batch_data) / data_var
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()
        
        train_res_loss.append(loss.item())
        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity)

      if (i+1) == 1:
          torch.save(vae, output_prefix + "/models/" + output_prefix + "_" + str(i+1) + '.pt')
          #PLOT LOSS
          fig = go.Figure(data=go.Scatter(y=train_res_loss, x=[*range(0,len(train_res_loss))]),
          layout = Layout(
          paper_bgcolor='rgba(0,0,0,0)',
          plot_bgcolor='rgba(0,0,0,0)',
          font_color = 'white'
          ))
          
          fig.write_image(output_prefix + "/figs/loss_" + str(i+1) + ".png")

      if (i+1) % 5 == 0:
          torch.save(vae, output_prefix + "/models/" + output_prefix + "_" + str(i+1) + '.pt')
          #PLOT LOSS
          fig = go.Figure(data=go.Scatter(y=train_res_loss, x=[*range(0,len(train_res_loss))]),
          layout = Layout(
          paper_bgcolor='rgba(0,0,0,0)',
          plot_bgcolor='rgba(0,0,0,0)',
          font_color = 'white'
          ))

          fig.write_image(output_prefix + "/figs/loss_" + str(i+1) + ".png")

      if (i+1) % 100 == 0:
          log = open(output_file + "_log.txt", "a")
          log.write(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)) + "\n")
          log.write('%d iterations' % (i+1) + "\n")
          log.write('loss: %.3f' % np.mean(train_res_loss[-100:]) + "\n")
          #log.write('recon_error: %.3f' % np.mean(train_res_recon_error[-100:])+ "\n")
          #log.write('perplexity: %.3f' % np.mean(train_res_perplexity[-100:])+ "\n\n")
          log.close()


      '''if (i+1) % 100 == 0:          
          f = plt.figure(figsize=(16,8))
          ax = f.add_subplot(1,2,1)
          ax.plot(train_res_loss)
          ax.set_yscale('log')
          ax.set_title('Smoothed NMSE.')
          ax.set_xlabel('iteration')

          ax = f.add_subplot(1,2,2)
          ax.plot(train_res_perplexity)
          ax.set_title('Smoothed Average codebook usage (perplexity).')
          ax.set_xlabel('iteration')

          f.savefig(output_file + "_loss.png")
      
      if (i+1) > 2000:
          if min(train_res_loss[-2000:-1000]) <= min(train_res_loss[-1000:]):
            torch.save(vae, output_file + ".pt")
            
            train_res_loss_smooth = savgol_filter(train_res_loss[10:], 201, 7)
            train_res_perplexity_smooth = savgol_filter(train_res_perplexity[10:], 201 , 7)

            f = plt.figure(figsize=(16,8))
            ax = f.add_subplot(1,2,1)
            ax.plot(train_res_loss_smooth)
            ax.set_yscale('log')
            ax.set_title('Smoothed NMSE.')
            ax.set_xlabel('iteration')

            ax = f.add_subplot(1,2,2)
            ax.plot(train_res_perplexity_smooth)
            ax.set_title('Smoothed Average codebook usage (perplexity).')
            ax.set_xlabel('iteration')

            f.savefig(output_file + "_loss.png")
            return'''

  log = open(output_file + "_log.txt", "a")
  log.close()

## TESTING ##

testing_fasta = list(SeqIO.parse(test_file, "fasta"))
testing_data = fasta_data(testing_fasta, lin_arc[0])
testing_loader = DataLoader(testing_data, batch_size = batch_size, shuffle = False)

def test(data, model, batch_size = 100):
  
  log = open(output_file + "_log.txt", "a")
  log.write("BEGIN TESTING " + time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)) + "\n\n")
  log.close()

  #model = torch.load(model, map_location=device)
  #model = model
  model.eval()

  testing_loader = DataLoader(data, batch_size = batch_size, shuffle = False)

  output = []
  header = []

  if model_type == 'ae':
    for i, batch in enumerate(testing_loader):

      test_id = batch['id']

      test_seqs = batch['seq']
      test_seqs = test_seqs.to(device)

      recon, encoded = model(test_seqs)

      output.append(np.column_stack([test_id, encoded.squeeze(2).detach().cpu().numpy()]))

  if model_type == 'vae':
    for i, batch in enumerate(testing_loader):

      test_id = batch['id']

      test_seqs = batch['seq']
      test_seqs = test_seqs.to(device)

      recon, mu, logvar = model(test_seqs)

      output.append(np.column_stack([test_id, mu.squeeze(2).detach().cpu().numpy()]))
  
  dims = []
  for i in range(len(output[0][0])-1):
    dims.append("Dim_" + str(i))
    header = ['Entry'] + dims

  df = pd.DataFrame(np.concatenate(output), columns=header)

  return df


  output = []
  header = ['Entry', 'out', 'out2']

  for i, batch in enumerate(testing_loader):

      validation_id = batch['id']

      validation_seqs = batch['seq']
      validation_seqs = validation_seqs.to(device)

      x_recon, m1, m2 = model(validation_seqs)

      output.append(np.column_stack([validation_id, mu.squeeze(1).detach().cpu().numpy().squeeze(1)]))
    

  #return pd.DataFrame(output, columns=header)
  df = pd.DataFrame(np.concatenate(output), columns=header)
  #df = df.sort_values(by='Encoding').reset_index(drop=True)

  out = {}

  return df
#model_file = output_file + ".pt"
#encodings = gen_embed(test_file, model_file)

testiter = iter(testing_loader)

next1 = next(testiter)
testid = next1['id']
testseq = next1['seq']

fig = go.Figure(data=go.Heatmap(
        z=testseq.to(device)[0].detach().cpu().numpy(),
        colorscale='Viridis', showscale = False),
        layout = Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color = 'white',
        font_size = 18,
        width = 750, height = 750
        ))

fig.update_yaxes(categoryarray=[
  'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
  'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
], type='category', showgrid=False, zeroline = False)

fig.update_xaxes(showgrid = False, zeroline = False, visible = False)

fig.write_image(output_prefix + "/figs/input.png")

import re

for filename in os.listdir(output_prefix + '/models'):
  tmpmodel = torch.load(output_prefix + '/models/' + filename, map_location=device)
  suf = re.split('\.', re.split('_', filename)[1])[0]

  fig2 = go.Figure(data=go.Heatmap(
        z=tmpmodel(testseq.to(device))[0][0].detach().cpu().numpy(),
        colorscale='Viridis', showscale = False),
        layout = Layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color = 'white',
        font_size = 18,
        width = 750, height = 750
        ))

  fig2.update_yaxes(categoryarray=[
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
    'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
  ], type='category', showgrid=False, zeroline = False)

  fig2.update_xaxes(showgrid = False, zeroline = False, visible = False)

  fig2.write_image(output_prefix + "/figs/output_" + suf + ".png")

  out = test(testing_data, tmpmodel)
  uniprot_df = pd.read_csv("data/uniprot_reference.txt", sep='\t', names = ['Entry', 'Organism', 'Protein families', 'Gene ontology IDs'])
  uniprot_df = out.merge(uniprot_df)
  uniprot_df['plot_fams'] = np.where(uniprot_df['Protein families'].isin(uniprot_df['Protein families'].value_counts().head(5).index), uniprot_df['Protein families'], 'Other')
  fig = px.scatter(uniprot_df, x='Dim_0', y='Dim_1', color='plot_fams', hover_name='Entry')

  fig.update_layout(Layout(
          paper_bgcolor='rgba(0,0,0,0)',
          plot_bgcolor='rgba(0,0,0,0)',
          width = 750, height = 750,
          showlegend = False))

  fig.update_yaxes(showgrid=False, zeroline = False, visible = False)

  fig.update_xaxes(showgrid = False, zeroline = False, visible = False)

  fig.write_image(output_prefix + "/figs/embeds_" + suf + ".png")

