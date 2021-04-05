import os
import time
import torch
import random
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
from Bio import SeqIO
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from math import ceil
from functools import partial
import matplotlib.pyplot as plt

## PARAMETERS & MODEL ##
start_time = time.time()

#encoder
arch = [2000, 1500, 1000, 500, 1]

#vector quantizer        
num_embeddings = 1000
commitment_cost = 0.1
decay = 0.9

#training
batch_size = 32
learning_rate = 1e-5
max_training_updates = 100000

#inputs and outputs
#test_file = sys.argv[1]
#train_file = sys.argv[1]
#output_suffix = sys.argv[1].split("/")[-1][:-3]

test_file = 'data/mammalia/mammalia_test.fa'
train_file = 'data/mammalia/mammalia_train.fa'
output_suffix = 'mammalia_short500c'

#write log
os.mkdir(output_suffix)
output_file = output_suffix + "/" + output_suffix
log = open(output_file + "_log.txt", "w")
log.write("PARAMETERS\n\n")
log.write("Encoder\n")
log.write("encoder architecture = " + str(arch) + "\n")
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

        self.conv = nn.Conv1d(20,2,1,1)

    def forward(self, x):
        for i, block in enumerate(self.blocks) :            
            x = block(x)
        #x = self.conv(x)
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

        self._sm = nn.Softmax(dim=2)
        self._kl = nn.KLDivLoss(reduction='batchmean')
        
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
        e_latent_loss = self._kl(self._sm(quantized.detach()), self._sm(inputs))
        loss = self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim = 0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        #convert quantized from BLC -> BCL #clean this
        return quantized.permute(0,2,1).contiguous(), loss, perplexity, self._embedding.weight.data, encoding_indices

#decoder 
class decoder(nn.Module):
    def __init__(self, arch=arch, *args, **kwargs):
        super().__init__()

        self.arch = arch[::-1]

        self.deconv = nn.ConvTranspose1d(2,20,1,1)

        self.blocks = nn.ModuleList([
                      nn.Linear(self.arch[0], self.arch[1]),
                      *[nn.Linear(input, output)
                      for (input, output) in zip(self.arch[1:], self.arch[2:])]
        ])

    def forward(self, x):
        #x = self.deconv(x)

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
        quantized, loss, perplexity, encodings, embeddings = self._vq(encoded)
        x_recon = self._decoder(quantized)
            
        return loss, x_recon, perplexity, quantized, encoded, embeddings, encodings

## LOAD DATA & MODEL ##

data = fasta_data(train_file, arch[0])
training_loader = DataLoader(data, batch_size = batch_size, shuffle = True)

data_var = 0.032 #*32/20 #average variance per sequence? hardcoded for now because I'm impatient
embedding_dim = 20 * arch[-1]

vae = model(arch, num_embeddings, embedding_dim, commitment_cost, decay)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae.to(device)

optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate, amsgrad=False)

## TRAINING ##

vae.train()

train_res_loss = []
train_res_recon_error = []
train_res_perplexity = []

log = open(output_file + "_log.txt", "a")
log.write("BEGIN TRAINING " + time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)) + "\n\n")
log.close()

#list to store embeddings to observe learning
embeddings_list = list()

class done_learning(Exception): pass

try:
  for i in range(max_training_updates):
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

      if (i+1) % 1000 == 0:
          torch.save(vae, output_file + ".pt")
      
      if (i+1) > 1000:
          if np.mean(train_res_loss[-1000:-500]) <= np.mean(train_res_loss[-500:]):
            torch.save(vae, output_file + ".pt")
                        
            train_res_loss_smooth = savgol_filter(train_res_loss[100:], 75, 7)
            train_res_perplexity_smooth = savgol_filter(train_res_perplexity[100:], 75, 7)

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
            raise done_learning

except done_learning:
  pass

log = open(output_file + "_log.txt", "a")
log.close()

## TESTING ##

#get encoding for each sequence in test fasta
def gen_embed(fasta, model):
  batch_size = 1
  
  log = open(output_file + "_log.txt", "a")
  log.write("BEGIN TESTING " + time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)) + "\n\n")
  log.close()

  model = torch.load(model, map_location=device)
  model.eval()

  validation_data = fasta_data(fasta, arch[0])
  validation_loader = DataLoader(validation_data, batch_size = batch_size, shuffle = False)

  output = []
  header = []

  for i, batch in enumerate(validation_loader):

      validation_id = validation_data[i]['id']

      validation_seqs = batch['seq']
      validation_seqs = validation_seqs.to(device)
      vq_output_eval = model._encoder(validation_seqs)
      valid_quantize, loss, perplexity, encodings, embeddings = model._vq(vq_output_eval)

      encoding = int(embeddings.detach().cpu().numpy().flatten())
      embeds = valid_quantize.view(batch_size, -1).detach().cpu().numpy().flatten()

      output.append([validation_id, encoding] + list(embeds))
      
      if i == 0:
        dims = []
        for i in range(len(embeds)):
          dims.append("Dim_" + str(i))
          header = ['Entry', 'Encoding'] + dims

      if (i+1) % 1000 == 0:
        log = open(output_file + "_log.txt", "a")
        log.write(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)) + "\n")
        log.write("%d sequences processed" % (i+1*batch_size)+ "\n\n")
    


  return pd.DataFrame(output, columns=header)
  
model_file = output_file + ".pt"
encodings = gen_embed(test_file, model_file)

#split test fasta by encoding 
os.mkdir(output_suffix + "/fastas")
for e in np.unique(encodings["Encoding"].tolist()):
    
    input_seq_iterator = SeqIO.parse(test_file, "fasta")
    entries = encodings[encodings["Encoding"] == e]["Entry"].tolist()
        
    subfasta = [record for record in input_seq_iterator if record.id in entries]
    SeqIO.write(subfasta, output_suffix + "/fastas/subfasta" + str(e) + ".fa", "fasta")

## VALIDATION ##

#aggregate coordinates of each encoding for plotting
cols = []
cols.append(encodings.groupby('Encoding')['Encoding'].size().rename("n"))
for dim in list(encodings.columns)[2:]:
  cols.append(encodings.groupby('Encoding')[dim].mean())
coords = pd.concat(cols, axis =1)
coords.insert(loc=0, column='Encoding', value=coords.index)

coords.to_csv(output_file + "_coordinates.txt", sep='\t', index=False)

#incorporate uniprot info
uniprot_ref = pd.read_csv("data/uniprot_reference.txt", sep='\t', names = ['Entry', 'Organism', 'Protein families', 'Gene ontology IDs'])
uniprot_df = encodings.iloc[:, 0:2].merge(uniprot_ref)
uniprot_df['n'] = uniprot_df.groupby('Encoding')['Encoding'].transform('count')
results = []

uniprot_df.to_csv(output_file + "_clusters.txt", sep='\t', header=False, index=False)

results.append(["# of categories", len(set(uniprot_df.Encoding))])

#only look at protein families with at least 2 members, group by family
v = uniprot_df['Protein families'].value_counts()
fams = uniprot_df[uniprot_df['Protein families'].isin(v.index[v.gt(1)])]
group = fams.groupby('Protein families')['Encoding']

#percentage of complete families (all members have the same encoding)
com = group.nunique()
results.append(["complete families",(len(com[com==True]) / len(com))])

#family completeness (largest number of members that share a cluster / family size)
results.append(["family completeness",
                group.apply(lambda x: x.value_counts().head(1)).sum() / group.size().sum()])

#run gene ontology enrichment analysis
# Get http://geneontology.org/ontology/go-basic.obo
from goatools.base import download_go_basic_obo
obo_fname = download_go_basic_obo()

uniprot_df['Gene ontology IDs'] = uniprot_df['Gene ontology IDs'].str.replace(' ','')
uniprot_df.drop(['Encoding', 'Organism', 'Protein families', 'n'], axis=1).to_csv("GOA.txt", sep='\t', header=False, index=False)

from goatools.anno.idtogos_reader import IdToGosReader
objanno = IdToGosReader("GOA.txt")
ns2assoc = objanno.get_id2gos()

from goatools.obo_parser import GODag
obodag = GODag("go-basic.obo")

from goatools.go_enrichment import GOEnrichmentStudy
goeaobj = GOEnrichmentStudy(
        uniprot_df.Entry, 
        ns2assoc, # geneid/GO associations
        obodag, # Ontologies
        propagate_counts = False,
        alpha = 0.001, # default significance cut-off
        methods = ['fdr_bh']) # default multipletest correction method

gos = []
for e in set(uniprot_df[uniprot_df['n']>=2]['Encoding']):
  goea_results = goeaobj.run_study(list(uniprot_df[uniprot_df['Encoding']==e].Entry))
  for r in goea_results:
      if (r.enrichment=='e') & (r.p_fdr_bh < 0.001) :
        id = r.goterm.id
        name = r.name
        cat = r.goterm.namespace
        members = r.study_items
        gos.append([id, name, cat, e, members])

godf = pd.DataFrame(gos, columns=['id', 'name', 'category', 'encoding', 'members'])
godf['unique?'] = ~godf['name'].duplicated(keep=False)
godf['member_count'] = godf.members.apply (lambda x: len(x))
godf['representation'] = godf.apply (lambda row: row.member_count / int(np.unique(uniprot_df[uniprot_df.Encoding==row.encoding]['n'])), axis=1)

#what % of encodings have at least one significant GO?
results.append(["good categories",
                (len(set(godf.encoding)) / len(set(uniprot_df.Encoding)))])

members = [item for sublist in godf.members for item in sublist]

#how many members have at least one significant GO?
results.append(["GO accuracy",
                len(set(members)) / len(uniprot_df[uniprot_df["Gene ontology IDs"].notnull()].Entry)])

#how many members have at least one significant unique GO?
v = godf['name'].value_counts()
uniqgodf = godf[godf['unique?']==True]

uniqmembers = [item for sublist in uniqgodf.members for item in sublist]

results.append(["unique GO accuracy",
                len(set(uniqmembers)) / len(uniprot_df[uniprot_df["Gene ontology IDs"].notnull()].Entry)])

godf.drop(['members'], axis=1).to_csv(output_file + "_GO.txt", sep='\t', header=True, index=False)

resultsdf = pd.DataFrame(results)
resultsdf.to_csv(output_file + "_report.txt", sep='\t', header=False, index=False)
