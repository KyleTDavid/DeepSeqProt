# DeepSeqProt

#### DeepSeqProt is an unsupervised deep learning framework for protein sequence data

---
### DEPENDENCIES
Nonstandard Python Libraries:
- [`PyTorch`](https://pytorch.org)
- [`NumPy`](https://numpy.org/)
- [`pandas`](https://pandas.pydata.org/)
- [`Biopython`](https://biopython.org/)

These can be installed by running `pip install -r requirements.txt` or `conda install --file requirements.txt`

---
### USAGE

```
> python DeepSeqProt.py -i [input] -o [output] -t [train] [options]

-i [fasta file]         Input fasta file, each sequence will be passed through the trained model [REQUIRED]
-o [string]             Name of the output directory and prefix of several output files [REQUIRED]
-t [fasta | model file] Either a fasta file of sequences used to train the model or a previously trained model file [REQUIRED]
-w [integer list]       List of integers representing the length and width of the encoder architecture [2000, 1500, 1000, 500, 1]
-n [integer]            Number of discerete embeddings used to populate latent space
-a [integer]            Number of latent space dimensions, this paramater introduces a simple convolutional layer to transform the default 20 channels into 'a' channels prior to quantization
-c [float]              Beta, hyperparameter used to weight commitment loss [0.1]
-d [float]              Weight decay [0.9]
-b [integer]            Batch size [32]
-l [float]              Learning rate [1e-5]
-m [float]              Maximum number of training updates [1e6]
-h                      Print help
```

---
### OUTPUT
Contents of `output` directory:

- `fastas/` Directory containing fasta files of each embedding with their associated input sequences

- `output_coordinates.txt` Assigned embedding and coordinates in latent space for each sequence in tab delimted format

- `output.pt` Pickled PyTorch object of the trained model which can be used for future runs

- `output_log.txt` Log file

---

### VALIDATION & BENCHMARKING
For the sake of reproducibility we also include several scripts used to validate and benchmark DeepSeqProt during development

#### run_benchmarks.sh
```
> ./run_benchmarks.sh -i [fasta file]
```
Will run each clustering algorithm used during development to benchmark DeepSeqProt and format it into a `*_coordinates.txt` file which can be used by `validate.py`. Note that each algorithm ([CD-HIT](https://github.com/weizhongli/cdhit0), [MMseqs2](https://github.com/soedinglab/mmseqs2), [MCL](https://micans.org/mcl/man/mcl.html)) must be installed and included in your PATH

#### validate.py
```
> python validate.py -i [coordinates file] -r [reference table]
```
Calculates several summary statistics from a `*_coordinates.txt` file generated either by `DeepSeqProt.py` or `run_benchmarks.sh`. Requires a tab-delimited reference table in the format:
header | organism | protein family | and gene ontology IDs 
 --- | --- | --- | --- 
 
for each sequence in `*_coordinates.txt`. We recommend using the Uniprot database for this purpose, an example table can be found [here](https://www.uniprot.org/uniprot/?query=amphipod&format=tab&limit=10&columns=id,organism,families,go-id&sort=score). In addition to the libraries used by DeepSeqProt `validate.py` also requires the nonstandard Python libraries [`scikit-learn`](https://scikit-learn.org/stable/) and [`goatools`](https://github.com/tanghaibao/goatools)
