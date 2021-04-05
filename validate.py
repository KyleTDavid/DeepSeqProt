import pandas as pd
import sys

uniprot_ref = pd.read_csv(sys.argv[1], sep='\t')
results = pd.read_csv(sys.argv[2], sep ='\t', skiprows=(0), names=['Entry', 'Encoding'])

df = results.iloc[:, 0:2].merge(uniprot_ref)
df['n'] = df.groupby('Encoding')['Encoding'].transform('count')

results = []

results.append(["# of categories", len(set(df.Encoding))])

#only look at protein families with at least 2 members, group by family
v = df['Protein families'].value_counts()
fams = df[df['Protein families'].isin(v.index[v.gt(1)])]
group = fams.groupby('Protein families')['Encoding']

#percentage of complete families (all members have the same encoding)
com = group.nunique()
results.append(["complete families",(len(com[com==True]) / len(com))])

#family completeness (largest number of members that share a cluster / family size)
results.append(["family completeness",
                group.apply(lambda x: x.value_counts().head(1)).sum() / group.size().sum()])
                
#run gene ontology enrichment analysis
import goatools
# Get http://geneontology.org/ontology/go-basic.obo
from goatools.base import download_go_basic_obo
obo_fname = download_go_basic_obo()

df['Gene ontology IDs'] = df['Gene ontology IDs'].str.replace(' ','')
df.drop(['Encoding', 'Organism', 'Protein families', 'n'], axis=1).to_csv("GOA.txt", sep='\t', header=False, index=False)

from goatools.anno.idtogos_reader import IdToGosReader
objanno = IdToGosReader("GOA.txt")
ns2assoc = objanno.get_id2gos()

from goatools.obo_parser import GODag
obodag = GODag("go-basic.obo")

from goatools.go_enrichment import GOEnrichmentStudy
goeaobj = GOEnrichmentStudy(
        df.Entry, 
        ns2assoc, # geneid/GO associations
        obodag, # Ontologies
        propagate_counts = False,
        alpha = 0.001, # default significance cut-off
        methods = ['fdr_bh']) # default multipletest correction method

gos = []
#sort by size
bdf = df[df.n > 1]
for e in set(bdf['Encoding']):
#for e in set(encodings.Encoding):
  goea_results = goeaobj.run_study(list(df[df['Encoding']==e].Entry))
  for r in goea_results:
      if (r.p_fdr_bh < 0.001) & (r.enrichment=='e') :
        id = r.goterm.id
        name = r.name
        cat = r.goterm.namespace
        members = r.study_items
        gos.append([id, name, cat, e, members])

godf = pd.DataFrame(gos, columns=['id', 'name', 'category', 'encoding', 'members'])

#what % of encodings have at least one significant GO?
results.append(["significant categories",
                (len(set(godf.encoding)) / len(set(df.Encoding)))])

members = [item for sublist in godf.members for item in sublist]

#how many members have at least one significant GO?
results.append(["GO accuracy",
                len(set(members)) / len(df[df["Gene ontology IDs"].notnull()].Entry)])

#how many members have at least one significant unique GO?
v = godf['name'].value_counts()
uniqgodf = godf[godf['name'].isin(v.index[v.lt(2)])]

uniqmembers = [item for sublist in uniqgodf.members for item in sublist]

results.append(["unique GO accuracy",
                len(set(uniqmembers)) / len(df[df["Gene ontology IDs"].notnull()].Entry)])
                
results.append(["# of Good Categories",
				len(set(uniqgodf['encoding']))])

resultsdf = pd.DataFrame(results)
resultsdf.to_csv(sys.argv[2].replace("results", "report"), sep='\t', header=False, index=False)

uniqgodf['member count'] = len(uniqgodf.members)

godf.to_csv(sys.argv[2].replace("results", "GO"), sep='\t', header=False, index=False)

