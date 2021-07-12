import pandas as pd
import sys

uniprot_ref = pd.read_csv(sys.argv[1], sep='\t')
results = pd.read_csv(sys.argv[2], sep ='\t', skiprows=(0), names=['Entry', 'Encoding'])

uniprot_df = results.iloc[:, 0:2].merge(uniprot_ref)
uniprot_df['n'] = uniprot_df.groupby('Encoding')['Encoding'].transform('count')

results = []

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

#adjusted mutual information
uniprot_notnull = uniprot_df[uniprot_df['Protein families'].notnull()]
results.append(["adjusted mutual information",
                adjusted_mutual_info_score(uniprot_notnull['Protein families'], uniprot_notnull['Encoding'])])

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
      if (r.enrichment=='e'):
        id = r.goterm.id
        name = r.name
        cat = r.goterm.namespace
        members = r.study_items
        p = r.p_fdr_bh
        gos.append([id, name, cat, e, members, p])

godf = pd.DataFrame(gos, columns=['id', 'name', 'category', 'encoding', 'members', 'p'])
godf['unique?'] = ~godf['name'].duplicated(keep=False)
godf['member_count'] = godf.members.apply (lambda x: len(x))
godf['representation'] = godf.apply (lambda row: row.member_count / int(np.unique(uniprot_df[uniprot_df.Encoding==row.encoding]['n'])), axis=1)

godf_sig = godf[godf.p <= 0.01]

#what % of encodings have at least one significant GO?
results.append(["good categories",
                (len(set(godf_sig.encoding)) / len(set(uniprot_df.Encoding)))])

members = [item for sublist in godf_sig.members for item in sublist]

#how many members have at least one significant GO?
results.append(["GO accuracy",
                len(set(members)) / len(uniprot_df[uniprot_df["Gene ontology IDs"].notnull()].Entry)])

#how many members have at least one significant unique GO?
uniqgodf = godf_sig[godf_sig['unique?']==True]

uniqmembers = [item for sublist in uniqgodf.members for item in sublist]

results.append(["unique GO accuracy",
                len(set(uniqmembers)) / len(uniprot_df[uniprot_df["Gene ontology IDs"].notnull()].Entry)])

resultsdf = pd.DataFrame(results)
resultsdf.to_csv(sys.argv[2].replace("results", "report"), sep='\t', header=False, index=False)

uniqgodf['member count'] = len(uniqgodf.members)

godf.to_csv(sys.argv[2].replace("results", "GO"), sep='\t', header=False, index=False)

