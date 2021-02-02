#!/bin/bash

#dependency https://github.com/tanghaibao/goatools

cd test_output/GOs/

if [ ! -f go-basic.obo ]; then
	wget "http://geneontology.org/ontology/go-basic.obo"
fi

cat GO* > population.txt

for GO in GO*
do
find_enrichment.py $GO population.txt ../../GOA.txt --pval=0.001 --method=fdr_bh --pval_field=fdr_bh > ${GO%.txt}_results.txt
done

for RES in *_results.txt
do
E=$(echo $RES | awk -F '[O_]' '{print $2}')
grep "\te\t" $RES | awk -v var=$E '{print $0 "\t" var}'  > ${RES%_results.txt}_table.txt
done

echo -e "GO\tName\tEncoding" > out.txt
cat *_table.txt | awk -F '\t' '{print $1 "\t" $4 "\t" $12}' >> ../GO_results.txt

rm *_results.txt
rm *_table.txt
rm population.txt