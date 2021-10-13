#!/bin/bash
echo -e "Entry\tEncoding" > header.txt
#MCL

diamond makedb --in $1 -d mcl

diamond blastp -d mcl -q $1 > blastall.txt

awk '{if($1!~/^#/ && $3>=50) print $1"\t"$2"\t"($3/100)}' blastall.txt > blastmatrix.txt

mcl blastmatrix.txt --abc

sed 's/$/\t/g' out.blastmatrix.txt.I20 | awk '{gsub("\t","\t"NR" ")}; $0' | sed 's/\ /\n/g' | sed  '/^$/d' > mcl_results.txt
cat header.txt mcl_results.txt > mcl_coordinates.txt

#mmseq2

mmseqs createdb $1 DB
mmseqs linclust --min-seq-id 0.25 DB DB_clu tmp
mmseqs createtsv DB DB DB_clu tmp.tsv

awk '{print $2"\t"$1}' tmp.tsv > mmseq2_25_results.txt
cat header.txt mmseq2_25_results.txt > mmseq2_25_coordinates.txt

rm -rf DB*

mmseqs createdb $1 DB
mmseqs linclust --min-seq-id 0.50 DB DB_clu tmp
mmseqs createtsv DB DB DB_clu tmp.tsv

awk '{print $2"\t"$1}' tmp.tsv > mmseq2_50_results.txt
cat header.txt mmseq2_50_results.txt > mmseq2_50_coordinates.txt

rm -rf DB*

mmseqs createdb $1 DB
mmseqs linclust --min-seq-id 0.75 DB DB_clu tmp
mmseqs createtsv DB DB DB_clu tmp.tsv

awk '{print $2"\t"$1}' tmp.tsv > mmseq2_75_results.txt
cat header.txt mmseq2_75_results.txt > mmseq2_75_coordinates.txt

rm -rf DB*

#cdhit

cd-hit -o cdhit_50 -c 0.50 -n 3 -i $1

while read line
do
if [[ "$line" == ">"* ]]; then
    encoding=$(echo $line | sed 's/>Cluster //g')
else
	ID=$(echo $line | grep -o -P '(?<=>).*(?=\.\.\.)')
    echo -e "$ID\t$encoding" >> cdhit_50_results.txt
fi
done<cdhit_50.clstr
cat header.txt cdhit_50_results.txt > cdhit_50_coordinates.txt

cd-hit -o cdhit_75 -c 0.75 -n 5 -i $1

while read line
do
if [[ "$line" == ">"* ]]; then
    encoding=$(echo $line | sed 's/>Cluster //g')
else
	ID=$(echo $line | grep -o -P '(?<=>).*(?=\.\.\.)')
    echo -e "$ID\t$encoding" >> cdhit_75_results.txt
fi
done<cdhit_75.clstr
cat header.txt cdhit_75_results.txt > cdhit_75_coordinates.txt

rm blast*
rm out*
rm -r tmp*
