## How to prepare  file to retrain UTR-CODE  with custom data


### 1. Extract mRNA sequences from genome and gene annotation

`Rscript extract_mRNA.R hg38.fa gencode.v44.annotation.gtf human_utr5.fa human_utr3.fa human_cds.fa`

### 2. Prepare RNA-seq and Ribo-seq RPKM matrix
| Gene_id   | Sample1 | Sample2 |
|-----------|---------| --------|
| ENSG00000186092.7    | 1.4     | 3.4|
| ENSG00000284733.2 | 2.6     | 8.1|

### 3. Python script to prepare training rawdata
```shell
# human

from Bio import SeqIO
utr5 = {i.id:str(i.seq) for i in SeqIO.parse("human.utr5.fa","fasta")}
utr3 = {i.id:str(i.seq) for i in SeqIO.parse("human.utr3.fa","fasta")}
cds = {i.id:str(i.seq) for i in SeqIO.parse("human.cds.fa","fasta")}

import pandas as pd
annotation = pd.read_table("gencode.v44.annotation.gtf",comment = "#",header = None)

mRNA = annotation[annotation[2] == 'transcript']
mRNA = mRNA[mRNA[8].str.contains("protein_coding")]
mRNA = mRNA[mRNA[8].str.contains("MANE")]
mRNA["Gene_ID"] = mRNA[8].str.extract('gene_id \"(.*?)\";')
mRNA["transcript"] = mRNA[8].str.extract('transcript_id \"(.*?)\";')

mRNA["utr5"] = [utr5.get(i,"") for i in mRNA["transcript"]]
mRNA["utr3"] = [utr3.get(i,"") for i in mRNA["transcript"]]
mRNA["cds"] = [cds.get(i,"") for i in mRNA["transcript"]]


RNA = pd.read_csv("human.RNA.csv")
RPF = pd.read_csv("human.RPF.csv")

# RNA-seq sample ID same with Ribo-seq
col = list(RPF.columns )
col[0] = "Gene_ID"
RNA.columns = col
RPF.columns = col

final = RNA.merge(mRNA.loc[:,["Gene_ID","transcript"]]).melt(id_vars = ["Gene_ID","transcript"],
                             var_name = "sample_RPF_id",
                             value_name = "RNA_RPKM")
final["RPF_RPKM"] = RPF.merge(mRNA.loc[:,["Gene_ID","transcript"]]).melt(id_vars = ["Gene_ID","transcript"],
                             var_name = "sample_RPF_id",
                             value_name = "RPF_RPKM")["RPF_RPKM"]

final.columns = "gene_id","isoform_id","sample_RPF_id","RNA_RPKM","RPF_RPKM"

final.iloc[:,[2,0,1,3,4]].to_parquet("training_data/human.parquet")

mRNA["transcript"].to_csv("training_data/human_transcript_ids.txt",header = None,index=False)


import seqpro as sp
data = mRNA.copy()
data["utr5"] = data["utr5"].fillna("").str.slice(-1000).str.pad(1000, "left", "N")
data["utr3"] = data["utr3"].fillna("").str.slice(-4000).str.pad(4000, "right", "N")
data["utr_sequence"] = data["utr5"].str.cat(data["utr3"], sep="")
data["cds_sequence"] = data["cds"].str.slice(0,4500).str.pad(4500,"right","N")
import numpy as np
np.savez_compressed("training_data/human_utr_ohe.npz",**{"ohe":sp.ohe(data["utr_sequence"].to_list(),alphabet=sp.DNA)})
np.savez_compressed("training_data/human_cds_ohe.npz",**{"ohe":sp.ohe(data["cds_sequence"].to_list(),alphabet=sp.DNA)})

```