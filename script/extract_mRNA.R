library(GenomicFeatures)
library(stringr)
library(rtracklayer)
library(Biostrings)
library(BSgenome)
args <- commandArgs(trailingOnly = TRUE)
genome <- args[1]
gff <- args[2]
out1 <- args[3]
out2 <- args[4]
out3 <- args[5]

Hsapiens <- readDNAStringSet(genome)
split_name <- str_split(names(Hsapiens ), " ")
names(Hsapiens ) <- sapply(split_name , function(x) x[1])



txdb1 <- makeTxDbFromGFF(gff)

utr5 <- fiveUTRsByTranscript(txdb1, use.names=TRUE)
fiveUTR_seqs <- extractTranscriptSeqs(Hsapiens,utr5)
writeXStringSet(fiveUTR_seqs,out1,format = "fasta")

utr3 <- threeUTRsByTranscript(txdb1, use.names=TRUE)
threeUTR_seqs <- extractTranscriptSeqs(Hsapiens,utr3)
writeXStringSet(threeUTR_seqs,out2,format = "fasta")


cds <- cdsBy(txdb1, use.names=TRUE)
cds_seqs <- extractTranscriptSeqs(Hsapiens,cds)
writeXStringSet(cds_seqs,out3,format = "fasta")
