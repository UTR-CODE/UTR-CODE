import numpy as np

from Pred_batch import load_model
import argparse
import seqpro as sp
import torch
import pandas as pd
def ism(seq):

    length = len(seq)
    variants = [seq]
    tmp = [list(seq) for _ in range(length)]
    for item in range(length):
        t = tmp[item]
        raw = t[item]
        for base in ("A", "T", "C", "G"):
            if raw != base:
                t[item] = base
                variants.append("".join(t))
    return variants




def design(model,
           utr5,
           utr3,
           cds,
           mRNA,
           rnaCount,
           rounds=5,):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    raw_seq = utr5
    evo = []
    pred = []


    for _ in range(rounds):
        candidates = ism(raw_seq)
        input_data = prepare_data(candidates, utr3, cds)
        with torch.no_grad():
            out = model(
                Tensor(input_data["utr"]).transpose(2, 1).float().to(device),
                Tensor(input_data["cds"]).transpose(2, 1).float().to(device),
                mRNA.repeat(input_data["utr"].shape[0], 1),
                Tensor([rnaCount]).reshape(1, 1).repeat(input_data["utr"].shape[0], 1)
            )[:, 0].detach().cpu().numpy()
            out = np.expm1(out) / 5

        raw_seq = candidates[out.argmax()]
        pred.append(out.max())
        evo.append(raw_seq)
    return evo, pred


def ohe(seq):
    return sp.ohe(seq,alphabet=sp.alphabets.DNA)


def prepare_data(utr5s, utr3, cds):
    utr = [i[-1000:].rjust(1000, "N") + utr3[-4500:].ljust(4000, "N") for i in utr5s]

    cds = [cds[:4500].ljust(4500, "N")]

    return {"utr": ohe(utr),
            "cds": ohe(cds)}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--mRNA", type=str, default="../outer_data_verify/HEK293T.txt")
    parser.add_argument("--cds", type=str, default = "ATGGGAGTCAAAGTTCTGTTTGCCCTGATCTGCATCGCTGTGGCCGAGGCCAAGCCCACCGAGAACAACGAAGACTTCAACATCGTGGCCGTGGCCAGCAACTTCGCGACCACGGATCTCGATGCTGACCGCGGGAAGTTGCCCGGCAAGAAGCTGCCGCTGGAGGTGCTCAAAGAGATGGAAGCCAATGCCCGGAAAGCTGGCTGCACCAGGGGCTGTCTGATCTGCCTGTCCCACATCAAGTGCACGCCCAAGATGAAGAAGTTCATCCCAGGACGCTGCCACACCTACGAAGGCGACAAAGAGTCCGCACAGGGCGGCATAGGCGAGGCGATCGTCGACATTCCTGAGATTCCTGGGTTCAAGGACTTGGAGCCCATGGAGCAGTTCATCGCACAGGTCGATCTGTGTGTGGACTGCACAACTGGCTGCCTCAAAGGGCTTGCCAACGTGCAGTGTTCTGACCTGCTCAAGAAGTGGCTGCCGCAACGCTGTGCGACCTTTGCCAGCAAGATCCAGGGCCAGGTGGACAAGATCAAGGGGGCCGGTGGTGACTAA")

    parser.add_argument("--utr5", type=str,required=True )
    parser.add_argument("--utr3", type=str,
                        default="GCTGGAGCCTCGGTGGCCTAGCTTCTTGCCCCTTGGGCCTCCCCCCAGCCCCTCCTCCCCTTCCTGCACCCGTACCCCCGTGGTCTTTGAATAAAGTCTGAGTGGGCGGCA")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--rnaCount", type=float, default=6.3)
    parser.add_argument("--weights", required=False)
    parser.add_argument("--sample", type=str, default="SRX10567366")
    args = parser.parse_args()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    model, device = load_model("../",args.weights)
    mRNA = pd.read_parquet("VAE_1024.parquet")
    mRNA_array = Tensor(mRNA.loc[:, args.sample].values.reshape(1, -1)).float()

    result, predict_value = design(model,
                   args.utr5,
                   args.utr3,
                   args.cds,
                   mRNA_array,
                   args.rnaCount,
                   args.rounds,
                   )
    for line in zip(result, predict_value):
        print(f"{line[0]}\t{line[1]}")

