from Pred_batch import load_model
import argparse
import seqpro as sp
import torch
import pandas as pd
import numpy as np




def get_orig_rpf(val: float) -> float:
    return (np.e**val - 1) / 5



def ohe(seq):
    return sp.ohe(seq,alphabet=sp.alphabets.DNA)

def prepare_data(utr5, utr3, cds):
    utr = [utr5[-1000:].rjust(1000, "N") + utr3[-4500:].ljust(4000, "N")]

    cds = [cds[:4500].ljust(4500, "N")]

    return {"utr": ohe(utr),
            "cds": ohe(cds)}



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--cds", type=str, default = "ATGGGAGTCAAAGTTCTGTTTGCCCTGATCTGCATCGCTGTGGCCGAGGCCAAGCCCACCGAGAACAACGAAGACTTCAACATCGTGGCCGTGGCCAGCAACTTCGCGACCACGGATCTCGATGCTGACCGCGGGAAGTTGCCCGGCAAGAAGCTGCCGCTGGAGGTGCTCAAAGAGATGGAAGCCAATGCCCGGAAAGCTGGCTGCACCAGGGGCTGTCTGATCTGCCTGTCCCACATCAAGTGCACGCCCAAGATGAAGAAGTTCATCCCAGGACGCTGCCACACCTACGAAGGCGACAAAGAGTCCGCACAGGGCGGCATAGGCGAGGCGATCGTCGACATTCCTGAGATTCCTGGGTTCAAGGACTTGGAGCCCATGGAGCAGTTCATCGCACAGGTCGATCTGTGTGTGGACTGCACAACTGGCTGCCTCAAAGGGCTTGCCAACGTGCAGTGTTCTGACCTGCTCAAGAAGTGGCTGCCGCAACGCTGTGCGACCTTTGCCAGCAAGATCCAGGGCCAGGTGGACAAGATCAAGGGGGCCGGTGGTGACTAA")
    parser.add_argument("--utr5", type=str, default="GGGAAATAAGAGAGAAAAGAAGAGTAAGAAGAAATATAAGACCCCGGCGCCGCCACC")
    parser.add_argument("--utr3", type=str,
                         default="GCTGGAGCCTCGGTGGCCTAGCTTCTTGCCCCTTGGGCCTCCCCCCAGCCCCTCCTCCCCTTCCTGCACCCGTACCCCCGTGGTCTTTGAATAAAGTCTGAGTGGGCGGCA")
    parser.add_argument("--sample", type=str, default="SRX10567366")
    parser.add_argument("--rnaCount", type=float, default=6.3)
    parser.add_argument("--device", type=str, default= "0")
    parser.add_argument("--weights", required=False)

    args = parser.parse_args()
    bs = 1
    Tensor = torch.Tensor
    model, device = load_model("../",args.weights)
    device = torch.device(f"cuda:{args.device}")
    model = model.to(device)


    mRNA = pd.read_parquet("VAE_1024.parquet")
    mRNA_array = Tensor(mRNA.loc[:,args.sample].values.reshape(1,-1)).float()

    with torch.no_grad():

        input_data = prepare_data(args.utr5,args.utr3,args.cds)

        out = model(
        Tensor(input_data["utr"]).transpose(2, 1).float().to(device),
        Tensor(input_data["cds"]).transpose(2, 1).float().to(device),
        mRNA_array.repeat(1, 1).to(device),
        Tensor([args.rnaCount]).reshape(1, 1).repeat(1, 1).to(device).float()
    )[:, 0].detach().cpu().numpy()


    print(get_orig_rpf(out))