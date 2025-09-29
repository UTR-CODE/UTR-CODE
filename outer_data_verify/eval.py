import sys
sys.path.insert(0,"../")
from util.load_data import UTRDataset_combinedenv
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import json
from torch import nn
from util.module import CNN_Encoder_Model #多物种_vae
import random
import seqpro as sp


def load_model(path,weight=None):#此处改权重路径
    config = os.path.join(path, "util/model_config.json")

    model_config = json.load(open(config, "r"))
    inputshape = {'gene': 1, 'reg': (4, 5000), 'cds': (4, 4500), 'trans': (1024,), 'rna': (1,), 'expression': [1]}
    model = CNN_Encoder_Model(model_config=model_config, input_shapes=inputshape)
    device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.to(device)

    if not weight:
        path = os.path.join(path,"epochs")
        files = os.listdir(path)
        best_epoch = sorted(files, key = lambda x:int(x.split(".")[0].split("_")[-1]))[-1]

        checkpoint = torch.load(os.path.join(path, best_epoch), map_location=device)
    else:
        checkpoint = torch.load(weight,map_location=device)
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict,strict = False)
    model = model.eval()
    return model, device


from torch.utils.data import Dataset, DataLoader


class verify_dataset(Dataset):
    def __init__(self, file):
        data = pd.read_csv(file)
        data.columns = "utr5", "utr3", "cds","label"

        data["utr5"] = data["utr5"].fillna("").str.slice(-1000).str.pad(1000, "left", "N")
        data["utr3"] = data["utr3"].fillna("").str.slice(-4000).str.pad(4000, "right", "N")
        data["utr_sequence"] = data["utr5"].str.cat(data["utr3"], sep="")
        data["cds_sequence"] = data["cds"].str.slice(0,4500).str.pad(4500,"right","N")
        self.data = data.reset_index(drop=True)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        utr = self.data.loc[item, "utr_sequence"]
        cds = self.data.loc[item, "cds_sequence"]
        return {"utr":sp.ohe(utr, alphabet=sp.alphabets.DNA),
                "cds":sp.ohe(cds, alphabet=sp.alphabets.DNA)}

if __name__ == '__main__':
    import argparse
    from tqdm import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="..")
    parser.add_argument("--file", type=str)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--rna", type=int, default=6.3)
    parser.add_argument("--sample", type=str, default="SRX10567366")
    parser.add_argument("--weight", required=False)
    parser.add_argument("--output",)
    args = parser.parse_args()

    model, device = load_model(args.model_path,args.weight)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    dataset = verify_dataset(args.file)

    raw_df = pd.read_csv(args.file)

    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    mRNA = pd.read_parquet("VAE_1024.parquet")
    
    mRNA_array = Tensor(mRNA.loc[:,args.sample].values.reshape(1,-1)).float()
        

    predict_out = []
    with torch.no_grad():
        for item in tqdm(dataloader):
            utr = item["utr"].transpose(2, 1).float().to(device)
            cds = item["cds"].transpose(2, 1).float().to(device)
            rna = Tensor([args.rna]).reshape(1, 1).repeat(utr.shape[0],1)
            out = model(
                    utr,
                    cds,
                    mRNA_array.repeat(utr.shape[0],1),
                    rna
                )
            predict_out.append(out.data.cpu()[:,0])
        result = torch.concat(predict_out).cpu().detach().numpy()
        raw_df["pred"] = result

        print(f" pearson : ", raw_df.iloc[:, -2:].corr().iloc[0,1])
        print(f" spearman :",raw_df.iloc[:, -2:].corr(method = "spearman").iloc[0,1])
    raw_df.to_csv(args.output,index=False)

