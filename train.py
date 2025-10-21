import json
from util.module import CNN_Encoder_Model
import torch
import torch.optim as opt
import numpy as np
from torch import nn
import os
from util.load_data import UTRDataset_combinedenv
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset


class MODEL_MODULE:

    def __init__(self,checkpoint = None):
        model_config = "util/model_config.json"
        self.dataset_dir = "training_rawdata"

        self.model_config = json.load(open(model_config, "r"))

        self.predictor, self.device = self.model_from_config()
        hyper_use = self.model_config["training"]["hyper_use"]
        self.optimizer = opt.AdamW(params=self.predictor.parameters(),lr=1e-3,
                                   weight_decay=1e-2)
        if checkpoint:
            checkpoint = torch.load(checkpoint,map_location= "cuda:0")
            state_dict = checkpoint["model_state_dict"]
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.predictor.load_state_dict(new_state_dict)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler = opt.lr_scheduler.MultiStepLR(self.optimizer, [10], gamma=0.1, last_epoch=checkpoint['epoch'] )#-1
            self.cur_epoch = checkpoint['epoch'] + 1
        else:
            self.scheduler = opt.lr_scheduler.MultiStepLR(self.optimizer, [10], gamma=0.1, last_epoch=-1)
            self.cur_epoch = 0


        self.loss_fun = nn.SmoothL1Loss()
        self.best_epoch = 0
        self.batch_size = 256
        self.epoch_r2_df_init = True
        self.model_save_dir = "./"
        self.epoch_save_dir = f"epochs"
        os.makedirs(self.epoch_save_dir, exist_ok=True)
        self.model_description = self.model_config["training"]["model_description"]
        self.best_val_r2 = 0


    def model_from_config(self):
        inputshape = {'gene': 1, 'reg': (4, 5000), 'cds': (4, 4500), 'trans': (1024,), 'rna': (1,), 'expression': [1]}
        model = CNN_Encoder_Model(model_config=self.model_config, input_shapes=inputshape)
        device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        model.to(device)
        return model, device

    def train_model(self, dataset):
        self.predictor.train()
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for inputList in tqdm(dataloader):
            self.optimizer.zero_grad()  # 清空梯度
            regSequence = inputList["utr"].transpose(2, 1).to(self.device).float()
            cdsSequence = inputList["cds"].transpose(2, 1).to(self.device).float()
            rnaCount = inputList["RNA"].float().to(self.device).reshape(-1, 1)
            rpfLabel = inputList["RPF"].reshape(-1, 1).float().to(self.device)
            transArray_vae = inputList["mRNA_vae"].float().to(self.device)

            predictResult = self.predictor(regSequence, cdsSequence, transArray_vae, rnaCount)
            loss = self.loss_fun(predictResult[:, [0]].to(self.device), rpfLabel)
            loss.backward()

            grad_norm = sum(
                [param.grad.pow(2).sum() for param in self.predictor.parameters() if param.grad is not None]).pow(1 / 2)
            nn.utils.clip_grad_norm_(parameters=self.predictor.parameters(), max_norm=0.5, norm_type=2)
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.scheduler.step()
        print("")
        print("Epoch {}\n--------------".format(self.cur_epoch))
        self.valid()
        self.cur_epoch += 1

    def run_predict(self, dataloader, species):
        label_pred_dict = {"rpf": {"label": np.array([]), "preds": np.array([])}}
        with torch.no_grad():
            for inputList in tqdm(dataloader):
                regSequence = inputList["utr"].transpose(2, 1).to(self.device).float()
                cdsSequence = inputList["cds"].transpose(2, 1).to(self.device).float()
                rnaCount = inputList["RNA"].float().to(self.device).reshape(-1, 1)
                rpfLabel = inputList["RPF"].reshape(-1, 1).float().to(self.device)
                transArray_vae = inputList["mRNA_vae"].float().to(self.device)
                predictResult = self.predictor(regSequence, cdsSequence, transArray_vae, rnaCount)
                rpf_preds = predictResult.data.cpu().numpy()[:, 0]
                rpf_label = rpfLabel.data.cpu().numpy()[:, 0]

                label_pred_dict["rpf"]["label"] = np.append(label_pred_dict["rpf"]["label"], rpf_label)
                label_pred_dict["rpf"]["preds"] = np.append(label_pred_dict["rpf"]["preds"], rpf_preds)

        return label_pred_dict

    def save_epoch_model(self):
        torch.save(
            {"epoch": self.cur_epoch, "model_state_dict": self.predictor.state_dict(),},
            os.path.join(self.epoch_save_dir, "epochs_{}.p".format(self.cur_epoch)),
        )

    def valid(self, ):
        self.predictor.eval()

        dataset_human = UTRDataset_combinedenv(species="human", sample="valid", gene="valid", dataset_dir=self.dataset_dir)
        dataloader_human = DataLoader(dataset_human, batch_size=128, shuffle=True)
        dataset_mouse = UTRDataset_combinedenv(species="mouse", sample="valid", gene="valid", dataset_dir=self.dataset_dir)
        dataloader_mouse = DataLoader(dataset_mouse, batch_size=128, shuffle=True)
        dataset_yeast = UTRDataset_combinedenv(species="yeast", sample="valid", gene="valid", dataset_dir=self.dataset_dir)
        dataloader_yeast = DataLoader(dataset_yeast, batch_size=128, shuffle=True)
        dataset_chicken = UTRDataset_combinedenv(species="chicken", sample="valid", gene="valid", dataset_dir=self.dataset_dir)
        dataloader_chicken = DataLoader(dataset_chicken, batch_size=128, shuffle=True)
        dataset_atla = UTRDataset_combinedenv(species="A.thaliana", sample="valid", gene="valid", dataset_dir=self.dataset_dir)
        dataloader_atla = DataLoader(dataset_atla, batch_size=128, shuffle=True)
        dataset_elegans = UTRDataset_combinedenv(species="C.elegans", sample="valid", gene="valid", dataset_dir=self.dataset_dir)
        dataloader_elegans = DataLoader(dataset_elegans, batch_size=128, shuffle=True)
        
        res_human = self.run_predict(dataloader_human, "human")
        res_mouse = self.run_predict(dataloader_mouse, "mouse")
        res_yeast = self.run_predict(dataloader_yeast, "yeast")
        res_chicken = self.run_predict(dataloader_chicken, "chicken")
        res_atla = self.run_predict(dataloader_atla, "A.thaliana")
        res_elegans = self.run_predict(dataloader_elegans, "C.elegans")
        from scipy.stats import pearsonr
        rpf_pcc_human = pearsonr(res_human["rpf"]["label"], res_human["rpf"]["preds"]).statistic
        rpf_pcc_mouse = pearsonr(res_mouse["rpf"]["label"], res_mouse["rpf"]["preds"]).statistic
        rpf_pcc_chicken = pearsonr(res_chicken["rpf"]["label"], res_chicken["rpf"]["preds"]).statistic
        rpf_pcc_yeast = pearsonr(res_yeast["rpf"]["label"], res_yeast["rpf"]["preds"]).statistic
        rpf_pcc_atla = pearsonr(res_atla["rpf"]["label"], res_atla["rpf"]["preds"]).statistic
        rpf_pcc_elegans = pearsonr(res_elegans["rpf"]["label"], res_elegans["rpf"]["preds"]).statistic

        print(f"current epoch {self.cur_epoch}")
        print(f"valid sample, valid gene_human {rpf_pcc_human}")
        print(f"valid sample, valid gene_mouse {rpf_pcc_mouse}")
        print(f"valid sample, valid gene_yeast {rpf_pcc_yeast}")
        print(f"valid sample, valid gene_chicken {rpf_pcc_chicken}")
        print(f"valid sample, valid gene_A.thaliana {rpf_pcc_atla}")
        print(f"valid sample, valid gene_C.elegans {rpf_pcc_elegans}")

        self.save_epoch_model()


def r2_score_func(y_true, y_pred):
    a = np.square(y_pred - y_true)
    b = np.sum(a)
    c = np.mean(y_true)
    d = np.square(y_true - c)
    e = np.sum(d)
    f = 1 - b / (e + 1e-12)
    return f


def main(epochs,checkpoint):
    test = MODEL_MODULE(checkpoint)
    dataset_dir = "training_rawdata"
    dataset_human = UTRDataset_combinedenv(species = "human", sample="train", gene="train", dataset_dir=dataset_dir)
    dataset_mouse = UTRDataset_combinedenv(species = "mouse", sample="train", gene="train", dataset_dir=dataset_dir)
    dataset_yeast = UTRDataset_combinedenv(species = "yeast", sample="train", gene="train", dataset_dir=dataset_dir)
    dataset_chicken = UTRDataset_combinedenv(species = "chicken", sample="train", gene="train", dataset_dir=dataset_dir)
    dataset_atla = UTRDataset_combinedenv(species = "A.thaliana", sample="train", gene="train", dataset_dir=dataset_dir)
    dataset_elegans  = UTRDataset_combinedenv(species = "C.elegans", sample="train", gene="train", dataset_dir=dataset_dir)
    dataset = ConcatDataset([dataset_human, dataset_mouse, dataset_yeast, dataset_chicken, dataset_atla, dataset_elegans])
    for epoch in range(epochs):
        test.train_model(dataset)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=False,default = 30)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    seed = 42
    main(args.epochs,args.checkpoint)
