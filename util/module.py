import numpy as np
import torch
from torch import nn
import warnings
warnings.filterwarnings("ignore")


class Transpose(nn.Module):
    def __init__(self):
        super(Transpose, self).__init__()

    def forward(self, x):
        return torch.transpose(x, 2, 1)


"""
-------------------------------------------
        RnaModel Class Construction
-------------------------------------------
"""


class CNN_Encoder_Model(nn.Module):
    r"""
    input shape must be: [batch_size, channel_size, sequence_length]
    """
    def __init__(self, model_config, input_shapes):
        super(CNN_Encoder_Model, self).__init__()
        self.model_config = model_config
        self.input_shapes = input_shapes
        print("input shapes is {}".format(self.input_shapes))
        hyper_use = model_config["training"]["hyper_use"]
        p = model_config[hyper_use]
        self.hyperParam = p
        self.motif_depth = 4
        self.bias = nn.Parameter(torch.zeros([1]))
        self.moving_bias = 0


        self.detectorDict = nn.ModuleDict()
        for k in ["reg", "cds"]:
            self.detectorDict[k] = nn.ModuleList()
            for _ in range(self.motif_depth):
                self.detectorDict[k].append(
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels=input_shapes[k][0],
                            out_channels=int(p["filters1"]),
                            kernel_size=int(p["kernel_size1"]),
                            stride=int(p["filters_stride1"]),
                            dilation=int(p["dilation1"]),
                        ),
                        nn.ReLU(),
                        nn.BatchNorm1d(int(p["filters1"])),
                    )
                )

        self.filterDict = nn.ModuleDict()
        for k in ["reg", "cds"]:
            self.filterDict[k] = nn.Sequential(
                nn.BatchNorm1d(int(p["filters1"])),
                nn.Dropout1d(float(p["dropout1"])),
                nn.MaxPool1d(
                    kernel_size=int(p["pool_size1"]),
                    stride=int(p["stride1"]),
                    padding=0,
                ),
            )

        self.encoderDict = nn.ModuleDict()
        for k in ["reg", "cds"]:
            self.encoderDict[k] = nn.Sequential(
                nn.Conv1d(
                    in_channels=int(p["filters1"]),
                    out_channels=int(p["filters3"]),
                    kernel_size=int(p["kernel_size3"]),
                    stride=int(p["filters_stride3"]),
                    dilation=int(p["dilation3"]),
                ),
                nn.ReLU(),
                nn.BatchNorm1d(int(p["filters3"])),
                nn.MaxPool1d(
                    kernel_size=int(p["pool_size3"]),
                    stride=int(p["stride3"]),
                    padding=0,
                ),
                Transpose(),
                nn.Dropout1d(float(p["dropout3"])),
                nn.Flatten(),
            )
        utr_flatten_size = self.conv_shape("reg")

        self.utr_mlp = nn.Sequential(
            nn.Linear(in_features=utr_flatten_size, out_features=int(p["dense4"])),
            nn.ReLU(),
            nn.BatchNorm1d(int(p["dense4"])),
            nn.Dropout(float(p["dropout4"])),
            nn.Linear(in_features=int(p["dense4"]), out_features=int(p["dense6"])),
            nn.ReLU(),
            nn.BatchNorm1d(int(p["dense6"])),
        )

        cds_flatten_size = self.conv_shape("cds")

        self.RPF_fc = nn.Sequential(
            nn.Linear(in_features=cds_flatten_size, out_features=int(p["dense5"])),
            nn.ReLU(),
            nn.BatchNorm1d(int(p["dense5"])),
            nn.Dropout(float(p["dropout5"])),
            nn.Linear(in_features=int(p["dense5"]), out_features=1),
        )

        self.attention_utr = nn.Sequential(

            nn.Linear(
                in_features=int(p["dense6"]),
                out_features=int(p["filters1"]) * self.motif_depth,
            ),
            nn.Tanh(),
        )

        self.attention_cds = nn.Sequential(
            nn.Linear(
                in_features=int(p["dense6"]) + 1,
                out_features=int(p["filters1"]) * self.motif_depth,
            ),
            nn.Tanh(),
        )

        self.mRNA_layer = nn.Sequential(
            nn.Linear(
                in_features=self.input_shapes["trans"][0], out_features=int(p["dense4"])
            ),
            nn.ReLU(),
            nn.BatchNorm1d(int(p["dense4"])),
            nn.Dropout(float(p["dropout4"])),
            nn.Linear(in_features=int(p["dense4"]), out_features=int(p["dense6"])),
            nn.ReLU(),
            nn.BatchNorm1d(int(p["dense6"])),
        )

    def motif_detection(self, sequence_input, a, seq_type: str):

        results = [
            motif_detector(sequence_input)
            for motif_detector in self.detectorDict[seq_type]
        ]
        avg_filter = torch.exp(a)
        features = torch.sum(torch.stack(results, 3) * avg_filter, 3) / torch.sum(
            avg_filter, 3
        )
        motif_result = self.filterDict[seq_type](features)
        return motif_result

    def conv_shape(self, seqType: str):
        x_input_1 = torch.zeros(1, *self.input_shapes[seqType])
        x_output = self.motif_detection(
            x_input_1, torch.zeros([1, 256, 1, self.motif_depth]), seqType
        )
        x_output = self.encoderDict[seqType](x_output)
        flatten_shape = int(np.prod(x_output.size()))
        return flatten_shape

    def forward(self, regSequence, cdsSequence, transArray, rnaCount):
        mRNA_features = self.mRNA_layer(transArray)
        attention_Reg = torch.reshape(
            self.attention_utr(mRNA_features),
            [-1, int(self.hyperParam["filters1"]), 1, self.motif_depth],
        )
        regOutput = self.motif_detection(regSequence, attention_Reg, "reg")
        regOutput = self.encoderDict["reg"](regOutput)
        regFeatures = self.utr_mlp(regOutput)
        attention_cds = torch.reshape(
            self.attention_cds(torch.concat([regFeatures, rnaCount], dim=1)),
            [-1, self.hyperParam["filters1"], 1, self.motif_depth],
        )
        cds_output = self.motif_detection(cdsSequence, attention_cds, "cds")
        cds_seq_features = self.encoderDict["cds"](cds_output)
        #self.new_bias = 0.99 * self.moving_bias + 0.01 * self.bias
        TE = self.RPF_fc(cds_seq_features)
        RPF = TE * rnaCount + self.bias
        #self.moving_bias = self.new_bias.detach()

        return torch.concat([RPF, TE], 1)
