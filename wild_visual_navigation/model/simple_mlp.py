#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from wild_visual_navigation.utils import Data
import torch

class SimpleMLP(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 64,
        hidden_sizes: [int] = [255],
        reconstruction: bool = False,
    ):
        super(SimpleMLP, self).__init__()

        self.reconstruction = reconstruction
        # 共通の脳（路面認識）
        self.shared_layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_sizes[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            torch.nn.ReLU()
        )
        # 指標予測ヘッド: 5つの指標を個別に予測
        self.metric_head = torch.nn.Linear(hidden_sizes[1], 5)
        # 重み予測ヘッド: どの指標を重視するか(路面状況に応じて変化)
        self.weight_head = torch.nn.Linear(hidden_sizes[1], 5)
        # 再構築用（既存機能を維持）
        if reconstruction:
            self.reco_head = torch.nn.Linear(hidden_sizes[1], input_size)
        # layers = []
        # self.nr_sigmoid_layers = hidden_sizes[-1]

        # if reconstruction:
        #     hidden_sizes[-1] = hidden_sizes[-1] + input_size

        # for hs in hidden_sizes[:-1]:
        #     layers.append(torch.nn.Linear(input_size, hs))
        #     layers.append(torch.nn.ReLU())
        #     input_size = hs
        # layers.append(torch.nn.Linear(input_size, hidden_sizes[-1]))

        # self.layers = torch.nn.Sequential(*layers)
        # self.output_features = hidden_sizes[-1]

    def forward(self, data: Data) -> torch.Tensor:
        # # dataがDataクラスのインスタンスではなく、直接Tensorで渡された場合にも対応させる
        # if hasattr(data, 'x'):
        #     x = data.x
        #     # x = data.x.contiguous()
        # else:
        #     x = data
        # # 2次元（バッチサイズ, 特徴量）であることを確認する処理を入れるとより安全
        # if x.dim() == 1:
        #     x = x.unsqueeze(0)
        # x = self.layers(x)
        # # 出力のスライス処理
        # if x.shape[1] >= self.nr_sigmoid_layers:
        #     x[:, : self.nr_sigmoid_layers] = torch.sigmoid(x[:, : self.nr_sigmoid_layers])
        
        x = data.x
        # # Checked data is correctly memory aligned and can be reshaped
        # # If you change something in the dataloader make sure this is still working ↑
        # x = self.layers(x)
        # x[:, : self.nr_sigmoid_layers] = torch.sigmoid(x[:, : self.nr_sigmoid_layers])
        # return x
        features = self.shared_layers(x)
        # 1. 各指標を予測 (Sigmoidで 0.0~1.0)
        pred_5_metrics = torch.sigmoid(self.metric_head(features)) # [Batch, 5]
        # 2. 各指標の重みを予測 (Softmaxで合計 1.0)
        T = 1.0  # 標準値1.0より大きいと滑らかになり、小さいと尖る
        weights = torch.softmax(self.weight_head(features) / T, dim=1)
        # weights = torch.softmax(self.weight_head(features), dim=1) # [Batch, 5]
        # 3. 最終的な1次元スコア（重み付き和）
        final_score = torch.sum(pred_5_metrics * weights, dim=1, keepdim=True) # [Batch, 1]
        if self.reconstruction:
            reco = self.reco_head(features)
            # 形式を合わせる: [最終スコア(1), 指標予測(5), 再構築(384)]
            return torch.cat([final_score, weights, pred_5_metrics, reco], dim=1)
        return torch.cat([final_score, weights, pred_5_metrics], dim=1)


class DoubleMLP(torch.nn.Module):
    def __init__(self, input_size: int = 64, hidden_sizes: [int] = [255]):
        super(DoubleMLP, self).__init__()
        self.nr_sigmoid_layers = hidden_sizes[-1]
        self.networks = []
        for network_last_layer in [hidden_sizes[-1], input_size]:
            layers = []
            inter_size = input_size
            for hs in hidden_sizes[:-1]:
                layers.append(torch.nn.Linear(inter_size, hs))
                layers.append(torch.nn.ReLU())
                inter_size = hs

            layers.append(torch.nn.Linear(inter_size, network_last_layer))

            self.networks.append(torch.nn.Sequential(*layers))
        self.networks = torch.nn.ModuleList(self.networks)
        self.output_features = hidden_sizes[-1] + input_size

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        # Checked data is correctly memory aligned and can be reshaped
        # If you change something in the dataloader make sure this is still working
        x1 = torch.sigmoid(self.networks[0](x))
        x2 = self.networks[1](x)
        return torch.cat([x1, x2], dim=1)
