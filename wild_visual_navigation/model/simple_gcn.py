#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, knn_graph
from wild_visual_navigation.utils import Data

import rospy


class SimpleGCN(torch.nn.Module):
    # def __init__(self, input_size: int, reconstruction: bool, hidden_sizes=[64, 32, 1]):
    def __init__(self, input_size: int, reconstruction: bool, hidden_sizes=[64, 32]):
        super(SimpleGCN, self).__init__()

        self.reconstruction = reconstruction
        self.layers = torch.nn.ModuleList()
        # 共通の脳（GCNによる特徴抽出）
        inp = input_size
        for h in hidden_sizes:
            self.layers.append(GCNConv(inp, h))
            inp = h
        # 指標予測ヘッド: 5つの指標を個別に予測
        self.metric_head = torch.nn.Linear(inp, 5)
        # 重み予測ヘッド: どの指標を重視するか
        self.weight_head = torch.nn.Linear(inp, 5)
        # 再構築用ヘッド（既存機能を維持）
        if reconstruction:
            self.reco_head = torch.nn.Linear(inp, input_size)
        # self.layers = []
        # self.nr_sigmoid_layers = hidden_sizes[-1]
        # inp = input_size
        # for j, h in enumerate(hidden_sizes):
        #     if reconstruction and j == len(hidden_sizes) - 1:
        #         h += input_size

        #     self.layers.append(GCNConv(inp, h))
        #     inp = h

        # # self.layers = torch.nn.ModuleList(self.layers)
        # self.layers = torch.nn.Sequential(*self.layers)

    def forward(self, data: Data) -> torch.tensor:
        x = data.x
        num_nodes = x.size(0)
        side = int(num_nodes**0.5)
        if side * side == num_nodes:
            # 2次元インデックスの作成
            idx = torch.arange(num_nodes, device=x.device).view(side, side)
            # 横方向の接続
            edge_h_1 = torch.stack([idx[:, :-1].reshape(-1), idx[:, 1:].reshape(-1)], dim=0)
            edge_h_2 = torch.stack([idx[:, 1:].reshape(-1), idx[:, :-1].reshape(-1)], dim=0)
            # 縦方向の接続
            edge_v_1 = torch.stack([idx[:-1, :].reshape(-1), idx[1:, :].reshape(-1)], dim=0)
            edge_v_2 = torch.stack([idx[1:, :].reshape(-1), idx[:-1, :].reshape(-1)], dim=0)
            # 全て結合して edge_index とする
            edge_index = torch.cat([edge_h_1, edge_h_2, edge_v_1, edge_v_2], dim=1)
            rospy.loginfo(f"CC: Created Grid Edges for {side}x{side}")
        else:
            edge_index = getattr(data, 'edge_index', None)
            rospy.loginfo("CC: Non-square nodes, using original edge_index")

        # x, edge_index = data.x, data.edge_index

        # for j, layer in enumerate(self.layers):
        #     if edge_index is not None and edge_index.numel() > 0:
        #         x = layer(x, edge_index)
        #     else:
        #         x = layer.lin(x)
                
        #     if j < len(self.layers) - 1:
        #         x = F.relu(x)
        # for j, layer in enumerate(self.layers):
        #     if j != len(self.layers) - 1:
        #         x = F.relu(layer(x, edge_index))
        #     else:
        #         x = layer(x, edge_index)

        # x = F.dropout(x, training=self.training)

        # x[:, : self.nr_sigmoid_layers] = torch.sigmoid(x[:, : self.nr_sigmoid_layers])
        # return x

        for layer in self.layers:
            if edge_index is not None and edge_index.numel() > 0:
                mask = (edge_index[0] < x.size(0)) & (edge_index[1] < x.size(0))
                active_edge_index = edge_index[:, mask] # edge_index -> active_edge_index as number of nodes in mask

                num_nodes = x.size(0)
                num_edges = active_edge_index.size(1)
                # rospy.loginfo(f"GCN Stats: Nodes={num_nodes}, Valid Edges={num_edges}")

                if active_edge_index.numel() > 0: # index number between nodes < number of nodes -> GCN
                    x = F.relu(layer(x, active_edge_index))
                    # rospy.loginfo(f"DEBUG: Layer is acting as GCN (using edge_index)")
                else:
                    x = F.relu(layer.lin(x)) # -> MLP
                    # rospy.loginfo(f"DEBUG: Layer fallback to MLP (index out of bounds)")
            else:
                x = F.relu(layer.lin(x)) # -> MLP
                # rospy.loginfo(f"DEBUG: Layer is acting as MLP (no edges)")

        features = x # GCNで抽出された特徴量
        # 1. 各指標を予測 (Sigmoidで 0.0~1.0)
        pred_5_metrics = torch.sigmoid(self.metric_head(features))
        # 2. 各指標の重みを予測 (Softmaxで合計 1.0)
        T = 0.7
        weights = torch.softmax(self.weight_head(features) / T, dim=1)
        # 3. 最終的な1次元スコア（重み付き和）
        final_score = torch.sum(pred_5_metrics * weights, dim=1, keepdim=True)
        output_list = [final_score, weights, pred_5_metrics]
        if self.reconstruction:
            output_list.append(self.reco_head(features))
        res = torch.cat(output_list, dim=1)
        return res