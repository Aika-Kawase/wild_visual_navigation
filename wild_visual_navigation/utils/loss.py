#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from wild_visual_navigation.utils import ConfidenceGenerator

import torch.nn.functional as F
from wild_visual_navigation.utils import Data

import torch
from typing import Optional
from torch import nn


class AnomalyLoss(nn.Module):
    def __init__(
        self,
        confidence_std_factor: float,
        method: str,
        log_enabled: bool,
        log_folder: str,
    ):
        super(AnomalyLoss, self).__init__()

        self._confidence_generator = ConfidenceGenerator(
            std_factor=confidence_std_factor, method=method, log_enabled=log_enabled, log_folder=log_folder
        )

    def forward(
        self,
        graph: Optional[Data],
        res: dict,
        update_generator: bool = True,
        step: int = 0,
        log_step: bool = False,
    ):
        loss_aux = {}
        loss_aux["loss_trav"] = torch.tensor([0.0])
        loss_aux["loss_reco"] = torch.tensor([0.0])

        losses = res["logprob"].sum(1) + res["log_det"]  # Sum over all channels, resulting in h*w output dimensions

        if update_generator:
            confidence = self._confidence_generator.update(
                x=-losses.clone().detach(), x_positive=-losses.clone().detach(), step=step
            )

        loss_aux["confidence"] = confidence

        return -torch.mean(losses), loss_aux, confidence

    def update_node_confidence(self, node):
        node.confidence = 0


class TraversabilityLoss(nn.Module):
    def __init__(
        self,
        w_trav: float,
        w_reco: float,
        w_temp: float,
        anomaly_balanced: bool,
        model: nn.Module,
        method: str,
        confidence_std_factor: float,
        log_enabled: bool,
        log_folder: str,
        trav_cross_entropy=False,
        nr_channel_reco: int = 0,
    ):
        # TODO remove trav_cross_entropy default param when running in online mode
        super(TraversabilityLoss, self).__init__()
        self._w_trav = w_trav

        self._w_reco = w_reco
        self._w_temp = w_temp
        self._model = model
        self._anomaly_balanced = anomaly_balanced
        self._trav_cross_entropy = trav_cross_entropy
        if self._trav_cross_entropy:
            self._trav_loss_func = F.binary_cross_entropy
        else:
            self._trav_loss_func = F.mse_loss

        self._confidence_generator = ConfidenceGenerator(
            std_factor=confidence_std_factor, method=method, log_enabled=log_enabled, log_folder=log_folder
        )
        self._nr_channel_reco = nr_channel_reco

    def reset(self):
        if self._anomaly_balanced:
            self._confidence_generator.reset()

    def forward(
        self,
        graph: Data,
        res: torch.Tensor,
        update_generator: bool = True,
        step: int = 0,
        log_step: bool = False,
    ):
        # 予測値の取得 (resが1次元ならそのまま、多次元なら先頭を使用)
        res_trav_pred = res if res.shape[1] == 1 else res[:, 0:1]

        # Compute reconstruction loss
        nr_channel_reco = graph.x.shape[1] # shape [N,1]
        # loss_reco = F.mse_loss(res[:, -nr_channel_reco:], graph.x, reduction="none").mean(dim=1)

        # 正解ラベルの整形 (trainで1次元化されているので [N, 1] にするだけ)
        # 警告を消すために view(-1, 1) で次元を明示的に合わせる
        label = graph.y.view(-1, 1).to(res.device)
        # 4. 走行性損失の計算 (MSE)
        loss_trav_raw = self._trav_loss_func(res_trav_pred, label, reduction="none")
        if loss_trav_raw.dim() > 1:
            loss_trav_raw = loss_trav_raw.mean(dim=1)
        # 再構築損失 (Reconstruction Loss) の処理
        # train から 1次元しか渡されていない場合、ここでの計算はスキップまたは 0 にする
        if res.shape[1] > nr_channel_reco and nr_channel_reco > 0:
            # フルサイズの res が渡された場合のみ計算
            res_reco_pred = res[:, -nr_channel_reco:]
            loss_reco = F.mse_loss(res_reco_pred, graph.x, reduction="none").mean(dim=1)
        else:
            # 1次元渡しの場合、再構築損失は計算できないため 0 を返す
            # (注意: これにより Anomaly Detection 機能は機能しなくなります)
            loss_reco = torch.zeros_like(loss_trav_raw)
        # # res は [final_score(1), weights(5), metrics(5), reco(384)] の構成
        # # 走行性損失（Traversability Loss）には先頭の1次元目のみを使用する
        # res_trav_pred = res[:, 0:1] 
        # # 再構築用データは末尾から nr_channel_reco 分を取り出す
        # if nr_channel_reco > 0:
        #     res_reco_pred = res[:, -nr_channel_reco:]
        # # 走行性損失の計算
        # # graph.y が 5次元 [N, 5] の場合、平均値 [N, 1] を作成してターゲットにする
        # if graph.y.dim() > 1 and graph.y.shape[1] > 1:
        #     label = graph.y.mean(dim=1, keepdim=True).to(res.device)
        # else:
        #     label = graph.y.view(-1, 1).to(res.device)
        # loss_trav_raw = self._trav_loss_func(res_trav_pred, label, reduction="none").mean(dim=1)
        # if nr_channel_reco > 0:
        #     res_reco_pred = res[:, -nr_channel_reco:]
        #     loss_reco = F.mse_loss(res_reco_pred, graph.x, reduction="none").mean(dim=1)
        # else:
        #     # loss_reco = torch.tensor(0.0, device=res.device)
        #     loss_reco = torch.zeros_like(loss_trav_raw, device=res.device)

        with torch.no_grad():
            if update_generator:
                confidence = self._confidence_generator.update(
                    x=loss_reco,
                    x_positive=loss_reco[graph.y_valid],
                    step=step,
                    log_step=log_step,
                )
            else:
                confidence = self._confidence_generator.inference_without_update(x=loss_reco)

        # ele = graph.y_valid.shape[0]  # 400 #
        # selector = torch.zeros_like(graph.y_valid)
        # selector[:ele] = 1
        # loss_trav_raw_labeled = loss_trav_raw[graph.y_valid * selector]
        # loss_trav_raw_not_labeled = loss_trav_raw[~graph.y_valid * selector]

        # # Scale the loss
        # loss_trav_raw_not_labeled_weighted = loss_trav_raw_not_labeled * (1 - confidence)[~graph.y_valid * selector]

        # 追記
        labeled_mask = graph.y_valid.bool()
        unlabeled_mask = ~labeled_mask
        loss_trav_raw_labeled = loss_trav_raw[labeled_mask]
        loss_trav_raw_unlabeled = loss_trav_raw[unlabeled_mask]
        # 最終的な Loss の統合
        loss_trav_mean = loss_trav_raw.mean()
        loss_reco_mean = loss_reco[labeled_mask].mean() if labeled_mask.any() else loss_reco.mean()

        # Scale the loss
        loss_trav_raw_unlabeled_weighted = loss_trav_raw_unlabeled * (1 - confidence[unlabeled_mask])

        # --- unlabeled weighting ---
        loss_trav_raw_not_labeled_weighted = torch.zeros_like(loss_trav_raw_unlabeled)
        if self._anomaly_balanced:
            loss_trav_confidence = (loss_trav_raw_not_labeled_weighted.sum() + loss_trav_raw_labeled.sum()) / (
                graph.y.shape[0]
            )
        else:
            # loss_trav_confidence = loss_trav_raw[selector].mean()
            loss_trav_confidence = loss_trav_raw.mean()

        loss_temp = torch.zeros_like(loss_trav_confidence)
        
        # # loss_reco_mean = loss_reco[graph.y_valid * selector].mean()
        loss_reco_mean = loss_reco[labeled_mask].mean() if labeled_mask.any() else loss_reco.mean()

        # Compute total loss
        loss = self._w_trav * loss_trav_confidence + self._w_reco * loss_reco_mean + self._w_temp * loss_temp

        # traversability_estimator.py 側で使うため、1次元の予測値を返す
        res_updated = res_trav_pred
        
        # res_updated = res
        return (
            loss,
            {
                "loss_reco": loss_reco_mean,
                "loss_trav": loss_trav_raw.mean(),
                "loss_temp": loss_temp.mean(),
                "loss_trav_confidence": loss_trav_confidence,
                "confidence": confidence,
            },
            res_updated,
        )

    def update_node_confidence(self, node):
        reco_loss = F.mse_loss(node.prediction[:, 1:], node.features, reduction="none").mean(dim=1)
        node.confidence = self._confidence_generator.inference_without_update(reco_loss)
