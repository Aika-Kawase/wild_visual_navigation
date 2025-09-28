#
# Copyright (c) 2022-2024, ETH Zurich, Jonas Frey, Matias Mattamala.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
from typing import List
from typing_extensions import Self
import torch


class Data:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Batch:
    def __init__(self):
        pass

    @classmethod
    def from_data_list(cls, list_of_data: List[Data]) -> 'Batch':
        if len(list_of_data) == 0:
            return None

        # ! 新しいBatchインスタンスを作成
        batch_instance = cls() 
        
        base = ["x"]
        tensors_to_concatenate = [
            k for k in dir(list_of_data[0]) if k[0] != "_" and getattr(list_of_data[0], k) is not None and not k in base
        ]
        base = base + tensors_to_concatenate

        # Node index offsets (ptr and batch should be calculated first)
        running = 0
        ptrs = [running]
        batches = []
        for j, data in enumerate(list_of_data):
            # テンソル 'x' の shape[0] を使ってノード数を計算
            x_tensor = getattr(data, "x")
            if x_tensor is not None:
                num_nodes = x_tensor.shape[0]
                running += num_nodes
                ptrs.append(running)
                batches += [j] * int(num_nodes)

        # ptr, batchをインスタンス変数として設定
        batch_instance.ptr = torch.tensor(ptrs, dtype=torch.long)
        batch_instance.batch = torch.tensor(batches, dtype=torch.long)
        
        # ! 各属性をインスタンスに追加し、値を設定
        for k in base:
            ls = []
            for j, data in enumerate(list_of_data):
                data_attr = getattr(data, k)
                
                if k == "edge_index":
                    # edge_index はノードオフセット (ptr) を加える必要がある
                    # PyGのロジックを模倣して、オフセットを適用
                    offset = batch_instance.ptr[j]
                    ls.append(data_attr + offset) # offsetはptr[j]
                
                else:
                    # その他の属性は単純に連結 (x, y, y_validなど)
                    ls.append(data_attr)

            if k == "edge_index":
                # edge_index はここで設定 (dim=1 or -1)
                setattr(batch_instance, k, torch.cat(ls, dim=-1))
            else:
                # x やその他の属性を連結して設定
                setattr(batch_instance, k, torch.cat(ls, dim=0))

        # 最後の属性設定
        batch_instance.ba = batch_instance.x.shape[0] # ノードの総数
        
        # ! インスタンスを返す
        return batch_instance

    # def from_data_list(cls, list_of_data: List[Data]) -> Self:
    #     if len(list_of_data) == 0:
    #         return None

    #     base = ["x"]

    #     tensors_to_concatenate = [
    #         k for k in dir(list_of_data[0]) if k[0] != "_" and getattr(list_of_data[0], k) is not None and not k in base
    #     ]
    #     base = base + tensors_to_concatenate

    #     for k in base:
    #         if k == "edge_index":
    #             ls = []
    #             for j, data in enumerate(list_of_data):
    #                 ls.append(getattr(data, k) + cls.ptr[j])

    #             cls.edge_index = torch.cat(ls, dim=-1)
    #         else:

    #             if k == "x":
    #                 running = 0
    #                 ptrs = [running]
    #                 batches = []

    #                 for j, data in enumerate(list_of_data):
    #                     running = running + getattr(data, k).shape[0]
    #                     ptrs.append(running)
    #                     batches += [j] * int(getattr(data, k).shape[0])

    #                 cls.ptr = torch.tensor(ptrs, dtype=torch.long)
    #                 cls.batch = torch.tensor(batches, dtype=torch.long)

    #             setattr(cls, k, torch.cat([getattr(data, k) for data in list_of_data], dim=0))

    #     cls.ba = cls.x.shape[0]
    #     return cls


if __name__ == "__main__":
    from torch_geometric.data import Data as DataTorchGeometric
    from torch_geometric.data import Batch as BatchTorchGeometric

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    data_tg1 = DataTorchGeometric(x=x, edge_index=edge_index)
    data1 = Data(x=x, edge_index=edge_index)
    edge_index2 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x2 = torch.tensor([[-1], [12], [1]], dtype=torch.float)
    data_tg2 = DataTorchGeometric(x=x2, edge_index=edge_index2)
    data2 = Data(x=x2, edge_index=edge_index2)
    batch = BatchTorchGeometric.from_data_list([data_tg1, data_tg2])
