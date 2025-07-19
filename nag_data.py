"""
This module defines the SemanticNAG class, which constructs a hierarchical superpoint graph with semantic features.
It builds upon the Nested Adjacency Graph (NAG) structure from the SPT (Superpoint Transformer) library (https://arxiv.org/abs/2306.08045).
"""
import torch
from typing import List
import sys
sys.path.append('ext/')
from spt.data import Data, NAG, Cluster
# from ext.spt.data import Data, NAG, Cluster

class SemanticNAG():
    def __init__(self, labels, feat):
        labels = [label.cuda() for label in labels]
        self.labels = labels # [4, N]
        self.nag = self.build_nag_from_multilevel_labels(labels)
        self.feat = feat
        self.gaussian_num = labels[0].shape[0]

    # sim in, gaussian_num * 1 out, indicating the 0-1 mask
    def get_related_gaussian(self, sim: List[torch.Tensor], topk: int = 1, level: int = -1) -> torch.Tensor:
        """
        sim: List of Tensors, each shape like superpoint
        topk: int, number of top similar points to consider
        level: int, choose certain level to get the related gaussian, or -1 to get all levels
        """
        assert len(sim) == len(self.labels) - 1, "Number of similarity matrices must match number of levels"
        if level == -1:
            lvls = [i for i in range(1, len(sim))]
        elif isinstance(level, list):
            lvls = [i - 1 for i in level]
        else:
            lvls = [level - 1]
        
        # 1. find the topk similar points in sim list, get tuple (level, sim_val, index)
        related_sp_lvl = []
        for i in lvls:
            sim_array = sim[i]
            sim_val, indices = torch.topk(sim_array, topk)
            related_sp_lvl.extend(list(zip([i+1] * topk, sim_val, indices)))
        # sort the list based on sim_val, descending order
        related_sp_lvl.sort(key=lambda x: x[1], reverse=True)
        # get the topk similar points
        related_sp_lvl = related_sp_lvl[:topk]
        
        # 2. get the related gaussian
        rel_gaussians = torch.zeros(self.gaussian_num, 1, dtype=torch.float32)
        for i, tup in enumerate(related_sp_lvl):
            level, _, index = tup
            lowest_idx = torch.where(self.labels[level] == index)[0]
            rel_gaussians[lowest_idx, 0] = 1
        return rel_gaussians
    
    def get_gs_by_mask(self, mask, level):
        """
        mask: N, boolean mask
        level: int, choose certain level to get the related gaussian
        """
        assert level > 0 and level <= len(self.labels), "Level out of range"
        # get the sub from NAG, iterative down to level 0
        label = self.labels[level]
        sps = label[mask]
        # get gaussian in the sps
        rel_gaussians = torch.zeros(self.gaussian_num, 1, dtype=torch.float32)
        for sp in sps:
            lowest_idx = torch.where(label == sp)[0]
            rel_gaussians[lowest_idx, 0] = 1
        return rel_gaussians

    
    @staticmethod
    def build_nag_from_multilevel_labels(labels: List[torch.Tensor]) -> NAG:
        """
        labels: List of Tensors, each shape = (N,), level-0 point to higher level cluster IDs
            e.g. [label_lvl0, label_lvl1, label_lvl2]
        pos / feat: Optional, only used at level 0
        """
        assert len(labels) >= 2, "At least two levels required to construct NAG"
        device = labels[0].device
        N = labels[0].shape[0]
        num_levels = len(labels)

        data_list = [Data(num_nodes=N, super_index=labels[0])]
        prev_sub = None

        for i in range(num_levels):
            data = Data()
            # 求每层的sub和上一层的super_index
            if i == 0:
                upper_labels = labels[i]
                lower_labels = torch.arange(N, device=device)
                sorted_upper_labels, perm = torch.sort(upper_labels)
                sorted_lower_labels = lower_labels[perm]
                num_clusters = int(sorted_upper_labels.max()) + 1
                cluster_sizes = torch.bincount(sorted_upper_labels, minlength=num_clusters)
                pointers = torch.zeros(num_clusters + 1, dtype=torch.long, device=device)
                pointers[1:] = torch.cumsum(cluster_sizes, dim=0)
                prev_sub = Cluster(pointers=pointers, points=sorted_lower_labels, dense=False)
            else:
                data = Data(num_nodes=labels[i-1].max().item() + 1)
                data.sub = prev_sub
                upper_labels = labels[i].long()
                lower_labels = labels[i-1].long()

                # 1. remove duplicates based on lower labels
                unique_lower_labels, inv = torch.unique(lower_labels, sorted=True, return_inverse=True)
                unique_upper_labels = torch.zeros_like(unique_lower_labels)
                unique_upper_labels[inv] = upper_labels
                # print('uni:', lower_labels, unique_lower_labels, unique_upper_labels, inv)

                data.super_index = unique_upper_labels
                # 2. sort based on upper labels, construct next sub
                sorted_upper_labels, perm = torch.sort(unique_upper_labels)
                sorted_lower_labels = unique_lower_labels[perm]
                # print("sort:", sorted_lower_labels, sorted_upper_labels)
                # upper label take down the change
                num_clusters = int(sorted_upper_labels.max()) + 1
                cluster_sizes = torch.bincount(sorted_upper_labels, minlength=num_clusters)
                pointers = torch.zeros(num_clusters + 1, dtype=torch.long, device=device)
                pointers[1:] = torch.cumsum(cluster_sizes, dim=0)
                # print('csr:', pointers, sorted_lower_labels)
                prev_sub = Cluster(pointers=pointers, points=sorted_lower_labels, dense=False)
                data_list.append(data)
        data_list.append(Data(sub=prev_sub, num_nodes=labels[-1].max().item() + 1))
        return NAG(data_list)

if __name__ == '__main__':
    lvl1 = torch.tensor([4, 5, 2, 3, 1, 5, 0, 3])
    lvl2 = torch.tensor([3, 4, 3, 2, 1, 4, 0, 2])
    lvl3 = torch.tensor([1, 2, 1, 1, 0, 2, 0, 1])
    nag = SemanticNAG.build_nag_from_multilevel_labels([lvl1, lvl2, lvl3])
    print(nag, nag[1].sub, 777)
    print(nag[3].sub[1].points)
    pt = 1
    for i in range(3, 0, -1):
        pt = nag[i].sub[pt].points
        print(55, pt)
    print(nag.get_super_index(1, 0))
    print(nag.get_super_index(2, 0))
    print(nag.get_super_index(3, 0))
    print(nag.get_super_index(2, 1))
    # print(from_super_index(lvl2, lvl1))
    # cls = Cluster(pointers=torch.tensor([0, 3, 5, 8]), points=torch.tensor([2, 0, 1, 3, 4, 5, 6, 7]))
    # print(cls, cls.pointers)
