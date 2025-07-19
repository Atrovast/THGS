"""
This script performs two main steps:
1. Constructs the Gaussian Centroids Adjacency Graph from the input Gaussian scene.
2. Partitions the Adjacency Graph into superpoints through a graph cut algorithm.
Please refer to the launcher script for running this script.
"""
import os
import sys
import torch
import numpy as np
from plyfile import PlyData
import torch.nn as nn

from utils.sh_utils import eval_sh
from utils.general_utils import build_rotation

# Add the project's files to the python path
sys.path.append('ext/')
from spt.data import Data
from spt.utils.color import to_float_rgb
from spt.dependencies.FRNN import frnn
from spt.transforms import NAGRemoveKeys, instantiate_datamodule_transforms
from spt.utils import init_config



@torch.no_grad()
def load_ply(path, pos_center=torch.tensor([0, 0, 0]), semantic=None, use_normal=False):
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(3 + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (3 + 1) ** 2 - 1))

    xyz = torch.tensor(xyz, dtype=torch.float, device="cuda")
    features_dc = torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
    features_rest = torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()

    if use_normal:
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        R = build_rotation(rotation)
        normal = R[:, 2, :]

    pos_center = pos_center.to(xyz.device)
    get_features = torch.cat((features_dc, features_rest), dim=1)
    shs_view = get_features.transpose(1, 2).view(-1, 3, (3+1)**2)
    dir_pp = (xyz - pos_center.repeat(get_features.shape[0], 1))
    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    sh2rgb = eval_sh(3, shs_view, dir_pp_normalized)
    colors_precomp = torch.clamp(sh2rgb + 0.5, 0.0, 1.0)

    data = Data()
    data.pos = xyz
    data.rgb = colors_precomp
    if semantic is not None:
        data.semantic = torch.tensor(semantic, dtype=torch.float)
    if use_normal:
        data.normal = normal
        if args.align_normal:
            # Align normals to point up
            data.normal[data.normal[:, 2] < 0] *= -1

    return data


def partition(data, filepath, transforms_dict, graph_cut=None, **kwargs):

    if graph_cut is not None:
        neibor = torch.load(graph_cut)
        data.neighbor_index = neibor['neighbors'].cuda()
        data.neighbor_distance = neibor['distances'].cuda()

    # Pre-transforms
    nag = transforms_dict['cut_transform'](data)

    # Simulate the behavior of the dataset's I/O behavior with only
    # `point_load_keys` and `segment_load_keys` loaded from disk
    nag = NAGRemoveKeys(level=0, keys=[k for k in nag[0].keys if k not in cfg.datamodule.point_load_keys])(nag)
    nag = NAGRemoveKeys(level='1+', keys=[k for k in nag[1].keys if k not in cfg.datamodule.segment_load_keys])(nag)
    nag = nag.cuda()

    return nag


def pre_knn(data, filepath, transforms_dict, **kwargs):
    # 展示一些信息
    xyz = data.pos
    xyz_q = xyz.unsqueeze(0)

    # cut before point_cloud
    scene_path = filepath.split('point_cloud')[0]

    # Pre-transforms
    nag = transforms_dict['knn_transform'](data)

    if (nag['neighbor_index'] < 0).any():
        q = (nag['neighbor_index'] < 0).sum()
        print(f'Negative neighbor index found, {q} times')
    torch.save({
        'neighbors': nag['neighbor_index'],
        'distances': nag['neighbor_distance'],
    }, os.path.join(scene_path, 'neighbor.pt'))

    return nag


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", '-m', type=str)
    parser.add_argument("--iteration", '-i', type=int, default=30000)
    parser.add_argument("--graph_cut", '-k', type=str, default=None)
    parser.add_argument("--centered", '-c', action='store_true')
    parser.add_argument("--pcp_regularization", type=float, nargs='+')
    parser.add_argument("--pcp_spatial_weight", type=float, nargs='+')
    parser.add_argument("--verbose", '-v', action='store_true')
    parser.add_argument("--align_normal", '-a', action='store_true')
    args, extras = parser.parse_known_args()
    scene_path = args.model_path
    iteration = args.iteration

    fpath = os.path.join(scene_path, f'point_cloud/iteration_{iteration}/point_cloud.ply')
    data = load_ply(fpath, 
                    pos_center=torch.tensor([0, -1, 3.5]) if not args.centered else torch.tensor([0, 0, 0]),
                    semantic=None,
                    use_normal=True)

    cfg = init_config(overrides=['experiment=semantic/scannet'])
    if args.pcp_regularization is not None and args.pcp_spatial_weight is not None:
        cfg.datamodule.pcp_regularization = args.pcp_regularization
        cfg.datamodule.pcp_spatial_weight = args.pcp_spatial_weight
    transforms_dict = instantiate_datamodule_transforms(cfg.datamodule)

    if args.graph_cut is not None:
        # count time
        import time
        start = time.time()
        nag = partition(data, filepath=fpath, 
                        transforms_dict=transforms_dict,
                        graph_cut=os.path.join(scene_path, args.graph_cut) if args.graph_cut is not None else None,)
        print(f"Time: {time.time()-start:.2f}s")
        print(nag)

        torch.save(nag.get_super_index(1, 0), os.path.join(scene_path, 'nag-l1.pt'))
        # save
        if args.verbose:
            torch.save(nag.get_super_index(2, 0), os.path.join(scene_path, 'nag-l2.pt'))
            torch.save(nag.get_super_index(3, 0), os.path.join(scene_path, 'nag-l3.pt'))
    else:
        # count time
        import time
        start = time.time()
        nag = pre_knn(data, fpath, transforms_dict)
        print(f"Time: {time.time()-start:.2f}s")

