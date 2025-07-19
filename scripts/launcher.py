import os
import time
import torch
from argparse import ArgumentParser
from omegaconf import OmegaConf


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--script', '-f', type=str, choices=['graph_weight', 'merge_proj', 'occam', 'sp_partition', "graph_weight.py", "merge_proj.py", "occam.py", "sp_partition.py", 'gui'])
    parser.add_argument('--scenes', '-sc', type=str, nargs='+')
    parser.add_argument('--output', '-o', type=str, default='') # not used for now
    parser.add_argument('--config', '-cf', type=str, default='configs/def.yml')
    parser.add_argument('--feature_level', '-l', type=str) # for graph_weight
    parser.add_argument('--graph_cut', '-k', action='store_true') # for sp_partition
    parser.add_argument('--quiet', '-q', action='store_true')
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    dataset = OmegaConf.to_container(cfg.dataset, resolve=True)

    script = args.script.split('.')[0]
    # additional args
    addi_args = ''
    if script == 'graph_weight':
        level = args.feature_level if args.feature_level else cfg.graph_weight.level
        addi_args = '--config ' + args.config + ' --level ' + str(level)
    elif script == 'merge_proj':
        addi_args = f'--thres_connect {cfg.merge_proj.thres_connect} --thres_merge {cfg.merge_proj.thres_merge} --feat_assign {cfg.merge_proj.feat_assign}'
        if hasattr(cfg.merge_proj, 'seg_enhance') and cfg.merge_proj.seg_enhance:
            addi_args += ' --seg_enhance'
    elif script == 'sp_partition':
        if args.graph_cut:
            addi_args += f' -k neighbor_new.pt --pcp_regularization {cfg.spt.pcp_regularization} --pcp_spatial_weight {cfg.spt.pcp_spatial_weight}'
        if hasattr(cfg.spt, 'aligned_normal') and cfg.spt.aligned_normal:
            addi_args += ' -a'

    if args.quiet:
        addi_args += '  > /dev/null'

    used_scenes = args.scenes
    if not used_scenes:
        used_scenes = dataset['scenes']
    idx = 0
    for scene_name in dataset['scenes']:
        if scene_name not in used_scenes:
            continue
        idx += 1
        source_path = f'{dataset["data_path"]}/{scene_name}/'
        model_path = f'output/{dataset["save_folder"]}/{scene_name}/'
        if script == 'gui':
            cmd = f"python gui/main.py 'source_path={source_path}' 'load={model_path}/point_cloud/iteration_30000'"
        else:
            cmd = f'python {script}.py -s {source_path} --model_path {model_path} {addi_args}'
        print('>>', cmd)
        code = os.system(cmd)
        if os.WIFSIGNALED(code):
            sig = os.WTERMSIG(code)
            print(f"[{idx}/{len(used_scenes)}] {scene_name}, Terminated by signal {sig}, exiting")
            break
        elif os.WIFEXITED(code):
            print(f'[{idx}/{len(used_scenes)}] Done with {scene_name}')