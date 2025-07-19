import os
import torch
from random import randint
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
import cv2
import json
import numpy as np
from utils.vlm_utils import ClipSimMeasure
from nag_data import SemanticNAG

def polygon_to_mask(img_shape, points_list):
    points = np.asarray(points_list, dtype=np.int32)
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask

@torch.no_grad()
def training(dataset, pipe):
    gaussians = GaussianModel(dataset.sh_degree, 20)
    scene = Scene(dataset, gaussians, 30000, load_sem=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    nag = torch.load(os.path.join(dataset.model_path, f"sai_nag.pt"))

    vlm = ClipSimMeasure()
    vlm.load_model()
    snag = SemanticNAG(nag['nag'], nag['nag_feat'])

    # "[scene]/[prompt]/[colmap_format_dataset]"
    scene_name = dataset.source_path.split('/')[-1]
    data_path = os.path.join(os.path.dirname(dataset.source_path), 'label', scene_name)
    out_path = os.path.join(args.path_pred, scene_name)
    os.makedirs(out_path, exist_ok=True)
    # img_list = os.listdir(data_path) find ends with .jpg
    img_list = [f for f in os.listdir(data_path) if f.endswith('.jpg')]
    for im in img_list:
        image_name = im.split('.')[0]
        js_file = os.path.join(data_path, image_name+'.json')
        anno = json.load(open(js_file))
        for cam in scene.getTrainCameras():
            if cam.image_name == image_name:
                break

        os.makedirs(os.path.join(out_path, cam.image_name), exist_ok=True)
        prompt_list = [obj['category'] for obj in anno['objects']]
        prompt_list = list(set(prompt_list))

        for prompt in prompt_list:
            # segmentation prediction
            vlm.encode_text(prompt)
            point_valid = snag.get_related_gaussian([vlm.compute_similarity(f) for f in snag.feat], topk=3, level=[2,3])
            point_valid = point_valid.expand(-1, 20).cuda()
            gaussians._semantics = point_valid        
            embd_sim = render(cam, gaussians, pipe, background)["semantics"]
            w, h = cam.image_width, cam.image_height
            mask = embd_sim.reshape(20, -1)[0] > 0.5
            binary_mask = mask.reshape(h, w)

            # get ground truth mask
            mask_gt = np.zeros((h, w), dtype=np.uint8)
            for obj in anno['objects']:
                if obj['category'] == prompt:
                    _mask_gt = polygon_to_mask((h, w), obj['segmentation'])
                    mask_gt = np.maximum(mask_gt, _mask_gt)

            cv2.imwrite(os.path.join(out_path, cam.image_name, prompt.replace(' ', '_')+'.png'), binary_mask.cpu().numpy() * 255)
            cv2.imwrite(os.path.join(out_path, cam.image_name, prompt.replace(' ', '_')+'_gt.png'), mask_gt * 255)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--path_pred', type=str, default='output/render/lerf')
    args = parser.parse_args(sys.argv[1:])

    safe_state(True)
    training(lp.extract(args), pp.extract(args))

    # All done
    print("\nPred complete.")