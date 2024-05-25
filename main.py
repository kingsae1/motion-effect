from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import numpy as np
import argparse
import glob
import os
from functools import partial
import vispy
import scipy.misc as misc
from tqdm import tqdm
import yaml
import time
import sys
from mesh import write_ply, read_ply, output_3d_photo
from utils import get_MiDaS_samples, read_MiDaS_depth
import torch
import cv2
from skimage.transform import resize
import imageio
import copy
from networks import Inpaint_Color_Net, Inpaint_Depth_Net, Inpaint_Edge_Net
from MiDaS.run import run_depth
from boostmonodepth_utils import run_boostmonodepth
from MiDaS.monodepth_net import MonoDepthNet
import MiDaS.MiDaS_utils as MiDaS_utils
from bilateral_filtering import sparse_bilateral_filtering

app = FastAPI()

origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

default_config = {
  "depth_edge_model_ckpt": "checkpoints/edge-model.pth",
  "depth_feat_model_ckpt": "checkpoints/depth-model.pth",
  "rgb_feat_model_ckpt": "checkpoints/color-model.pth",
  "MiDaS_model_ckpt": "MiDaS/model.pt",
  "use_boostmonodepth": True,
  "fps": 40,
  "num_frames": 240,
  "x_shift_range": [
    0,
    0,
    -0.015,
    -0.015
  ],
  "y_shift_range": [
    0,
    0,
    -0.015,
    0
  ],
  "z_shift_range": [
    -0.05,
    -0.05,
    -0.05,
    -0.05
  ],
  "traj_types": [
    "double-straight-line",
    "double-straight-line",
    "circle",
    "circle"
  ],
  "video_postfix": [
    "dolly-zoom-in",
    "zoom-in",
    "circle",
    "swing"
  ],
  "specific": "",
  "longer_side_len": 960,
  "src_folder": "image",
  "depth_folder": "depth",
  "mesh_folder": "mesh",
  "video_folder": "video",
  "load_ply": False,
  "save_ply": True,
  "inference_video": True,
  "gpu_ids": 0,
  "offscreen_rendering": False,
  "img_format": ".jpg",
  "depth_format": ".npy",
  "require_midas": True,
  "depth_threshold": 0.04,
  "ext_edge_threshold": 0.002,
  "sparse_iter": 5,
  "filter_size": [
    7,
    7,
    5,
    5,
    5
  ],
  "sigma_s": 4,
  "sigma_r": 0.5,
  "redundant_number": 12,
  "background_thickness": 70,
  "context_thickness": 140,
  "background_thickness_2": 70,
  "context_thickness_2": 70,
  "discount_factor": 1,
  "log_depth": True,
  "largest_size": 512,
  "depth_edge_dilate": 10,
  "depth_edge_dilate_2": 5,
  "extrapolate_border": True,
  "extrapolation_thickness": 60,
  "repeat_inpaint_edge": True,
  "crop_border": [
    0.03,
    0.03,
    0.05,
    0.03
  ],
  "anti_flickering": True
}

@app.post("/genMotionEffect")
async def gen_motion_effect(config):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, default='argument.yml',help='Configure of post processing')
    # args = parser.parse_args()
    # config = yaml.load(open(args.config, 'r'))
    if config == Null :
        config = default_config
        
    if config['offscreen_rendering'] is True:
        vispy.use(app='egl')
    os.makedirs(config['mesh_folder'], exist_ok=True)
    os.makedirs(config['video_folder'], exist_ok=True)
    os.makedirs(config['depth_folder'], exist_ok=True)
    sample_list = get_MiDaS_samples(config['src_folder'], config['depth_folder'], config, config['specific'])
    normal_canvas, all_canvas = None, None
    
    if isinstance(config["gpu_ids"], int) and (config["gpu_ids"] >= 0):
        device = config["gpu_ids"]
    else:
        device = "cpu"
    
    print(f"running on device {device}")
    
    for idx in tqdm(range(len(sample_list))):
        depth = None
        sample = sample_list[idx]
        print("Current Source ==> ", sample['src_pair_name'])
        mesh_fi = os.path.join(config['mesh_folder'], sample['src_pair_name'] +'.ply')
        image = imageio.imread(sample['ref_img_fi'])
    
        print(f"Running depth extraction at {time.time()}")
        if config['use_boostmonodepth'] is True:
            run_boostmonodepth(sample['ref_img_fi'], config['src_folder'], config['depth_folder'])
        elif config['require_midas'] is True:
            run_depth([sample['ref_img_fi']], config['src_folder'], config['depth_folder'],
                      config['MiDaS_model_ckpt'], MonoDepthNet, MiDaS_utils, target_w=640)
    
        if 'npy' in config['depth_format']:
            config['output_h'], config['output_w'] = np.load(sample['depth_fi']).shape[:2]
        else:
            config['output_h'], config['output_w'] = imageio.imread(sample['depth_fi']).shape[:2]
        frac = config['longer_side_len'] / max(config['output_h'], config['output_w'])
        config['output_h'], config['output_w'] = int(config['output_h'] * frac), int(config['output_w'] * frac)
        config['original_h'], config['original_w'] = config['output_h'], config['output_w']
        if image.ndim == 2:
            image = image[..., None].repeat(3, -1)
        if np.sum(np.abs(image[..., 0] - image[..., 1])) == 0 and np.sum(np.abs(image[..., 1] - image[..., 2])) == 0:
            config['gray_image'] = True
        else:
            config['gray_image'] = False
        image = cv2.resize(image, (config['output_w'], config['output_h']), interpolation=cv2.INTER_AREA)
        depth = read_MiDaS_depth(sample['depth_fi'], 3.0, config['output_h'], config['output_w'])
        mean_loc_depth = depth[depth.shape[0]//2, depth.shape[1]//2]
        if not(config['load_ply'] is True and os.path.exists(mesh_fi)):
            vis_photos, vis_depths = sparse_bilateral_filtering(depth.copy(), image.copy(), config, num_iter=config['sparse_iter'], spdb=False)
            depth = vis_depths[-1]
            model = None
            torch.cuda.empty_cache()
            print("Start Running 3D_Photo ...")
            print(f"Loading edge model at {time.time()}")
            depth_edge_model = Inpaint_Edge_Net(init_weights=True)
            depth_edge_weight = torch.load(config['depth_edge_model_ckpt'],
                                           map_location=torch.device(device))
            depth_edge_model.load_state_dict(depth_edge_weight)
            depth_edge_model = depth_edge_model.to(device)
            depth_edge_model.eval()
    
            print(f"Loading depth model at {time.time()}")
            depth_feat_model = Inpaint_Depth_Net()
            depth_feat_weight = torch.load(config['depth_feat_model_ckpt'],
                                           map_location=torch.device(device))
            depth_feat_model.load_state_dict(depth_feat_weight, strict=True)
            depth_feat_model = depth_feat_model.to(device)
            depth_feat_model.eval()
            depth_feat_model = depth_feat_model.to(device)
            print(f"Loading rgb model at {time.time()}")
            rgb_model = Inpaint_Color_Net()
            rgb_feat_weight = torch.load(config['rgb_feat_model_ckpt'],
                                         map_location=torch.device(device))
            rgb_model.load_state_dict(rgb_feat_weight)
            rgb_model.eval()
            rgb_model = rgb_model.to(device)
            graph = None
    
    
            print(f"Writing depth ply (and basically doing everything) at {time.time()}")
            rt_info = write_ply(image,
                                  depth,
                                  sample['int_mtx'],
                                  mesh_fi,
                                  config,
                                  rgb_model,
                                  depth_edge_model,
                                  depth_edge_model,
                                  depth_feat_model)
    
            if rt_info is False:
                continue
            rgb_model = None
            color_feat_model = None
            depth_edge_model = None
            depth_feat_model = None
            torch.cuda.empty_cache()
        if config['save_ply'] is True or config['load_ply'] is True:
            verts, colors, faces, Height, Width, hFov, vFov = read_ply(mesh_fi)
        else:
            verts, colors, faces, Height, Width, hFov, vFov = rt_info
    
    
        print(f"Making video at {time.time()}")
        videos_poses, video_basename = copy.deepcopy(sample['tgts_poses']), sample['tgt_name']
        top = (config.get('original_h') // 2 - sample['int_mtx'][1, 2] * config['output_h'])
        left = (config.get('original_w') // 2 - sample['int_mtx'][0, 2] * config['output_w'])
        down, right = top + config['output_h'], left + config['output_w']
        border = [int(xx) for xx in [top, down, left, right]]
        normal_canvas, all_canvas = output_3d_photo(verts.copy(), colors.copy(), faces.copy(), copy.deepcopy(Height), copy.deepcopy(Width), copy.deepcopy(hFov), copy.deepcopy(vFov),
                            copy.deepcopy(sample['tgt_pose']), sample['video_postfix'], copy.deepcopy(sample['ref_pose']), copy.deepcopy(config['video_folder']),
                            image.copy(), copy.deepcopy(sample['int_mtx']), config, image,
                            videos_poses, video_basename, config.get('original_h'), config.get('original_w'), border=border, depth=depth, normal_canvas=normal_canvas, all_canvas=all_canvas,
                        mean_loc_depth=mean_loc_depth)

@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)},
    )
