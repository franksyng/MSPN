import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import os
import pandas as pd
from utils.utils import *
from PIL import Image
from math import floor
import matplotlib.pyplot as plt
from datasets.wsi_dataset import Wsi_Region
import h5py
from wsi_core.WholeSlideImage import WholeSlideImage
from scipy.stats import percentileofscore
import math
from utils.file_utils import save_hdf5
from scipy.stats import percentileofscore

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def score2percentile(score, ref):
    percentile = percentileofscore(ref, score)
    return percentile

def drawHeatmap(scores, coords, opt_th, slide_path=None, wsi_object=None, vis_level = -1, **kwargs):
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)
        print(wsi_object.name)
    print('!!!!!', vis_level)
    wsi = wsi_object.getOpenSlide()
    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)
    
    heatmap, boxes_for_qupath_vis = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, **kwargs)
    return heatmap, boxes_for_qupath_vis
    

def initialize_wsi(wsi_path, seg_mask_path=None, seg_params=None, filter_params=None):
    wsi_object = WholeSlideImage(wsi_path)
    if seg_params['seg_level'] < 0:
        best_level = wsi_object.wsi.get_best_level_for_downsample(32)
        seg_params['seg_level'] = best_level

    wsi_object.segmentTissue(**seg_params, filter_params=filter_params)
    wsi_object.saveSegmentation(seg_mask_path)
    return wsi_object

# def compute_from_patches(wsi_object, clam_pred=None, model=None, feature_extractor=None, batch_size=512,
#     attn_save_path=None, ref_scores=None, feat_save_path=None, **wsi_kwargs):    
#     top_left = wsi_kwargs['top_left']
#     bot_right = wsi_kwargs['bot_right']
#     patch_size = wsi_kwargs['patch_size']
    
#     roi_dataset = Wsi_Region(wsi_object, **wsi_kwargs)
#     roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size, num_workers=8)
#     print('total number of patches to process: ', len(roi_dataset))
#     num_batches = len(roi_loader)
#     print('number of batches: ', len(roi_loader))
#     mode = "w"
#     for idx, (roi, coords) in enumerate(roi_loader):
#         roi = roi.to(device)
#         coords = coords.numpy()
#         print(roi.shape)
#         print(coords.shape)
        
#         with torch.no_grad():
#             features = feature_extractor(roi)

#             if attn_save_path is not None:
#                 A, _ = model(features, vis_heatmap=True)
           
#                 if A.size(0) > 1: #CLAM multi-branch attention
#                     A = A[clam_pred]

#                 A = A.view(-1, 1).cpu().numpy()

#                 if ref_scores is not None:
#                     for score_idx in range(len(A)):
#                         A[score_idx] = score2percentile(A[score_idx], ref_scores)

#                 asset_dict = {'attention_scores': A, 'coords': coords}
#                 save_path = save_hdf5(attn_save_path, asset_dict, mode=mode)
    
#         if idx % math.ceil(num_batches * 0.05) == 0:
#             print('processed {} / {}'.format(idx, num_batches))

#         if feat_save_path is not None:
#             asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
#             save_hdf5(feat_save_path, asset_dict, mode=mode)

#         mode = "a"
#     return attn_save_path, feat_save_path, wsi_object

def compute_from_patches(wsi_object, clam_pred=None, model=None, feature_extractor=None, batch_size=512,
    attn_save_path=None, ref_scores=None, feat_save_path=None, **wsi_kwargs):    
    top_left = wsi_kwargs['top_left']
    bot_right = wsi_kwargs['bot_right']
    patch_size = wsi_kwargs['patch_size']
    feature_extractor, backbone_name = feature_extractor
    
    roi_dataset = Wsi_Region(wsi_object, **wsi_kwargs)
    roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size, num_workers=8)
    print('total number of patches to process: ', len(roi_dataset))
    num_batches = len(roi_loader)
    print('number of batches: ', len(roi_loader))
    mode = "w"
    for idx, (roi, coords) in enumerate(roi_loader):
        roi = roi.to(device)
        coords = coords.numpy()
        print(roi.shape)
        print(coords.shape)
        
        with torch.no_grad():
            if backbone_name == 'conch':
                features = feature_extractor.encode_image(roi, proj_contrast=False, normalize=False)
            else:
                features = feature_extractor(roi)

            if attn_save_path is not None:
                coords = torch.from_numpy(coords)
                A, _ = model(features, vis_heatmap=True)
           
                if A.size(0) > 1 and A.size(0) != 8: #CLAM multi-branch attention
                    A = A[clam_pred]

                A = A.view(-1, 1).cpu().numpy()

                if ref_scores is not None:
                    for score_idx in range(len(A)):
                        A[score_idx] = score2percentile(A[score_idx], ref_scores)

                asset_dict = {'attention_scores': A, 'coords': coords.numpy()}
                save_path = save_hdf5(attn_save_path, asset_dict, mode=mode)
    
        if idx % math.ceil(num_batches * 0.05) == 0:
            print('processed {} / {}'.format(idx, num_batches))

        if feat_save_path is not None:
            asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
            save_hdf5(feat_save_path, asset_dict, mode=mode)

        mode = "a"
    return attn_save_path, feat_save_path, wsi_object

def compute_from_patches_trans(wsi_object, clam_pred=None, model=None, feature_extractor=None, batch_size=512,
    attn_save_path=None, ref_scores=None, feat_save_path=None, **wsi_kwargs):    
    top_left = wsi_kwargs['top_left']
    bot_right = wsi_kwargs['bot_right']
    patch_size = wsi_kwargs['patch_size']

    feature_extractor, backbone_name = feature_extractor
    
    roi_dataset = Wsi_Region(wsi_object, **wsi_kwargs)
    roi_loader = get_simple_loader(roi_dataset, batch_size=batch_size, num_workers=8)
    print('total number of patches to process: ', len(roi_dataset))
    num_batches = len(roi_loader)
    print('number of batches: ', len(roi_loader))
    mode = "w"
    for idx, (roi, coords) in enumerate(roi_loader):
        roi = roi.to(device)
        coords = coords.numpy()
        print(roi.shape)
        print(coords.shape)
        
        with torch.no_grad():
            if backbone_name == 'conch':
                features = feature_extractor.encode_image(roi, proj_contrast=False, normalize=False)
            else:
                features = feature_extractor(roi)

            if attn_save_path is not None:
                coords = torch.from_numpy(coords)
                # A, _ = model(features, coords, vis_heatmap=True)
                A, _ = model(features, coords, vis_coarse_map=True)
                A = A[3072]

                if A.size(0) > 1 and A.size(0) != 8: #CLAM multi-branch attention
                    A = A[clam_pred]
                elif A.size(0) == 8:
                    print('getting heatmap from multi-head attention')
                    A = torch.mean(A, dim=0)

                A = A.view(-1, 1).cpu().numpy()

                if ref_scores is not None:
                    for score_idx in range(len(A)):
                        A[score_idx] = score2percentile(A[score_idx], ref_scores)

                asset_dict = {'attention_scores': A, 'coords': coords.numpy()}
                save_path = save_hdf5(attn_save_path, asset_dict, mode=mode)
    
        if idx % math.ceil(num_batches * 0.05) == 0:
            print('processed {} / {}'.format(idx, num_batches))

        if feat_save_path is not None:
            asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
            save_hdf5(feat_save_path, asset_dict, mode=mode)

        mode = "a"
    return attn_save_path, feat_save_path, wsi_object

def compute_from_patches2(wsi_object, model, features, coords, clam_pred, attn_save_path=None, ref_scores=None, feat_save_path=None, **wsi_kwargs):    
    # top_left = wsi_kwargs['top_left']
    # bot_right = wsi_kwargs['bot_right']
    # patch_size = wsi_kwargs['patch_size']
    mode = "w"
    with torch.no_grad():
        if attn_save_path is not None:
            features = features.to(device)
            coords = torch.from_numpy(coords)
            A, _ = model(features, coords, vis_heatmap=True)
        
            if A.size(0) > 1 and A.size(0) != 8: #CLAM multi-branch attention
                A = A[clam_pred]
            elif A.size(0) == 8:
                print('getting heatmap from multi-head attention')
                A = torch.mean(A, dim=0)

            A = A.view(-1, 1).cpu().numpy()

            if ref_scores is not None:
                for score_idx in range(len(A)):
                    A[score_idx] = score2percentile(A[score_idx], ref_scores)

            asset_dict = {'attention_scores': A, 'coords': coords.numpy()}
            save_path = save_hdf5(attn_save_path, asset_dict, mode=mode)

    # if idx % math.ceil(num_batches * 0.05) == 0:
    #     print('processed {} / {}'.format(idx, num_batches))

    if feat_save_path is not None:
        asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
        save_hdf5(feat_save_path, asset_dict, mode=mode)

    mode = "a"
    return attn_save_path, feat_save_path, wsi_object

def compute_from_patches_coarse(wsi_object, model, features, coords, clam_pred, attn_save_path=None, ref_scores=None, feat_save_path=None, **wsi_kwargs):    
    # top_left = wsi_kwargs['top_left']
    # bot_right = wsi_kwargs['bot_right']
    # patch_size = wsi_kwargs['patch_size']
    mode = "w"
    with torch.no_grad():
        if attn_save_path is not None:
            features = features.to(device)
            coords = torch.from_numpy(coords)
            A, _ = model(features, coords, vis_coarse_map=True)
            # A, _ = model(features, coords, vis_heatmap=True)
            A = A[3072]
        
            if A.size(0) > 1 and A.size(0) != 8: #CLAM multi-branch attention
                A = A[clam_pred]
            elif A.size(0) == 8:
                print('getting heatmap from multi-head attention')
                A = torch.mean(A, dim=0)

            A = A.view(-1, 1).cpu().numpy()

            if ref_scores is not None:
                for score_idx in range(len(A)):
                    A[score_idx] = score2percentile(A[score_idx], ref_scores)

            asset_dict = {'attention_scores': A, 'coords': coords.numpy()}
            save_path = save_hdf5(attn_save_path, asset_dict, mode=mode)

    # if idx % math.ceil(num_batches * 0.05) == 0:
    #     print('processed {} / {}'.format(idx, num_batches))

    if feat_save_path is not None:
        asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
        save_hdf5(feat_save_path, asset_dict, mode=mode)

    mode = "a"
    return attn_save_path, feat_save_path, wsi_object

def compute_from_patches_fast(wsi_object, model, features, coords, clam_pred, attn_save_path=None, ref_scores=None, feat_save_path=None, **wsi_kwargs):    
    # top_left = wsi_kwargs['top_left']
    # bot_right = wsi_kwargs['bot_right']
    # patch_size = wsi_kwargs['patch_size']
    mode = "w"
    with torch.no_grad():
        if attn_save_path is not None:
            features = features.to(device)
            coords = torch.from_numpy(coords)
            A, _ = model(features, vis_heatmap=True)
            # A, _ = model(features, coords, vis_heatmap=True)
            # A = A[1536]
        
            if A.size(0) > 1 and A.size(0) != 8: #CLAM multi-branch attention
                A = A[clam_pred]
            elif A.size(0) == 8:
                print('getting heatmap from multi-head attention')
                A = torch.mean(A, dim=0)

            A = A.view(-1, 1).cpu().numpy()

            if ref_scores is not None:
                for score_idx in range(len(A)):
                    A[score_idx] = score2percentile(A[score_idx], ref_scores)

            asset_dict = {'attention_scores': A, 'coords': coords.numpy()}
            save_path = save_hdf5(attn_save_path, asset_dict, mode=mode)

    # if idx % math.ceil(num_batches * 0.05) == 0:
    #     print('processed {} / {}'.format(idx, num_batches))

    if feat_save_path is not None:
        asset_dict = {'features': features.cpu().numpy(), 'coords': coords}
        save_hdf5(feat_save_path, asset_dict, mode=mode)

    mode = "a"
    return attn_save_path, feat_save_path, wsi_object