from __future__ import print_function
import sys
sys.path.append('/home/frank/PycharmProjects/mrs_bc')

import numpy as np
import json

import argparse
import torch, torchvision
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
from utils.eval_utils import initiate_model as initiate_model
# from models.model_clam import CLAM_MB, CLAM_SB
# from models.resnet_custom import resnet50_baseline
# from preprocessing.ctrans_ckpt.ctran import ctranspath
from preprocessing.backbones.gigapath import gigapath
from preprocessing.backbones.resnet import resnet50
from preprocessing.backbones.conch import conch
# from receptor_pred.models.baseline import CLAM_SB
# from new_amil.models.baseline import CLAM_SB
from types import SimpleNamespace
from collections import namedtuple
import h5py
import yaml
from wsi_core.batch_process_utils import initialize_df
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap, compute_from_patches_trans, compute_from_patches2, compute_from_patches_coarse
from wsi_core.wsi_utils import sample_rois, to_percentiles
from utils.file_utils import save_hdf5
from new_amil.utils.universal_utils import find_coords_scale
import pandas as pd
from copy import deepcopy

# fix random seed
from receptor_pred.utils.universal_utils import setup_seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
setup_seed(2024, device)

parser = argparse.ArgumentParser(description='Heatmap inference script')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code')
parser.add_argument('--overlap', type=float, default=None)
parser.add_argument('--config_file', type=str, default="heatmap_config_template.yaml")
args = parser.parse_args()

def norm_heatmap(A):
    A = A / (A.sum(axis=-1, keepdims=True))
    print('a sum',A.sum(axis=-1, keepdims=True))
    return A

def infer_single_slide(model, features, coords, label, reverse_label_dict, opt_th, k=1):
    features = features.to(device)
    print(model)
    print(model.__class__)
    with torch.no_grad():
        A, logits = model(features, coords, vis_coarse_map=True)
        # A, logits = model(features, coords, vis_heatmap=True)
        A = A[3072]
        print('!!!!A shape', A.shape)
        # print('logits', logits)
        # print('logits topk', torch.topk(logits, 1, dim=1))
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)  # probability
        Y_hat = Y_hat.item()
        if opt_th != 'None':
            if Y_prob.cpu().tolist()[0][1] > opt_th:
                Y_hat_calibrated = 1
            else:
                Y_hat_calibrated = 0
        else:
            Y_hat_calibrated = Y_hat
        print('Y_hat non calibrated:', Y_hat)
        print('Y_hat calibrated:', Y_hat_calibrated)
        print('A shape', A.shape)
        A = torch.mean(A, dim=0).view(1,-1)
        # A_list = []
        # for i in range(A.shape[0]):
            # A_list.append(A[i,:,:].view(1,-1).cpu().numpy())
            # A_list.append(A[i,:,:].reshape(-1, 1))
        # A_list.append(A_mean.cpu().numpy())
        A = A.cpu().numpy()
        # if isinstance(model, (CLAM_MB,)):
        #     A = A[Y_hat]
        # A_list = []

        # A.append(np.mean(A, axis=0).reshape(-1, 1))
        # print('len alist', len(A_list))

        print('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat_calibrated], label,
                                                    ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))

        probs, ids = torch.topk(Y_prob, k)
        probs = probs[-1].cpu().numpy()
        ids = ids[-1].cpu().numpy()
        preds_str = np.array([reverse_label_dict[idx] for idx in ids])
    print('Original attn map size:', A.shape)

    return ids, preds_str, probs, A, Y_hat_calibrated


def load_params(df_entry, params):
    for key in params.keys():
        if key in df_entry.index:
            dtype = type(params[key])
            val = df_entry[key]
            val = dtype(val)
            if isinstance(val, str):
                if len(val) > 0:
                    params[key] = val
            elif not np.isnan(val):
                params[key] = val
            else:
                pdb.set_trace()

    return params


def parse_config_dict(args, config_dict):
    if args.save_exp_code is not None:
        config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
    if args.overlap is not None:
        config_dict['patching_arguments']['overlap'] = args.overlap
    return config_dict


def generate_geojson_from_boxes(boxes, path, label):
    features = []
    for box in boxes:
        x, y, width, height, box_id = box
        coords = [
            [str(x), str(y)],
            [str(x + width), str(y)],
            [str(x + width), str(y + height)],
            [str(x), str(y + height)],
            [str(x), str(y)]  # Close the polygon
        ]
        feature = {
            "type": "Feature",
            "id": box_id,
            "geometry": {
            "type": "Polygon",
            "coordinates": [coords]},
            "properties":{"objectType":"annotation", "classification":{"name":label,"color":[153, 0, 0]}}}

        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    out_file = open(path, "w")

    json.dump(geojson, out_file, indent=2)
    out_file.close()


if __name__ == '__main__':
    print('Curr working dir:', os.getcwd())
    config_path = os.path.join('configs', args.config_file)
    config_dict = yaml.safe_load(open(config_path, 'r'))
    config_dict = parse_config_dict(args, config_dict)

    for key, value in config_dict.items():
        if isinstance(value, dict):
            print('\n' + key)
            for value_key, value_value in value.items():
                print(value_key + " : " + str(value_value))
        else:
            print('\n' + key + " : " + str(value))

    # decision = input('Continue? Y/N ')
    # if decision in ['Y', 'y', 'Yes', 'yes']:
    #     pass
    # elif decision in ['N', 'n', 'No', 'NO']:
    #     exit()
    # else:
    #     raise NotImplementedError

    args = config_dict
    patch_args = argparse.Namespace(**args['patching_arguments'])
    data_args = argparse.Namespace(**args['data_arguments'])
    model_args = args['model_arguments']
    model_args.update({'n_classes': args['exp_arguments']['n_classes']})
    model_args = argparse.Namespace(**model_args)
    exp_args = argparse.Namespace(**args['exp_arguments'])
    heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
    sample_args = argparse.Namespace(**args['sample_arguments'])
    extractor_args = argparse.Namespace(**args['extractor_arguments'])

    patch_size = tuple([patch_args.patch_size for i in range(2)])
    step_size = tuple((np.array(patch_size) * (1 - patch_args.overlap)).astype(int))
    print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1],
                                                                                  patch_args.overlap, step_size[0], step_size[1]))

    preset = data_args.preset
    # def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False,
    #                   'keep_ids': 'none', 'exclude_ids': 'none'}
    def_seg_params = {'seg_level': 4, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                      'keep_ids': 'none', 'exclude_ids': 'none'}
    def_filter_params = {'a_t': 8, 'a_h': 4, 'max_n_holes': 8}
    def_vis_params = {'vis_level': 4, 'line_thickness': 100}
    def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    if preset is not None:
        preset_df = pd.read_csv(preset)
        for key in def_seg_params.keys():
            def_seg_params[key] = preset_df.loc[0, key]

        for key in def_filter_params.keys():
            def_filter_params[key] = preset_df.loc[0, key]

        for key in def_vis_params.keys():
            def_vis_params[key] = preset_df.loc[0, key]

        for key in def_patch_params.keys():
            def_patch_params[key] = preset_df.loc[0, key]

    if data_args.process_list is None:
        if isinstance(data_args.data_dir, list):
            slides = []
            for data_dir in data_args.data_dir:
                slides.extend(os.listdir(data_dir))
        else:
            slides = sorted(os.listdir(data_args.data_dir))
        slides = [slide for slide in slides if data_args.slide_ext in slide]
        df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params,
                           use_heatmap_args=False)
    else:
        df = pd.read_csv(os.path.join('process_lists', data_args.process_list))
        split_df = pd.read_csv(os.path.join(data_args.split_info))
        # curr_cases = split_df.loc[split_df['set_type'] == data_args.split_key, ['case_id']]
        curr_cases = split_df.loc[split_df['set_type'] == 'test', ['case_id']]
        curr_df = df.merge(curr_cases, how='inner', on='case_id')
        # print(os.getcwd())
        # df = pd.read_csv(data_args.process_list)
        # df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params,
        #                    use_heatmap_args=False)
        df = initialize_df(curr_df, def_seg_params, def_filter_params, def_vis_params, def_patch_params,
                           use_heatmap_args=False)

    mask = df['process'] == 1
    process_stack = df[mask].reset_index(drop=True)
    total = len(process_stack)
    print('\nlist of slides to process: ')
    print(process_stack.head(len(process_stack)))

    print('\ninitializing model from checkpoint')
    ckpt_path = model_args.ckpt_path
    print('\nckpt path: {}'.format(ckpt_path))

    if model_args.initiate_fn == 'initiate_model':
        model = initiate_model(model_args, ckpt_path, device)
    else:
        raise NotImplementedError

    # feature_extractor = resnet50_baseline(pretrained=True)
    # feature_extractor = ctranspath()
    # feature_extractor.head = nn.Identity()
    # td = torch.load(extractor_args.ctrans_ckpt)
    # feature_extractor.load_state_dict(td['model'], strict=True)
    # feature_extractor.eval()
    # --
    # feature_extractor = gigapath_backbone()
    # feature_extractor = resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT, use_fc=False, use_layer4=False)
    feature_extractors = (conch(), 'conch')
    print('Done!')

    label_dict = data_args.label_dict
    class_labels = list(label_dict.keys())
    class_encodings = list(label_dict.values())
    reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))}
    print('!!!!!! REV label dict', reverse_label_dict)
    if torch.cuda.device_count() > 1:
        device_ids = list(range(torch.cuda.device_count()))
        feature_extractor = nn.DataParallel(feature_extractors[0], device_ids=device_ids).to('cuda:0')
    else:
        feature_extractor = feature_extractors[0].to(device)

    os.makedirs(exp_args.production_save_dir, exist_ok=True)
    os.makedirs(exp_args.raw_save_dir, exist_ok=True)
    blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size,
                         'custom_downsample': patch_args.custom_downsample, 'level': patch_args.patch_level,
                         'use_center_shift': heatmap_args.use_center_shift}

    # slides = os.listdir('/media/frank/Elements/Leica_scanner/Leica_1st_HE_corrected')
    slides = os.listdir(data_args.data_dir)
    # print(slides)
    avg_attn_scores =  {'case_id': [], 'raw_scores': [], 'cal_scores':[], 'y_true': [], 'y_hat_cal': []}
    # print('process stack:', process_stack)
    receptor_type = data_args.receptor_type
    if receptor_type == 'er':
        label_name = 'labels_er_2cls'
    elif receptor_type == 'pr':
        label_name = 'labels_pr_2cls'
    elif receptor_type == 'her2':
        label_name = 'labels_her2_2cls'
    elif receptor_type == 'c16':
        label_name = 'label'
    elif receptor_type == 'thrb':
        label_name = 'label'
    elif receptor_type == 'rcc':
        label_name = 'label'
    elif receptor_type == 'dahep':
        label_name = 'label'
    elif receptor_type == 'surgen_surv':
        label_name = 'label'
    else:
        print('[main] Unknown receptor type.')
        NotImplementedError

    slide_name = None
    print(process_stack)
    for i in range(len(process_stack)):
        case_id = process_stack.loc[i, 'case_id']
        print(case_id)
        print(slides)
        for slide in slides:
            if case_id in slide:
                slide_name = slide
        if slide_name == None:
            continue
        if data_args.slide_ext not in slide_name:
            slide_name += data_args.slide_ext
        print('\nprocessing: ', slide_name)

        try:
            label = process_stack.loc[i, label_name]
        except KeyError:
            label = 'Unspecified'

        slide_id = slide_name.replace(data_args.slide_ext, '')
        print('=====slide id', slide_id)

        if not isinstance(label, str):
            grouping = reverse_label_dict[label]
        else:
            grouping = label

        p_slide_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, str(grouping))
        os.makedirs(p_slide_save_dir, exist_ok=True)

        r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, str(grouping), slide_id)
        os.makedirs(r_slide_save_dir, exist_ok=True)

        r_geojson_save_dir = os.path.join(exp_args.raw_save_dir, 'geojson', str(grouping))
        print('r geojson path', r_geojson_save_dir)
        os.makedirs(r_geojson_save_dir, exist_ok=True)

        if heatmap_args.use_roi:
            x1, x2 = process_stack.loc[i, 'x1'], process_stack.loc[i, 'x2']
            y1, y2 = process_stack.loc[i, 'y1'], process_stack.loc[i, 'y2']
            top_left = (int(x1), int(y1))
            bot_right = (int(x2), int(y2))
        else:
            top_left = None
            bot_right = None

        print('slide id: ', slide_id)
        print('top left: ', top_left, ' bot right: ', bot_right)

        if isinstance(data_args.data_dir, str):
            slide_path = os.path.join(data_args.data_dir, slide_name)
        elif isinstance(data_args.data_dir, dict):
            data_dir_key = process_stack.loc[i, data_args.data_dir_key]
            slide_path = os.path.join(data_args.data_dir[data_dir_key], slide_name)
        else:
            raise NotImplementedError

        mask_file = os.path.join(r_slide_save_dir, slide_id + '_mask.pkl')

        # Load segmentation and filter parameters
        seg_params = def_seg_params.copy()
        filter_params = def_filter_params.copy()
        vis_params = def_vis_params.copy()

        seg_params = load_params(process_stack.loc[i], seg_params)
        filter_params = load_params(process_stack.loc[i], filter_params)
        vis_params = load_params(process_stack.loc[i], vis_params)

        keep_ids = str(seg_params['keep_ids'])
        if len(keep_ids) > 0 and keep_ids != 'none':
            seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
        else:
            seg_params['keep_ids'] = []

        exclude_ids = str(seg_params['exclude_ids'])
        if len(exclude_ids) > 0 and exclude_ids != 'none':
            seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
        else:
            seg_params['exclude_ids'] = []

        for key, val in seg_params.items():
            print('{}: {}'.format(key, val))

        for key, val in filter_params.items():
            print('{}: {}'.format(key, val))

        for key, val in vis_params.items():
            print('{}: {}'.format(key, val))

        print('Initializing WSI object')
        wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params,
                                    filter_params=filter_params)
        print('Done!')

        wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]

        # the actual patch size for heatmap visualization should be the patch size * downsample factor * custom downsample factor
        vis_patch_size = tuple(
            (np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))

        block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
        mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))
        if vis_params['vis_level'] < 0:
            best_level = wsi_object.wsi.get_best_level_for_downsample(32)
            vis_params['vis_level'] = best_level
        mask = wsi_object.visWSI(**vis_params, number_contours=True)
        mask.save(mask_path)

        features_path = os.path.join(r_slide_save_dir, slide_id + '.pt')
        coords_path = os.path.join(r_slide_save_dir, slide_id + '_coords.pt')
        h5_path = os.path.join(r_slide_save_dir, slide_id + '.h5')
        print('====h5 path', h5_path)

        ##### check if h5_features_file exists ######
        if not os.path.isfile(h5_path):
            _, _, wsi_object = compute_from_patches_trans(wsi_object=wsi_object,
                                                    model=model,
                                                    feature_extractor=feature_extractors,
                                                    batch_size=exp_args.batch_size, **blocky_wsi_kwargs,
                                                    attn_save_path=None, feat_save_path=h5_path,
                                                    ref_scores=None)

        ##### check if pt_features_file exists ######
        if not os.path.isfile(features_path):
            file = h5py.File(h5_path, "r")
            features = torch.tensor(file['features'][:])
            coords = torch.tensor(file['coords'][:])
            torch.save(features, features_path)
            torch.save(coords, coords_path)
            file.close()
        # print('coords shape', coords.shape)

        # load features
        features = torch.load(features_path)
        coords = torch.load(coords_path)
        process_stack.loc[i, 'bag_size'] = len(features)

        wsi_object.saveSegmentation(mask_file)
        opt_th = data_args.optimal_threshold
        Y_hats, Y_hats_str, Y_probs, A, Y_hat_calibrated = infer_single_slide(model, features, coords, label, reverse_label_dict, opt_th,
                                                            exp_args.n_classes)

        # del features
        # DO NOT DEL
        if not os.path.isfile(block_map_save_path):
            file = h5py.File(h5_path, "r")
            coords = file['coords'][:]
            file.close()
            # if type(coords) != 'numpy.ndarray':
            #     asset_dict = {'attention_scores': A, 'coords': coords.numpy()}
            # else:
            asset_dict = {'attention_scores': A, 'coords': coords}
            block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')
        print('[main] block map save path:', block_map_save_path)

        # save top 3 predictions
        for c in range(exp_args.n_classes):
            process_stack.loc[i, 'Pred_{}'.format(c)] = Y_hats_str[c]
            process_stack.loc[i, 'p_{}'.format(c)] = Y_probs[c]

        os.makedirs('heatmaps/results/', exist_ok=True)
        if data_args.process_list is not None:
            process_stack.to_csv('heatmaps/results/{}.csv'.format(data_args.process_list.replace('.csv', '')),
                                index=False)
        else:
            process_stack.to_csv('heatmaps/results/{}.csv'.format(exp_args.save_exp_code), index=False)

        file = h5py.File(block_map_save_path, 'r')
        dset = file['attention_scores']
        coord_dset = file['coords']
        scores = dset[:]
        coords = coord_dset[:]
        file.close()

        samples = sample_args.samples
        for sample in samples:
            if sample['sample']:
                tag = "label_{}_pred_{}".format(label, Y_hats[0])
                sample_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, 'sampled_patches',
                                            str(tag), sample['name'])
                os.makedirs(sample_save_dir, exist_ok=True)
                print('sampling {}'.format(sample['name']))
                sample_results = sample_rois(scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'],
                                            score_start=sample.get('score_start', 0),
                                            score_end=sample.get('score_end', 1))
                for idx, (s_coord, s_score) in enumerate(
                        zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
                    print('coord: {} score: {:.3f}'.format(s_coord, s_score))
                    patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level,
                                                    (patch_args.patch_size, patch_args.patch_size)).convert('RGB')
                    patch.save(os.path.join(sample_save_dir,
                                            '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, slide_id, s_coord[0], s_coord[1],
                                                                                s_score)))

        wsi_kwargs = {'top_left': top_left, 'bot_right': bot_right, 'patch_size': patch_size, 'step_size': step_size,
                    'custom_downsample': patch_args.custom_downsample, 'level': patch_args.patch_level,
                    'use_center_shift': heatmap_args.use_center_shift}

        heatmap_save_name = '{}_blockmap.tiff'.format(slide_id)
        if os.path.isfile(os.path.join(r_slide_save_dir, heatmap_save_name)):
            pass
        else:
            print('Calling draw heatmap FIRST TIME.')
            heatmap, _ = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap,
                                alpha=heatmap_args.alpha, use_holes=True, binarize=False, vis_level=-1,
                                blank_canvas=False,
                                thresh=-1, patch_size=vis_patch_size, convert_to_percentiles=True)
            heatmap.save(os.path.join(r_slide_save_dir, '{}_blockmap.png'.format(slide_id)))

            del heatmap
        print('SCORES', scores)
        percentile_scores = to_percentiles(scores)/100
        # get positive patches
        percentile_scores = percentile_scores * (percentile_scores > 0.75)
        print('PERCENTILE SCORES', percentile_scores)
        # pos_raw_scores = np.where(scores > 0, scores, 0)  # score only considers positive area (specifically, tumour area)
        # avg_attn_scores =  {'case_id': [], 'scores': [], 'y_true': [], 'y_hat_cal': []}
        avg_attn_scores['raw_scores'].append(np.mean(scores))
        avg_attn_scores['cal_scores'].append(np.mean(percentile_scores))
        avg_attn_scores['case_id'].append(case_id)
        avg_attn_scores['y_true'].append(label)
        avg_attn_scores['y_hat_cal'].append(Y_hat_calibrated)
        print(avg_attn_scores)

        save_path = os.path.join(r_slide_save_dir,
                                '{}_{}_roi_{}.h5'.format(slide_id, patch_args.overlap, heatmap_args.use_roi))

        if heatmap_args.use_ref_scores:
            ref_scores = scores
        else:
            ref_scores = None

        if heatmap_args.calc_heatmap:
            print('CAL heatmap')
            # compute_from_patches(wsi_object=wsi_object, clam_pred=Y_hats[0], model=model,
            #                     feature_extractor=feature_extractor, batch_size=exp_args.batch_size, **wsi_kwargs,
            #                     attn_save_path=save_path, ref_scores=ref_scores)
            compute_from_patches_coarse(wsi_object=wsi_object, model=model, features=features, coords=coords, clam_pred=None, attn_save_path=save_path)
#  compute_from_patches2(wsi_object, model, features, coords, clam_pred, attn_save_path=None, ref_scores=None, feat_save_path=None, **wsi_kwargs):    
        if not os.path.isfile(save_path):
            print('heatmap {} not found'.format(save_path))
            if heatmap_args.use_roi:
                save_path_full = os.path.join(r_slide_save_dir,
                                            '{}_{}_roi_False.h5'.format(slide_id, patch_args.overlap))
                print('found heatmap for whole slide')
                save_path = save_path_full
            else:
                continue
        
        print('SECOND RELOAD save file', save_path)
        with h5py.File(save_path, 'r') as file:
            file = h5py.File(save_path, 'r')
            dset = file['attention_scores']
            coord_dset = file['coords']
            scores = dset[:]
            coords = coord_dset[:]
            file.close()

        heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': heatmap_args.vis_level,
                            'blur': heatmap_args.blur, 'custom_downsample': heatmap_args.custom_downsample}
        if heatmap_args.use_ref_scores:
            heatmap_vis_args['convert_to_percentiles'] = False

        heatmap_save_name = '{}_{}_roi_{}_blur_{}_rs_{}_bc_{}_a_{}_l_{}_bi_{}_{}.{}'.format(slide_id,
                                                                                            float(patch_args.overlap),
                                                                                            int(heatmap_args.use_roi),
                                                                                            int(heatmap_args.blur),
                                                                                            int(heatmap_args.use_ref_scores),
                                                                                            int(heatmap_args.blank_canvas),
                                                                                            float(heatmap_args.alpha),
                                                                                            int(heatmap_args.vis_level),
                                                                                            int(heatmap_args.binarize),
                                                                                            float(
                                                                                                heatmap_args.binary_thresh),
                                                                                            heatmap_args.save_ext)
        print('test out name: ', os.path.join(p_slide_save_dir, heatmap_save_name))
        if heatmap_args.gen_high_reso_map:
            if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
                pass
            else:
                print('Calling draw heatmap SECOND TIME.')
                heatmap, boxes_for_qupath_vis = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object,
                                    cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, **heatmap_vis_args,
                                    binarize=heatmap_args.binarize,
                                    blank_canvas=heatmap_args.blank_canvas,
                                    thresh=heatmap_args.binary_thresh, patch_size=vis_patch_size,
                                    overlap=patch_args.overlap,
                                    top_left=top_left, bot_right=bot_right)
                if heatmap_args.save_ext == 'jpg':
                    heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
                else:
                    heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))
                json_path = os.path.join(r_geojson_save_dir, '{}.json'.format(slide_id))
                generate_geojson_from_boxes(boxes_for_qupath_vis, json_path, grouping)

        if heatmap_args.save_orig:
            if heatmap_args.vis_level >= 0:
                vis_level = heatmap_args.vis_level
            else:
                vis_level = vis_params['vis_level']
            heatmap_save_name = '{}_orig_{}.{}'.format(slide_id, int(vis_level), heatmap_args.save_ext)
            if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
                pass
            else:
                heatmap = wsi_object.visWSI(vis_level=vis_level, view_slide_only=True,
                                            custom_downsample=heatmap_args.custom_downsample)
                if heatmap_args.save_ext == 'jpg':
                    heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
                else:
                    heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))
    # print('lalalalala:', os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code))
    with open(os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, 'config.yaml'), 'w') as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False)
    
    os.makedirs('summary', exist_ok=True)
    # curr_summary_dir = f'{model_args.model_type}_{data_args.receptor_type}_split_{data_args.split_num}'
    # avg_attn_scores['cal_scores'] = to_percentiles(avg_attn_scores['raw_scores']).tolist()
    attn_scores_df = pd.DataFrame(avg_attn_scores)
    attn_scores_df.to_csv(os.path.join('summary', f'{model_args.model_type}_{data_args.receptor_type}_split_{data_args.split_num}.csv'))


