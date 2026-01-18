# basic imports
import os
import pandas as pd
import h5py
import numpy as np

# torch
import torch
from torch.utils.data import Dataset

def normalize_case_id(series):
    # Convert to pandas StringDtype, trim, and strip trailing ".0"
    s = series.astype('string').str.strip()
    return s.str.replace(r'\.0+$', '', regex=True)

def load_slide_data(slides_root, annotations):
    slides = os.listdir(slides_root)
    split_ann_dict = {'case_id': [], 'img_path': []}
    case_num = len(annotations)
    annotations['case_id'] = normalize_case_id(annotations['case_id'])
    # print('load slide', len(annotations))
    for i in range(case_num):
        print(f'[dataset] Traversing slide [{i+1}/{case_num}]', end='\r')
        case_id = str(annotations['case_id'].iloc[i])
        case_id = annotations['case_id'].iloc[i]
        # print(case_id, slides[0])
        for slide in slides:
            if case_id in slide:
                slide_path = os.path.join(slides_root, slide)
                # camil_info = os.path.join(slides_root.rstrip('/') + '_camil/', slide)
                split_ann_dict['case_id'].append(case_id)
                split_ann_dict['img_path'].append(slide_path)
                # split_ann_dict['camil_info_path'].append(camil_info)
    split_ann = pd.DataFrame(split_ann_dict).sample(frac=1, random_state=358)
    # print('sp ann', len(split_ann))
    # split_ann['case_id'] = split_ann['case_id'].astype('string')
    split_ann['case_id'] = normalize_case_id(split_ann['case_id'])
    print(f'[dataset] A total of {case_num} slides are loaded.')
    return split_ann


class GenericDataset(Dataset):
    def __init__(self, annotations, splits, set_type, eval_mode=False):
        self.annotations = annotations
        if not eval_mode:
            self.splits = splits
            curr_cases = self.splits.loc[self.splits['set_type'] == set_type, ['case_id']]
            self.curr_annotations = self.annotations.merge(curr_cases, how='inner', on='case_id')
        else:
            self.curr_annotations = self.annotations

    def __len__(self):
        return None

    def __getitem__(self, idx):
        return None


class SlideDataset(GenericDataset):
    def __init__(self, annotations, slides_root, splits, label_col, set_type, pos_enc=False, eval_mode=False):
        super(SlideDataset, self).__init__(annotations, splits, set_type, eval_mode)
        self.curr_annotations['case_id'] = normalize_case_id(self.curr_annotations['case_id'])
        split_ann = load_slide_data(slides_root, self.curr_annotations)
        self.split_ann = split_ann.merge(self.curr_annotations, how='inner', on='case_id')
        self.label = label_col
        self.pos_enc = pos_enc

    def __len__(self):
        return len(self.split_ann)

    def __getitem__(self, idx):
        img_path = self.split_ann['img_path'].iloc[idx]
        img_id = self.split_ann['case_id'].iloc[idx]
        img = torch.tensor(np.array(h5py.File(img_path, 'r')['features']), dtype=torch.float32)
        coords = torch.tensor(np.array(h5py.File(img_path, 'r')['coords']), dtype=torch.float32)  # [x, y]
        label = self.split_ann[self.label].iloc[idx]

        if self.pos_enc:
            return (img, coords), label, img_id
        else:
            return img, label, img_id

    def get_len(self):
        return len(self.split_ann)

    def get_label_list(self):
        return self.split_ann[self.label].values.tolist()

class SlideDatasetMS(GenericDataset):
    def __init__(self, annotations, slides_root, splits, label_col, set_type, pos_enc=False, eval_mode=False):
        super(SlideDatasetMS, self).__init__(annotations, splits, set_type, eval_mode)
        # print('ann len in', len(self.curr_annotations))
        self.curr_annotations['case_id'] = normalize_case_id(self.curr_annotations['case_id'])
        slides_5x, slides_10x, slides_20x = slides_root
        split_ann_5x = load_slide_data(slides_5x, self.curr_annotations)
        split_ann_10x = load_slide_data(slides_10x, self.curr_annotations)
        split_ann_20x = load_slide_data(slides_20x, self.curr_annotations)
        self.split_ann_5x = split_ann_5x.merge(self.curr_annotations, how='inner', on='case_id')
        self.split_ann_10x = split_ann_10x.merge(self.curr_annotations, how='inner', on='case_id')
        self.split_ann_20x = split_ann_20x.merge(self.curr_annotations, how='inner', on='case_id')
        self.label = label_col
        self.pos_enc = pos_enc

    def __len__(self):
        return len(self.split_ann_20x)

    def __getitem__(self, idx):
        img_path_5x = self.split_ann_5x['img_path'].iloc[idx]
        img_path_10x = self.split_ann_10x['img_path'].iloc[idx]
        img_path_20x = self.split_ann_20x['img_path'].iloc[idx]
        img_id = self.split_ann_20x['case_id'].iloc[idx]

        img_5x = torch.tensor(np.array(h5py.File(img_path_5x, 'r')['features']), dtype=torch.float32)
        coords_5x = torch.tensor(np.array(h5py.File(img_path_5x, 'r')['coords']), dtype=torch.float32)  # [x, y]

        img_10x = torch.tensor(np.array(h5py.File(img_path_10x, 'r')['features']), dtype=torch.float32)
        coords_10x = torch.tensor(np.array(h5py.File(img_path_10x, 'r')['coords']), dtype=torch.float32)  # [x, y]

        img_20x = torch.tensor(np.array(h5py.File(img_path_20x, 'r')['features']), dtype=torch.float32)
        coords_20x = torch.tensor(np.array(h5py.File(img_path_20x, 'r')['coords']), dtype=torch.float32)  # [x, y]

        label = self.split_ann_20x[self.label].iloc[idx]

        if self.pos_enc:
            return [[img_5x, img_10x, img_20x], [coords_5x, coords_10x, coords_20x]], label, img_id
        else:
            return [img_5x, img_10x, img_20x], label, img_id


    def get_len(self):
        return len(self.split_ann_20x)

    def get_label_list(self):
        return self.split_ann_20x[self.label].values.tolist()

class SlideSurvDataset(GenericDataset):
    def __init__(self, annotations, slides_root, splits, label_col, set_type, pos_enc=False, eval_mode=False):
        super(SlideSurvDataset, self).__init__(annotations, splits, set_type, eval_mode)
        self.curr_annotations['case_id'] = normalize_case_id(self.curr_annotations['case_id'])
        split_ann = load_slide_data(slides_root, self.curr_annotations)
        self.split_ann = split_ann.merge(self.curr_annotations, how='inner', on='case_id')
        self.label = label_col
        self.pos_enc = pos_enc

    def __len__(self):
        return len(self.split_ann)

    def __getitem__(self, idx):
        img_path = self.split_ann['img_path'].iloc[idx]
        img_id = self.split_ann['case_id'].iloc[idx]
        img = torch.tensor(np.array(h5py.File(img_path, 'r')['features']), dtype=torch.float32)
        coords = torch.tensor(np.array(h5py.File(img_path, 'r')['coords']), dtype=torch.float32)  # [x, y]
        label = self.split_ann[self.label].iloc[idx]
        surv_month = self.split_ann['survival_month'].iloc[idx]
        censorship = self.split_ann['censorship'].iloc[idx]

        if self.pos_enc:
            return (img, coords), label, img_id, surv_month, censorship
        else:
            return img, label, img_id, surv_month, censorship


    def get_len(self):
        return len(self.split_ann)

    def get_label_list(self):
        return self.split_ann[self.label].values.tolist()

class SlideSurvDatasetMS(GenericDataset):
    def __init__(self, annotations, slides_root, splits, label_col, set_type, pos_enc=False, eval_mode=False):
        super(SlideSurvDatasetMS, self).__init__(annotations, splits, set_type, eval_mode)
        self.curr_annotations['case_id'] = normalize_case_id(self.curr_annotations['case_id'])
        slides_5x, slides_10x, slides_20x = slides_root
        split_ann_5x = load_slide_data(slides_5x, self.curr_annotations)
        split_ann_10x = load_slide_data(slides_10x, self.curr_annotations)
        split_ann_20x = load_slide_data(slides_20x, self.curr_annotations)
        self.split_ann_5x = split_ann_5x.merge(self.curr_annotations, how='inner', on='case_id')
        self.split_ann_10x = split_ann_10x.merge(self.curr_annotations, how='inner', on='case_id')
        self.split_ann_20x = split_ann_20x.merge(self.curr_annotations, how='inner', on='case_id')
        self.label = label_col
        self.pos_enc = pos_enc

    def __len__(self):
        return len(self.split_ann_20x)

    def __getitem__(self, idx):
        img_path_5x = self.split_ann_5x['img_path'].iloc[idx]
        img_path_10x = self.split_ann_10x['img_path'].iloc[idx]
        img_path_20x = self.split_ann_20x['img_path'].iloc[idx]
        img_id = self.split_ann_20x['case_id'].iloc[idx]

        img_5x = torch.tensor(np.array(h5py.File(img_path_5x, 'r')['features']), dtype=torch.float32)
        coords_5x = torch.tensor(np.array(h5py.File(img_path_5x, 'r')['coords']), dtype=torch.float32)  # [x, y]

        img_10x = torch.tensor(np.array(h5py.File(img_path_10x, 'r')['features']), dtype=torch.float32)
        coords_10x = torch.tensor(np.array(h5py.File(img_path_10x, 'r')['coords']), dtype=torch.float32)  # [x, y]

        img_20x = torch.tensor(np.array(h5py.File(img_path_20x, 'r')['features']), dtype=torch.float32)
        coords_20x = torch.tensor(np.array(h5py.File(img_path_20x, 'r')['coords']), dtype=torch.float32)  # [x, y]

        label = self.split_ann_20x[self.label].iloc[idx]
        surv_month = self.split_ann_20x['survival_month'].iloc[idx]
        censorship = self.split_ann_20x['censorship'].iloc[idx]

        if self.pos_enc:
            return [[img_5x, img_10x, img_20x], [coords_5x, coords_10x, coords_20x]], label, img_id, surv_month, censorship
        else:
            return [img_5x, img_10x, img_20x], label, img_id, surv_month, censorship


    def get_len(self):
        return len(self.split_ann_20x)

    def get_label_list(self):
        return self.split_ann_20x[self.label].values.tolist()
