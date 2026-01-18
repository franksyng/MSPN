import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
import argparse
import glob
import h5py

from backbones.resnet import resnet18, resnet50
from backbones.conch import conch
from backbones.gigapath import gigapath
from backbones.virchow2 import virchow2
from backbones.optimus import h_optimus_1
from backbones.uni import uni2
from huggingface_hub import login
login(token='your token')


def setup_seed(seed, device):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True


def check_folder_existence(dir_path):
    """
    Check folder exist or not. If not, create one.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


class RoIDataset(Dataset):
    def __init__(self, img_dir, trnsfrms, grayscale, img_format):
        super().__init__()
        self.transforms = trnsfrms
        # self.images_lst = img_csv
        # self.img_dir = img_dir
        # self.img_list = os.listdir(img_dir)
        self.img_list = sorted(glob.glob(os.path.join(img_dir, '*.'+img_format)))
        self.grayscale = grayscale
        self.img_format = img_format

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # img_name = self.img_list[idx]
        # path = os.path.join(self.img_dir, img_name)
        img_path = self.img_list[idx]
        img_info = img_path.rstrip('.' + self.img_format).split('/')[-1].split('_')  # id_x_y
        coord_x, coord_y = int(img_info[1]), int(img_info[2])
        if self.grayscale:
            image = Image.open(img_path).convert('L')
        else:
            image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)
        return image, np.array([coord_x, coord_y])
    

def get_features(model, ann_csv, transform, slide_root, save_root, grayscale, img_format):
    model.eval()
    slides = os.listdir(slide_root)
    # if specified annotations, process cases according to annotation
    # otherwise, process all cases in the root folder
    # if ann_csv is not None:
        # ann_slides = ann_csv['case_id'].values.tolist()
        # slides = [slide for slide in slides if slide[:10] in ann_slides]
    slide_num = len(slides)
    counter = 1

    # prepare transforms
    if transform is None:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        if grayscale:
            transform_list = [transforms.Grayscale(num_output_channels=3)]
        else:
            transform_list = []
        transform_list.append(transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=mean, std=std))
        transform = transforms.Compose(transform_list)

    # Start processing
    for slide in slides:
        print('[%d/%d] Extracting features from slide %s' % (counter, slide_num, slide))
        slide_path = os.path.join(slide_root, slide)
        counter += 1

        # Check .pt existence
        if os.path.exists(os.path.join(save_root, slide + '.h5')):
            continue

        # Prepare dataset
        dataset = RoIDataset(slide_path, transform, grayscale, img_format)
        dataloader = DataLoader(dataset, num_workers=8, batch_size=256, shuffle=False)
        total_patches = len(dataloader)
        if total_patches == 0:
            continue
        saved_features = None  # Initialise feature tensor
        saved_coords = None
        # print('!!! len', len(saved_features))
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            for idx, (data, coords) in enumerate(dataloader):
                print("Processed batch %d/%d" % (idx + 1, total_patches), end='\r')
                data = data.to(device, dtype=torch.float32)
                if idx == 0:  # assign the first coord and patch feature
                    # saved_features = model(data).detach().cpu()
                    saved_features = inference_data(mdl_name, model, data).detach().cpu()
                    saved_coords = coords
                else:
                    # curr_features = model(data).detach().cpu()
                    curr_features = inference_data(mdl_name, model, data).detach().cpu()
                    saved_features = torch.cat((saved_features, curr_features), dim=0)
                    saved_coords = np.concatenate((saved_coords, coords), axis=0)

            # torch.save(saved_features, os.path.join(save_root, slide + '.pt'))
            h5_f = h5py.File(os.path.join(save_root, slide + '.h5'), 'w')
            h5_f.create_dataset('features', data=saved_features)
            h5_f.create_dataset('coords', data=saved_coords)
            print('Saved tensor size:', h5_f['features'].shape)
            print('Saved coord size:', h5_f['coords'].shape)
            h5_f.close()

def inference_data(mdl_name, model, image):
    if mdl_name == 'conch':
        emb = model.encode_image(image, proj_contrast=False, normalize=False)
    elif mdl_name == 'virchow2':
        output = model(image)
        class_token = output[:, 0]    # size: 1 x 1280
        patch_tokens = output[:, 5:]  # size: 1 x 256 x 1280, tokens 1-4 are register tokens so we ignore those
        # concatenate class token and average pool of patch tokens
        emb = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
    else:
        emb = model(image)
    return emb


parser = argparse.ArgumentParser(description='slide-lvl feature extraction')
parser.add_argument('--backbone', type=str, default='ctranspath')
parser.add_argument('--ann', type=str, default=None, help='path to annotations.')
parser.add_argument('--src', type=str, help='path to slide folders containing patches.')
parser.add_argument('--save_dir', type=str, help='path to save generated slide-lvl features.')
parser.add_argument('--ckpt', type=str, default='./ctrans_ckpt/ctranspath.pth', help='path to checkpoint.')
parser.add_argument('--gray', default=False, action='store_true', help='in grayscale')
parser.add_argument('--use_layer4', action='store_true', default=False)
parser.add_argument('--img_format', type=str, default='jpg',)

if __name__ == '__main__':
    # read args
    args = parser.parse_args()

    # Setup device and seed
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    setup_seed(2024, device)
    transform = None

    # Prepare model
    mdl_name = args.backbone
    if mdl_name == 'ctranspath':
        print('CTrans Prohibited')
        # model = ctranspath()
        # model.head = nn.Identity()
        # td = torch.load(args.ckpt)
        # model.load_state_dict(td['model'], strict=True)
    elif mdl_name == 'resnet18':
        model = resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT, use_fc=False)  # use ImageNet v2 params
    elif mdl_name == 'resnet50':
        model = resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT, use_fc=False, use_layer4=args.use_layer4)  # use ImageNet v2 params
    elif mdl_name == 'gigapath':
        model = gigapath()
    elif mdl_name == 'conch':
        model = conch()
    elif mdl_name == 'virchow2':
        model = virchow2()
    elif mdl_name == 'h-optimus-1':
        model = h_optimus_1()
        transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), 
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])
    elif mdl_name == 'uni2':
        model = uni2()
        transform = transforms.Compose(
        [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
        )
    else:
        print('Unknown backbone:', args.backbone)
        raise NotImplementedError

    # Multi GPUs
    if torch.cuda.device_count() > 1:
        print('Running with {} GPUs'.format(torch.cuda.device_count()))
        device = torch.device("cuda")
        model = nn.DataParallel(model.to(device))
    else:
        print('Running with single GPU')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

    if args.ann is not None:
        ann_csv = pd.read_csv(args.ann)
    else:
        ann_csv = None
    slide_root = args.src
    save_root = args.save_dir
    grayscale = args.gray
    check_folder_existence(save_root)
    get_features(model, ann_csv, transform, slide_root, save_root, grayscale, args.img_format)

