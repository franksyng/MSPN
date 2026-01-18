import timm
import torch
import os
from huggingface_hub import hf_hub_download

def uni2():
    local_dir = "../assets/ckpts/uni2-h/"
    os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
    hf_hub_download("MahmoodLab/UNI2-h", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
    timm_kwargs = {
    'model_name': 'vit_giant_patch14_224',
    'img_size': 224, 
    'patch_size': 14, 
    'depth': 24,
    'num_heads': 24,
    'init_values': 1e-5, 
    'embed_dim': 1536,
    'mlp_ratio': 2.66667*2,
    'num_classes': 0, 
    'no_embed_class': True,
    'mlp_layer': timm.layers.SwiGLUPacked, 
    'act_layer': torch.nn.SiLU, 
    'reg_tokens': 8, 
    'dynamic_img_size': True
    }
    model = timm.create_model(**timm_kwargs)
    model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
    return model