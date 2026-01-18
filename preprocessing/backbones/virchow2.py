import timm
import torch
from timm.layers import SwiGLUPacked

def virchow2():
    return timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)