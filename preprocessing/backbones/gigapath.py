import timm

def gigapath():
    return timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
