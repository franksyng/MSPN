from conch.open_clip_custom import create_model_from_pretrained


def conch():
    model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch")
    return model
