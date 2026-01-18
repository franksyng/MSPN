# MSPN
Multi-scale Pyramidal Network. A plug-and-play module suitable for attention-based MIL frameworks that introduced progressive multi-scale analysis over WSI.

___
# Environment
We used ```torch 2.2.0``` with ```CUDA 12.3``` on Ubuntu 22.04.3 LTS for implementation. Any environment that able to run an MIL pipeline should be able to run MSPN.

# 1. Preprocessing
Patches are tiled and saved with [CLAM](https://github.com/mahmoodlab/CLAM)

# 2. Feature extraction
HuggingFace token is required for CONCH, UNI2 and GigaPath.
```bash
python generate_features.py --backbone [conch, uni2, gigapath] --src PATCH_DIR --save_dir FEAT_DIR
```

# 3. Training
For single-scale and MSPN please use ```main.py/main_surv.py```

For multi-scale benchmarking please use ```main_ms.py/main_ms_surv.py```

For pre-trained ABMIL please use ```main_pre.py/main_pre_surv.py```
```bash
python [main.py ...] --arch MODEL_NAME --ann ANNOTATION_PATH --split_dir SPLIT_PATH --data_dir FEAT_PATH --res_root RES_PATH  --task TASK  --lr 2e-4 --gc 32 --epochs 150  --in_dim FEAT_DIM --scheduler CALR --early_stopping
```
