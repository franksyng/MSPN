mple job script for anaconda
#$ -N vis_heatmap_script -cwd
#$ -l h_rt=12:00:00 # only runs for 30  mins!
#$ -l h_vmem=16G
#$ -m bea -M frank.wu@ed.ac.uk # CHANGE THIS TO YOUR EMAIL ADDRESS

# Request one GPU in the gpu queue:
#$ -q gpu 
#$ -pe gpu-a100 1

. /etc/profile.d/modules.sh
module load anaconda # this loads a specific version of anaconda
conda activate patho # this starts the 'mypython' environment
module load cuda
#nvidia-smi

# python -u main.py --arch afamildev --ann ../annotations/annotations_cohort_1_partial.csv --split_dir ../annotations/5fold_splits_er/ --data_dir /exports/eddie/scratch/s2170508/gigapath_feats/ --res_root results/gigapath/leica/small --receptor_name er --binary  --lr 2e-5 --gc 32 --epochs 150 --in_dim 1536 --scheduler CAWR --block_type attn --block_num 5

# python -u main.py --arch amil --ann ../annotations/annotations_cohort_1_partial.csv --split_dir ../annotations/5fold_splits_er/ --data_dir /exports/eddie/scratch/s2170508/gigapath_feats/ --res_root results/gigapath/leica/small --receptor_name er --binary  --lr 2e-5 --gc 32 --epochs 1 --in_dim 1536 --scheduler CAWR

# conda install -c conda-forge openslide
