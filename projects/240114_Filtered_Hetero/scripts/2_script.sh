#! [240112] 
# CUDA_VISIBLE_DEVICES=2 python yandex/train.py --name GAT_l3 --dataset squirrel-directed --model GAT --num_layers 3 --device cuda:0
# CUDA_VISIBLE_DEVICES=2 python yandex/train.py --name GAT_l3 --dataset squirrel-filtered-directed --model GAT --num_layers 3 --device cuda:0 

# CUDA_VISIBLE_DEVICES=2 python yandex/train.py --name GAT_l3 --dataset chameleon-directed --model GAT --num_layers 3 --device cuda:0  


#! [240113]
# CUDA_VISIBLE_DEVICES=2 python ACM/ACM-Pytorch/train.py --dataset_name squirrel --model acmgcnp --variant 0 --lr 0.05  --structure_info 1 --weight_decay 1e-4 --dropout 0.7 --optimizer Adam --fixed_splits 1
CUDA_VISIBLE_DEVICES=2 python GREET/main.py -dataset squirrel -ntrials 10 -sparse 0 -epochs 1000 -cl_batch_size 0 -nlayers_proj 2 -alpha 0.1 -k 0 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.8 -lr_disc 0.001 -margin_hom 0.1 -margin_het 0.3 -cl_rounds 2 -eval_freq 50