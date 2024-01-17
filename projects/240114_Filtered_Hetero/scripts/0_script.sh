#! [231130]
# hm_margin="0.1"
# ht_margins="0.8 0.3"
# command="python main.py -forward_mode 0 -ntrials 10 -sparse 0 -epochs 400 -cl_batch_size 0 -nlayers_proj 2 -alpha 0.3 -k 0 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.5 -dropedge_rate_1 0.5 -dropedge_rate_2 0.1 -lr_disc 0.001 -cl_rounds 2 -eval_freq 10 -dataset squirrel -margin_hom"
# command="$command $hm_margin"\ "-margin_het"
# for ht_margin in $ht_margins
# do
#     current_date_time="`date +%Y%m%d_%H:%M:%S`"
#     echo "============================================================="
#     echo $current_date_time
#     echo $command $ht_margin

#     CUDA_VISIBLE_DEVICES=0 $command $ht_margin
# done


#! [240112] 
# CUDA_VISIBLE_DEVICES=0 python yandex/train.py --name GAT_l3 --dataset chameleon-directed --model GAT --num_layers 3 --device cuda:0
# CUDA_VISIBLE_DEVICES=0 python yandex/train.py --name GAT_l3 --dataset chameleon-filtered-directed --model GAT --num_layers 3 --device cuda:0 


#! [240113] 
# CUDA_VISIBLE_DEVICES=0 python ACM/ACM-Pytorch/train.py --dataset_name chameleon --model acmgcnp --variant 0 --lr 0.05  --structure_info 1 --weight_decay 1e-4 --dropout 0.7 --optimizer Adam --fixed_splits 1
# CUDA_VISIBLE_DEVICES=0 python ACM/ACM-Pytorch/train.pypy --dataset_name chameleon --model acmgcnp --variant 1 --lr 0.05  --structure_info 1 --weight_decay 1e-4 --dropout 0.7 --optimizer Adam --fixed_splits 1

CUDA_VISIBLE_DEVICES=0 python GREET/main.py -dataset chameleon -ntrials 10 -sparse 0 -epochs 500 -cl_batch_size 0 -nlayers_proj 1 -alpha 0.1 -k 0 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.5 -dropedge_rate_1 0.5 -dropedge_rate_2 0.1 -lr_disc 0.001 -margin_hom 0.5 -margin_het 0.5 -cl_rounds 2 -eval_freq 20
# CUDA_VISIBLE_DEVICES=0 python GREET/main.py -dataset squirrel -ntrials 10 -sparse 0 -epochs 1000 -cl_batch_size 0 -nlayers_proj 2 -alpha 0.1 -k 0 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.1 -dropedge_rate_1 0.1 -dropedge_rate_2 0.8 -lr_disc 0.001 -margin_hom 0.1 -margin_het 0.3 -cl_rounds 2 -eval_freq 50