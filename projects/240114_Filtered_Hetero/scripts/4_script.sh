#! [240112] 
# CUDA_VISIBLE_DEVICES=4 python yandex/train.py --name GAT_l1 --dataset roman-empire --model GAT --num_layers 1 --device cuda:0
# CUDA_VISIBLE_DEVICES=4 python yandex/train.py --name GAT_l2 --dataset roman-empire --model GAT --num_layers 2 --device cuda:0 
# CUDA_VISIBLE_DEVICES=4 python yandex/train.py --name GAT_l3 --dataset roman-empire --model GAT --num_layers 3 --device cuda:0

# '''
# parser.add_argument('--dataset', type=str, default='roman-empire',
#                 choices=['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions',
#                             'squirrel', 'squirrel-directed', 'squirrel-filtered', 'squirrel-filtered-directed',
#                             'chameleon', 'chameleon-directed', 'chameleon-filtered', 'chameleon-filtered-directed',
#                             'actor', 'texas', 'texas-4-classes', 'cornell', 'wisconsin'])
# '''


CUDA_VISIBLE_DEVICES=4 python GREET/main.py -dataset chameleon -ntrials 10 -sparse 0 -epochs 500 -cl_batch_size 0 -nlayers_proj 1 -alpha 0.1 -k 0 -maskfeat_rate_1 0.1 -maskfeat_rate_2 0.5 -dropedge_rate_1 0.5 -dropedge_rate_2 0.1 -lr_disc 0.001 -margin_hom 0.5 -margin_het 0.5 -cl_rounds 2 -eval_freq 20