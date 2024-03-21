CUDA_VISIBLE_DEVICES=7 python main.py --dataname pubmed --epochs 60 --lr1 5e-4 --lr2 1e-2 --wd1 0 --wd2 1e-4 --n_layers 2 --hid_dim 1024  --temp 0.75 --moving_average_decay 0.99 --num_MLP 1
