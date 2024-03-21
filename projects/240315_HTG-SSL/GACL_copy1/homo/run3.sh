CUDA_VISIBLE_DEVICES=0 python main.py --dataname comp --epochs 50 --lr1 5e-4 --lr2 1e-2 --wd1 0 --wd2 5e-4 --n_layers 1 --hid_dim 2048  --temp 0.99 --moving_average_decay 0.99 --num_MLP 1
