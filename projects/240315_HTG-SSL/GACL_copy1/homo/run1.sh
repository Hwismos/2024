CUDA_VISIBLE_DEVICES=6 python main.py --dataname citeseer --epochs 15 --lr1 1e-3 --lr2 1e-2 --wd1 1e-4 --wd2 1e-2  --n_layers 1 --hid_dim 2048  --temp 0.99 --moving_average_decay 0.95 --num_MLP 1
