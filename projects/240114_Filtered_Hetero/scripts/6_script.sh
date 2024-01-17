#! [240112] 
CUDA_VISIBLE_DEVICES=6 python yandex/train.py --name GAT_l2 --dataset minesweeper --model GAT --num_layers 2 --device cuda:0 
CUDA_VISIBLE_DEVICES=6 python yandex/train.py --name GAT_l3 --dataset minesweeper --model GAT --num_layers 3 --device cuda:0
