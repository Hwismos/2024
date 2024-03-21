CUDA_VISIBLE_DEVICES=3 python main.py --dataset film --epoch_num 50 --lr 0.0001 --lambda_loss 1.0 --moving_average_decay 0.95 --dimension 4096 --sample_size 5 --wd2 1e-05 --num_MLP 2 --tau 1
