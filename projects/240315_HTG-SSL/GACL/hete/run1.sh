CUDA_VISIBLE_DEVICES=1 python main.py --dataset chameleon --epoch_num 30 --lr 0.0001 --lambda_loss 1.0 --moving_average_decay 0.999 --dimension 4096 --sample_size 5 --wd2 1e-05 --num_MLP 1 --tau 0.5
