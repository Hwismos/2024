CUDA_VISIBLE_DEVICES=2 python main.py --dataset squirrel  --epoch_num 40  --lr 0.0001 --lambda_loss 1 --moving_average_decay 0.97 --dimension 8192 --sample_size 10 --wd2 1e-05 --num_MLP 1 --tau 0.25 
