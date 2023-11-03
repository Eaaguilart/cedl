CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 template.py --lr 1e-1 --num_epochs 120 --memory_size 2000 --fixed_memory --lambda_kd 0.5 |& tee log.txt
