export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1
 
colossalai run --nproc_per_node 4 scripts/train_ppo.py configs/opensora-v1-3/train/rl_adapt_i2v.py --data-path dataset/combinatorial_out_of_template_eval_1K_nec.csv --ckpt-path ~/models/OpenSora-STDiT-v4-i2v/model.safetensors