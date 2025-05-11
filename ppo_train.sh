export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1
 
# if JIT loading failed:
# rm -r  ~/.cache/colossalai/torch_extensions/
export CUDA_HOME=/usr/local/cuda-11.8/

# colossalai run --nproc_per_node 4 scripts/train_ppo.py configs/opensora-v1-3/train/rl_adapt_i2v.py --data-path dataset/combinatorial_out_of_template_eval_1K_nec.csv --ckpt-path ~/models/OpenSora-STDiT-v4-i2v/model.safetensors

colossalai run --nproc_per_node 4 scripts/train_ppo.py configs/opensora-v1-3/train/rl_adapt_i2v.py --data-path dataset/combinatorial_out_of_template_eval_1K_nec.csv # --ckpt-path /users/xiaokangliu/projects/Open-Sora/outputs/0005-STDiT3-XL-2/epoch193-global_step6000/

