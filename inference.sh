export CUDA_VISIBLE_DEVICES=3

python scripts/inference_i2v.py configs/opensora-v1-3/inference/v2v.py --prompt 'A 2D physical environment involves multiple objects to free fall then collide with each other.{"reference_path": "/users/xiaokangliu/projects/Open-Sora/assets/eval_images/995-frame_1.jpg"}'