#/bin/sh
export MODELSCOPE_CACHE=/root/gpufree-data/cache
python train_video.py \
    --exp-name video_run_001 \
    --model SiT-B/1 \
    --data-dir videos_test/video_latents/ \
    --use-video-dataset \
    --use-json-dataset \
    --label-file videos/label_file.json \
    --videos-subdir videos \
    --latents-subdir vae-in \
    --batch-size 24 \
    --gradient-accumulation-steps 1 \
    --num-workers 0 \
    --mixed-precision bf16 \
    --learning-rate 1e-4 \
    --max-train-steps 100000 \
    --checkpointing-steps 10000 \
    --allow-tf32
