export MODELSCOPE_CACHE=/root/gpufree-data/cache
random_number=$((RANDOM % 100 + 1200))
NUM_GPUS=1
STEP="0080000"
SAVE_PATH="/root/gpufree-data/OmegaDiT-master/exps/video_run_001"
NUM_STEP=250
MODEL_SIZE='B'
CFG_SCALE=2.5
CLS_CFG_SCALE=2.5
GH=0.85
PATCH_SIZE=1
PATH_DROP=True
export NCCL_P2P_DISABLE=1

python -m torch.distributed.launch --master_port=$random_number --nproc_per_node=$NUM_GPUS generate_video.py \
  --model SiT-${MODEL_SIZE}/${PATCH_SIZE} \
  --num-fid-samples 50000 \
  --ckpt ${SAVE_PATH}/checkpoints/${STEP}.pt \
  --path-type=linear \
  --projector-embed-dims=768 \
  --per-proc-batch-size=4 \
  --mode=sde \
  --num-steps=${NUM_STEP} \
  --cfg-scale=${CFG_SCALE} \
  --cls-cfg-scale=${CLS_CFG_SCALE} \
  --guidance-high=${GH} \
  --sample-dir ${SAVE_PATH}/checkpoints \
  --cls=768 \












