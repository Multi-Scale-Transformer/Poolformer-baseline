DATA_PATH=/root/SharedData/datasets/ImageNet1k
CODE_PATH=/root/workspace/metaformer # modify code path here
INIT_CKPT=/root/workspace/metaformer/output/train/20240322-070252-csformer_s18_in21k-224/checkpoint-24.pth.tar

ALL_BATCH_SIZE=1024
NUM_GPU=8
GRAD_ACCUM_STEPS=2 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS


cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--model csformer_s18_in21k --img-size 224 --epochs 30 --opt lamb --lr 1e-4 --sched None \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--load-checkpoint-unstrict $INIT_CKPT \
--mixup 0 --cutmix 0 \
--drop-path 0.3 --head-dropout 0.4 \
--model-ema --model-ema-decay 0.9999 \
--log-wandb \
--amp \
--native-amp \
--pin-mem \
--use-multi-epochs-loader \