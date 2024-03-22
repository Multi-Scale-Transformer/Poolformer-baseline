DATA_PATH=/root/SharedData/datasets/CIFAR-100/cifar100
CODE_PATH=/root/workspace/metaformer # modify code path here
CKPT=/root/workspace/metaformer/output/train/20240304-132540-vformer_s18-224/checkpoint-133.pth.tar

ALL_BATCH_SIZE=4096
NUM_GPU=8
GRAD_ACCUM_STEPS=4 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS


cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--model aformer_s18 --opt lamb --lr 8e-3 --warmup-epochs 10 \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path 0.15 --head-dropout 0.0 \
--log-wandb \
--amp \
--native-amp \
--pin-mem \
--fuser nvfuser \
--use-multi-epochs-loader \
--clip-grad 2.0 \
--epochs 150 \
--load-checkpoint-unstrict $CKPT \
--num-classes 100 \