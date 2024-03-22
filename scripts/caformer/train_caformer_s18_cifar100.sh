DATA_PATH=/root/SharedData/datasets/CIFAR-100/cifar100
CODE_PATH=/root/workspace/metaformer # modify code path here


ALL_BATCH_SIZE=1024
NUM_GPU=8
GRAD_ACCUM_STEPS=16 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS


cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--model hformer_s18 --opt lamb --lr 8e-3 --warmup-epochs 20 \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path 0.15 --head-dropout 0.0 \
--log-wandb \
--amp \
--native-amp \
--pin-mem \
--clip-grad 2.0 \
--num-classes 100 \