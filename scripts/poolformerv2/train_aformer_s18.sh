DATA_PATH=/root/workspace/datasets/CIFAR-100-dataset
CODE_PATH=/root/workspace/Poolformer-baseline # modify code path here



ALL_BATCH_SIZE=128
NUM_GPU=4
GRAD_ACCUM_STEPS=16 # Adjust according to your GPU numbers and memory size.
let BATCH_SIZE=ALL_BATCH_SIZE/NUM_GPU/GRAD_ACCUM_STEPS


cd $CODE_PATH && sh distributed_train.sh $NUM_GPU $DATA_PATH \
--model aformer_s18 --opt lamb --lr 8e-3 --warmup-epochs 20 \
-b $BATCH_SIZE --grad-accum-steps $GRAD_ACCUM_STEPS \
--drop-path 0.15 --head-dropout 0.0 \
--log-wandb \
--amp \
--native-amp \
--pin-mem \
--fuser nvfuser \
--use-multi-epochs-loader \