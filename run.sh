#!/bin/bash

# Define arrays for datasets and test attacks
DATASETS=("mnist" "fashion" "cifar10" "cifar100" "svhn")
TEST_ATTACKS=("FGSM" "PGD-10" "PGD-100")

# Define flags for running with or without --clean
CLEAN_FLAGS=("--clean" "")

# Set batch size and max epochs
BATCH_SIZE=128
MAX_EPOCHS=30

# Set attack steps for training and testing
TRAIN_ATTACK_STEP=10

# Run test every test_step
TEST_STEP=1

# Set step for saving models during training
SAVE_STEP=1

# Set CUDA device to use
CUDA_DEVICE=0

# Set loss threshold for early stopping
LOSS_THRESHOLD=0.0001

# Define model architecture(s) to use
MODEL_ARCHITECTURE="preactresnet18" # can also add "preactresnet34" "preactresnet50" "preactresnet101" "preactresnet152"

for SOURCE_DATASET in "${DATASETS[@]}"
do
  if [ "$SOURCE_DATASET" == "cifar100" ]; then
    NUM_CLASSES=20
  else
    NUM_CLASSES=10
  fi

  for CLASS in $(seq 0 $(($NUM_CLASSES-1)))
  do
    for EXPOSURE_DATASET in "${DATASETS[@]}"
    do
      for CLEAN_FLAG in "${CLEAN_FLAGS[@]}"
      do
        echo "Running with params: source_dataset=$SOURCE_DATASET, source_class=$CLASS, exposure_dataset=$EXPOSURE_DATASET, test_attacks=${TEST_ATTACKS[@]}, batch_size=$BATCH_SIZE, max_epochs=$MAX_EPOCHS, clean_flag=$CLEAN_FLAG, train_attack_step=$TRAIN_ATTACK_STEP, test_step=$TEST_STEP, save_step=$SAVE_STEP, cuda_device=$CUDA_DEVICE, loss_threshold=$LOSS_THRESHOLD, model=$MODEL_ARCHITECTURE"
        python train_and_evaluate.py --source_dataset $SOURCE_DATASET --source_class $CLASS --exposure_dataset $EXPOSURE_DATASET --test_attacks ${TEST_ATTACKS[@]} --batch_size $BATCH_SIZE --max_epochs $MAX_EPOCHS --train_attack_step $TRAIN_ATTACK_STEP --test_step $TEST_STEP --save_step $SAVE_STEP --cuda_device $CUDA_DEVICE --loss_threshold $LOSS_THRESHOLD --model $MODEL_ARCHITECTURE $CLEAN_FLAG
      done
    done
  done
done