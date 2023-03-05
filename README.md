# ExposureExperiment

### Datasets

---

|        | **_Cifar10_** | **_Cifar100_** | **_Mnist_** | **_F-Mnist_** | **_MVTEC_** | **_Medical_** | **_SVHN_** |
| ------ | :-----------: | :------------: | :---------: | :-----------: | :---------: | :-----------: | :--------: |
| _eps_  |     8/255     |     8/255      |    8/255    |     8/255     |    2/255    |     2/255     |   8/255    |
| _size_ |     32x32     |     32x32      |    32x32    |     32x32     |   224x224   |    224x224    |   32x32    |

### Commands

---

```sh
python train_and_evaluate.py \
        --source_dataset $SOURCE_DATASET \
        --source_class $CLASS \
        --exposure_dataset $EXPOSURE_DATASET \
        --test_attacks ${TEST_ATTACKS[@]} \
        --batch_size $BATCH_SIZE \
        --max_epochs $MAX_EPOCHS \
        --train_attack_step $TRAIN_ATTACK_STEP \
        --attack_eps $ATTACK_EPS \
        --test_step $TEST_STEP \
        --save_step $SAVE_STEP \
        --cuda_device $CUDA_DEVICE \
        --loss_threshold $LOSS_THRESHOLD \
        --model $MODEL_ARCHITECTURE \
        --checkpoints_path $CHECKPOINT_PATH \
        --output_path $OUTPUT_PATH \
        --tensorboard_path $TENSORBOARD_PATH \
        --clean \
        --force_restart
```

#### More details on parameters

`--source_dataset`: Specifies the name of the normal dataset. Valid options are 'cifar10', 'cifar100', 'mnist', 'fashion', and 'svhn'.

`--source_class`: Specifies the index of the normal class in the normal dataset.

`--exposure_dataset`: Specifies the name of the exposure dataset. Valid options are 'cifar10', 'cifar100', 'mnist', 'fashion', and 'svhn'.

`--checkpoints_path`: Specifies the path to save the model checkpoints. The default path is './Model-Checkpoints/'.

`--output_path`: Specifies the path to which plots, results, and other data will be recorded. The default path is './results/'.

`--tensorboard_path`: Specifies the path to which plots, results, and other data will be recorded. The default path is './tensorboard/'.

`--max_epochs`: Specifies the maximum number of epochs to train the model. The default value is 30.

`--batch_size`: Specifies the size of each batch input to the model. The default value is 128.

`--attack_eps`: Specifies the attack eps used for both training and testing. The default value is 8/255.

`--test_attacks`: Specifies the desired attacks for adversarial testing. For example, '--test_attacks FGSM PGD-10 PGD-100'.

`--train_attack_step`: Specifies the desired attack step for adversarial training. The default value is 10.

`--clean`: Specifies whether to perform clean training (if present) or adversarial training

`--test_step`: Specifies the frequency at which to run tests. The default value is 1.

`--save_step`: Specifies the frequency at which to save model checkpoints. The default value is 1.

`--cuda_device`: Specifies the index of your CUDA device. The default value is 0.

`--force_restart`: Specifies whether to start training from scratch (if present) or use already available checkpoints

`--loss_threshold`: Specifies the loss threshold used for early stopping. The default value is 0.001.

`--model`: Specifies the backbone used for training. Valid options are 'preactresnet18', 'preactresnet34', 'preactresnet50', 'preactresnet101', and 'preactresnet152.

### Outputs / Run

---

- Clean AUROC after normal training
- Clean AUROC after adversarial training
- Robust AUROC after adversarial training
- FID Score Support ([in this repos](https://github.com/mohammadjafari80/pytorch-fid-exposure))

### TODO

---

- MVTEC Ram Efficient support
