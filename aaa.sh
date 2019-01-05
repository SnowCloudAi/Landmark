#!/usr/bin/env bash
source activate detection

# train_code: 0 stage-1 from scratch lr: (1e-3, 1e-7) adam epoch 100 loss: normal data: iwflw(pad_0.1, random_pad_0.1, scale_0.25,4.0)
# info: train from scratch to recure model-18
# CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 -H localhost:8 -bind-to none -x NCCL_DEBUG=INFO -map-by slot -mca pml ob1 -mca btl ^openib python train_boundary.py --tensorboard-path ../TB/TB0 --save-params-path ../param/exp0,0 --epochs 100 --lr-epoch 96 --mixup-epoch 0 --per-batch 20 --pretrained 0  --loss-type 0 --lr-mode cycleCosine

# train_code: 1 stage-1 from scratch lr: (1e-3, reduceonplateau) adam epoch 100 loss: normal data: iwflw(pad_0.1, random_pad_0.1, scale_0.25,4.0)
# info: exp-0 model crashed while changing lr with cycle-lr, trying to replace it with reduceonplateau
# CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 -H localhost:8 -bind-to none -x NCCL_DEBUG=INFO -map-by slot -mca pml ob1 -mca btl ^openib python train_boundary_single_card.py --tensorboard-path ../TB/TB1 --save-params-path ../param/exp1,0 --epochs 100 --lr-epoch 96 --mixup-epoch 0 --per-batch 18 --pretrained 0  --loss-type 0 --lr-mode cycleCosine

# train_code: 2 stage-1 finetune-1-12 lr: (1e-4, reduceonplateau) adam epoch 100 loss: normal data: iwflw(pad_0.1, random_pad_0.1, scale_0.25,4.0)
# info: exp-1 crashed again. trying to finetune it with the model before crashing
# CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 -H localhost:8 -bind-to none -x NCCL_DEBUG=INFO -map-by slot -mca pml ob1 -mca btl ^openib python train_boundary_single_card.py --tensorboard-path ../TB/TB2 --save-params-path ../param/exp2,0 --epochs 100 --lr-epoch 96 --mixup-epoch 0 --per-batch 20 --pretrained 0  --loss-type 0 --lr-mode cycleCosine

# train_code: 3 stage-1 finetune-2-18 lr: (1e-5, reduceonplateau) adam epoch 100 loss: normal data: iwflw(trans_0.1, rotate30, scale0.25_4,4.0, train_mod) 
# info: change augment and keep finetuning for better behavior on test dataset
# CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 -H localhost:8 -bind-to none -x NCCL_DEBUG=INFO -map-by slot -mca pml ob1 -mca btl ^openib python train_boundary_single_card.py --tensorboard-path ../TB/TB3 --save-params-path ../param/exp3,0 --epochs 100 --lr-epoch 96 --mixup-epoch 0 --per-batch 20 --pretrained 0 --loss-type 0 --lr-mode cycleCosine

# train_code: 4 stage-1(freeze, 3-62), stage-2(from scratch) lr: (1e-3, reduceonplateau) adam epoch 100 loss: wingloss(10, 2) data: iwflw(trans_0.1, rotate30, scale0.25_4,4.0, train_mod) 
# info: start stage-2 training from scratch based on exp-3's stage-1 model 
# CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 -H localhost:8 -bind-to none -x NCCL_DEBUG=INFO -map-by slot -mca pml ob1 -mca btl ^openib python train_landmarks_single_card.py --tensorboard-path ../TB/TB4 --save-params-path ../param/exp4,0 --epochs 100 --lr-epoch 96 --mixup-epoch 0 --per-batch 20 --pretrained 0 --loss-type 0 --lr-mode cycleCosine

# train_code: 5 stage-1(freeze, 3-62), stage-2(finetune 4-7) lr: (1e-4, reduceonplateau, patience_2, factor_0.5) adam epoch 100 loss: wingloss(10, 2) data: iwflw(trans_0.1, rotate30, scale0.25_4,4.0, train_mod) 
# info: start stage-2 training from scratch based on exp-3's stage-1 model 
# CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 -H localhost:8 -bind-to none -x NCCL_DEBUG=INFO -map-by slot -mca pml ob1 -mca btl ^openib python train_landmarks_single_card.py --tensorboard-path ../TB/TB5 --save-params-path ../param/exp5,0 --epochs 100 --lr-epoch 96 --mixup-epoch 0 --per-batch 20 --pretrained 0 --loss-type 0 --lr-mode cycleCosine

# train_code: 6 stage-1(finetune, 3-62), stage-2(finetune 5-100) lr: (1e-5-1e-8, 1e-5-1e-8, reduceonplateau, patience_4, factor_0.5) adam epoch 100 loss: wingloss(10, 2) data: iwflw(trans_0.1, rotate30, scale0.25_4,4.0, train_mod)
# info: start two stage training finetune for better together result heatmap loss + normLoss use newly made normed loss
# CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 -H localhost:8 -bind-to none -x NCCL_DEBUG=INFO -map-by slot -mca pml ob1 -mca btl ^openib python train_together_single_card.py --tensorboard-path ../TB/TB6 --save-params-path ../param/exp6,0 --epochs 100 --lr-epoch 96 --mixup-epoch 0 --per-batch 10 --pretrained 0 --loss-type 0 --lr-mode cycleCosine

# train_code: 7 stage-1(freeze, 6-100), stage-2(from scratch) lr: (1e-3-1e-5, reduceonplateau, patience_5, factor_0.5) adam epoch 100 loss: L1 loss
# info: compare l1 loss with normed l1 loss in stage 2
# CUDA_VISIBLE_DEVICES=0 mpirun -np 1 -H localhost:8 -bind-to none -x NCCL_DEBUG=INFO -map-by slot -mca pml ob1 -mca btl ^openib python train_landmarks_single_card.py --tensorboard-path ../TB/TB7 --save-params-path ../param/exp7,0 --epochs 100 --lr-epoch 96 --mixup-epoch 0 --per-batch 20 --pretrained 0 --loss-type 0 --lr-mode cycleCosine --train-code 7

# train_code: 8 stage-1(freeze, 6-100), stage-2(from scratch) lr: (1e-3-1e-5, reduceonplateau, patience_5, factor_0.5) adam epoch 100 loss: normed L1 loss
# info: compare l1 loss with normed l1 loss in stage 2
# CUDA_VISIBLE_DEVICES=1 mpirun -np 1 -H localhost:8 -bind-to none -x NCCL_DEBUG=INFO -map-by slot -mca pml ob1 -mca btl ^openib python train_landmarks_single_card.py --tensorboard-path ../TB/TB8 --save-params-path ../param/exp8,0 --epochs 100 --lr-epoch 96 --mixup-epoch 0 --per-batch 20 --pretrained 0 --loss-type 1 --lr-mode cycleCosine --train-code 8

# train_code: 9 stage-1(freeze, 6-100), stage-2(from scratch) lr: (1e-4-1e-5, reduceonplateau, patience_5, factor_0.5) adam epoch 100 loss: normed L1 loss
# info: finetune with wingloss with balanced datasets
# CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 -H localhost:8 -bind-to none -x NCCL_DEBUG=INFO -map-by slot -mca pml ob1 -mca btl ^openib python train_together_single_card.py --tensorboard-path ../TB/TB9 --save-params-path ../param/exp9,0 --epochs 100 --lr-epoch 96 --mixup-epoch 0 --per-batch 18 --pretrained 0 --loss-type 0 --lr-mode cycleCosine --train-code 9

# train_code: 10 stage-1(finetune, 8-42), stage-2(finetune, 8-42) lr: (1e-5-1e-7, 1e-4-1e-6, reduceonplateau, patience_5, factor_0.5) adam epoch 100 loss: wingloss
# info: finetune for better result
# CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 -H localhost:8 -bind-to none -x NCCL_DEBUG=INFO -map-by slot -mca pml ob1 -mca btl ^openib python train_together_single_card.py --tensorboard-path ../TB/TB10 --save-params-path ../param/exp10,0 --epochs 100 --lr-epoch 96 --mixup-epoch 0 --per-batch 18 --pretrained 0 --loss-type 0 --lr-mode cycleCosine --train-code 10

# train_code: 10 stage-1(finetune, 8-42), stage-2(finetune, 8-42) lr: (1e-5-1e-7, 1e-4-1e-6, reduceonplateau, patience_5, factor_0.5) adam epoch 100 loss: wingloss
# info: finetune for better result
CUDA_VISIBLE_DEVICES=0,1 mpirun -np 2 -H localhost:8 -bind-to none -x NCCL_DEBUG=INFO -map-by slot -mca pml ob1 -mca btl ^openib python train_landmarks_single_card.py --tensorboard-path ../TB/TB11 --save-params-path ../param/exp11,0 --epochs 100 --lr-epoch 96 --mixup-epoch 0 --per-batch 100 --pretrained 0 --loss-type 3 --lr-mode cycleCosine --train-code 11

