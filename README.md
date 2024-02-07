# MotionPredict-GuidanceReg
Official implementation of our IJCNN 2024 paper:

Enhancing Human Motion Prediction with Guidance Regularization via Fusion Feature Distillation

## Pre-train the FE-Net
- Pre-tain on Human3.6M:
`-- python main_h36m_3d.py --kernel_size 10 --dct_n 35 --input_n 10 --output_n 25 -- priv_n 10 --skip_rate 1 --batch_size 16 --test_batch_size 32 --in_features 66 --cuda_idx cuda:0 --d_model 16 --lr_now 0.005 --epoch 100 --test_sample_num -1`

## Train the DSP-Net
- Train on Human3.6M:
`-- python main_h36m_3d_fp.py --kernel_size 10 --dct_n 35 --input_n 10 --output_n 25 -- priv_n 10 --skip_rate 1 --batch_size 16 --test_batch_size 32 --in_features 66 --cuda_idx cuda:0 --d_model 16 --lr_now 0.005 --epoch 50 --test_sample_num -1`

