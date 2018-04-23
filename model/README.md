# Main code for "Controllable Video Generation with Sparse Trajectories", CVPR'18.

1. Setup dataset paths with `./dataset/utils/set_dataset_path.py`. **READ THE FILE FOR MORE HINTS**
2. Run `train_[dataset_name].py` to train the video generation model.
-- By default, the model takes 1-5 trajectories as input, which is suitable for human evaluation.
-- You should increase the number of input trajectories to 10 for quantitative quality (PSNR, SSIM) evaluation. Too few trajectories brings too much ambiguity.
3. Run `aeeval_[dataset_name].py` to evaluate the model on testsets using PSNR and SSIM metrics.
-- Note that our model is not designed for video prediction. Results are for reference only.
-- Our work aims at generating video clips in a user-controllable manner.
4. For examples on how to build GUI for user evaluation, refer to a simplified example `guieval_rp.py`.
-- Edit `./reader/*.py` to match dataset paths
-- First click defines the start point of a motion vector
-- Second click defines end point of the motion vector
-- Next click add a new vector
-- Left click outside canvas to clear all the vectors
-- Press right mouse button to go to the next image

- Requires PyTorch3 for train/test and visdom for monitoring.

**Warning: The code is provided in its original form without any cleanup. Read each program before running. Most files are self-explainable.**
