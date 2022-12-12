# Optimization model based on attention for Few-shot Learning 
- By Ruizhi Liao, Junhai zhai.
- This repo is the pytorch implementation of [Optimization model based on attention for Few-shot Learning ]

## Model
<div align="center">
   <img src='imgs/MetaLearner.png' height="400px"><img src='imgs/BaseLearner.png' height="400px">
</div>

## Prerequisites
- python 3+
- pytorch 0.4+ (developed on 1.0.1 with cuda 9.0)
- [pillow](https://pillow.readthedocs.io/en/stable/installation.html)
- [tqdm](https://tqdm.github.io/) (a nice progress bar)

## Data
- Mini-Imagenet as described [here](https://github.com/twitter/meta-learning-lstm/tree/master/data/miniImagenet)
  - You can download it from [here](https://drive.google.com/file/d/1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR/view?usp=sharing) (~2.7GB, google drive link)

## Preparation
- Make sure Mini-Imagenet is split properly. For example:
  ```
  - data/
    - miniImagenet/
      - train/
        - n01532829/
          - n0153282900000005.jpg
          - ...
        - n01558993/
        - ...
      - val/
        - n01855672/
        - ...
      - test/
        - ...
  - main.py
  - ...
  ```
  - It'd be set if you download and extract Mini-Imagenet from the link above
- Check out `scripts/train_5s_5c.sh`, make sure `--data-root` is properly set

## Run
For 5-shot, 5-class training, run
```bash
bash scripts/train_5s_5c.sh
```
Hyper-parameters are referred to the [author's repo](https://github.com/twitter/meta-learning-lstm).

For 5-shot, 5-class evaluation, run *(remember to change `--resume` and `--seed` arguments)*
```bash
bash scripts/eval_5s_5c.sh
```

## Notes
- Training with the default settings takes ~2.5 hours on a single Titan Xp while occupying ~2GB GPU memory.
- The implementation replicates two learners similar to the author's repo:
  - `learner_w_grad` functions as a regular model, get gradients and loss as inputs to meta learner.
  - `learner_wo_grad` constructs the graph for meta learner:
    - All the parameters in `learner_wo_grad` are replaced by `cI` output by meta learner.
    - `nn.Parameters` in this model are casted to `torch.Tensor` to connect the graph to meta learner.
- Several ways to **copy** a parameters from meta learner to learner depends on the scenario:
  - `copy_flat_params`: we only need the parameter values and keep the original `grad_fn`.
  - `transfer_params`: we want the values as well as the `grad_fn` (from `cI` to `learner_wo_grad`).
    - `.data.copy_` v.s. `clone()` -> the latter retains all the properties of a tensor including `grad_fn`.
    - To maintain the batch statistics, `load_state_dict` is used (from `learner_w_grad` to `learner_wo_grad`).

## Acknowledement
- This code borrows heavily from the [meta-learning-lstm](https://github.com/twitter/meta-learning-lstm) framework.

