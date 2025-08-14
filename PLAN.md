### Training stability plan

- [ ] Track gradient norm and KL divergence each step; pause training if they explode
  - [ ] Record gradient norm at every training step
  - [ ] Record KL divergence at every training step
  - [ ] Log `gradient_norm` and `kl_divergence` alongside existing step metrics (console and `TRAIN_LOG_PATH`)
  - [ ] Define explicit thresholds for both metrics that constitute "explosion"
  - [ ] If a threshold is exceeded, pause training immediately and write a clear reason to the logs

- [ ] Keep more than the most recent checkpoint; ensure a checkpoint from last 5 steps is always retained
  - [ ] Save the latest checkpoint each step and also retain the checkpoint from steps-5
  - [ ] Update checkpoint cleanup so it preserves the step-5 checkpoint in addition to the latest
  - [ ] After each step, verify that both the last five checkpoints exist on disk (and that the steps-5 checkpoints are older than the latest)

