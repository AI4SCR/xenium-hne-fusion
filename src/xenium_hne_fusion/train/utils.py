from xenium_hne_fusion.train.config import Config


def set_fast_dev_run_settings(cfg: Config) -> Config:
    cfg.wandb.project = 'debug'
    cfg.data.batch_size = 2
    cfg.data.num_workers = 0
    cfg.data.prefetch_factor = None
    cfg.trainer.max_epochs = 3
    cfg.trainer.limit_train_batches = 2
    cfg.trainer.limit_val_batches = 2
    cfg.trainer.limit_test_batches = 2
    cfg.lit.num_warmup_epochs = 2
    return cfg
