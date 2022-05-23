from pydantic import StrictStr, StrictInt, StrictBool
from tensorfn.config import (
    Checker,
    Config,
    DataLoader,
    Instance,
    MainConfig,
    Optimizer,
    Scheduler,
)


class Training(Config):
    datasource: Instance
    loader: DataLoader
    optimizer: Optimizer
    scheduler: Scheduler

    render_rays: Instance
    ndc: StrictBool
    n_iter: StrictInt


class Evaluate(Config):
    eval_freq: StrictInt = 10000

    render_rays: Instance


class NeRFConfig(MainConfig):
    model: Instance
    training: Training
    evaluate: Evaluate

    log_freq: StrictInt = 10
    checker: Checker = Checker()
    logger: StrictStr = "rich"

    out: StrictStr = "samples.mp4"
