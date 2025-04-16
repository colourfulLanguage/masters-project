import muonium_config_andres
from read_checkpoints import load_last_n_checkpoints

from ferminet import train

import os

WORK_DIR = "../../runs"

if __name__ == "__main__":
    cfg = muonium_config_andres.get_config()
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    cfg.log.save_path = f"{WORK_DIR}/muonium_andres_inference_{slurm_job_id}"
    cfg.log.restore_path = f"{WORK_DIR}/muonium_andres_{10885095}"

    cfg.optim.iterations = 1000
    cfg.log.save_frequency = 1
    cfg.optim.optimizer = "none"

    train.train(cfg)

    load_last_n_checkpoints(f"{WORK_DIR}/muonium_andres_inference_{slurm_job_id}", 1000)
