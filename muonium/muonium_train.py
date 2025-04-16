import muonium_config_andres as muonium_config

from ferminet import train
import os

WORK_DIR = "../../runs"


if __name__ == "__main__":
    slurm_job_id = os.environ.get("SLURM_JOB_ID")

    cfg = muonium_config.get_config()
    cfg.log.save_path = f"{WORK_DIR}/muonium_{slurm_job_id}/"
    cfg.log.restore_path = f"{WORK_DIR}/muonium_{slurm_job_id}/"
    cfg.optim.iterations = 300_000

    train.train(cfg)
