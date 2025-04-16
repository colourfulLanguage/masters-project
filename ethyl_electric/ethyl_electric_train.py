import ethyl_electric_config

from ferminet import train
import os

WORK_DIR = "../../runs"


if __name__ == "__main__":
    # get slurm env variable
    slurm_job_id = os.environ.get("SLURM_JOB_ID")

    cfg = ethyl_electric_config.get_config()
    cfg.log.save_path = f"{WORK_DIR}/ethyl_electric_pimc_{12994872}/"
    cfg.log.restore_path = f"{WORK_DIR}/ethyl_electric_pimc_{12994872}/"
    cfg.optim.iterations = 200_000

    train.train(cfg)
