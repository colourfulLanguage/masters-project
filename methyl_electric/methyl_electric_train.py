import methyl_electric_config

from ferminet import train
import os

WORK_DIR = "../../runs"


if __name__ == "__main__":
    # get slurm env variable
    slurm_job_id = os.environ.get("SLURM_JOB_ID")

    cfg = methyl_electric_config.get_config()
    cfg.log.save_path = f"{WORK_DIR}/methyl_electric_extended_{12995673}/"
    cfg.log.restore_path = f"{WORK_DIR}/methyl_electric_extended_{12995673}/"
    cfg.optim.iterations = 400_000
    # methyl config 8964457
    train.train(cfg)
