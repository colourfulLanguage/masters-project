import methyl_config_andres

from ferminet import train
import os

WORK_DIR = "../../runs"


if __name__ == "__main__":
    # get slurm env variable
    slurm_job_id = os.environ.get("SLURM_JOB_ID")

    cfg = methyl_config_andres.get_config()
    cfg.log.save_path = f"{WORK_DIR}/methyl_andres_{slurm_job_id}/"
    cfg.log.restore_path = f"{WORK_DIR}/methyl_andres_{slurm_job_id}/"
    cfg.optim.iterations = 100_000

    cfg.log.save_frequency = 1000

    train.train(cfg)
