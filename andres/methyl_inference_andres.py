import methyl_config_andres

from read_checkpoints import load_last_n_checkpoints
from ferminet import train

import os

WORK_DIR = "../../runs"

if __name__ == "__main__":
    cfg = methyl_config_andres.get_config()
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    cfg.log.save_path = f"{WORK_DIR}/methyl_inference_andres_{slurm_job_id}"
    cfg.log.restore_path = f"{WORK_DIR}/methyl_andres_{10891610}"

    cfg.optim.iterations = 100_000
    cfg.log.save_frequency = 1000
    cfg.optim.optimizer = "none"

    cfg.observables.muon_spin_correlation.calculate = True

    train.train(cfg)
