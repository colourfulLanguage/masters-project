import ethyl_quantum_config

import read_checkpoints
from ferminet import train

import os

WORK_DIR = "../../runs"

if __name__ == "__main__":
    cfg = ethyl_quantum_config.get_config()
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    cfg.log.save_path = f"{WORK_DIR}/ethyl_quantum_muon_inference_{12564938}"
    cfg.log.restore_path = f"{WORK_DIR}/ethyl_quantum_muon_{12564938}"

    cfg.mcmc.steps = 10
    cfg.optim.iterations = 100
    cfg.log.save_frequency = 1
    cfg.optim.optimizer = "none"

    # cfg.observables.ethyl_muon_spin_correlation.calculate = True

    train.train(cfg)

    read_checkpoints.load_last_n_checkpoints(cfg.log.save_path, 100)
