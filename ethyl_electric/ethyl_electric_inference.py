import ethyl_electric_config

import read_checkpoints
from ferminet import train

import os

WORK_DIR = "../../runs"

if __name__ == "__main__":
    cfg = ethyl_electric_config.get_config()
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    cfg.log.save_path = (
        f"{WORK_DIR}/ethyl_electric_inference_{11974154}_{slurm_job_id}_100_000"
    )
    cfg.log.restore_path = f"{WORK_DIR}/ethyl_electric_{11974154}"

    cfg.optim.iterations = 100_000
    cfg.log.save_frequency = 5000
    cfg.optim.optimizer = "none"

    cfg.observables.any_muon_spin_correlation.calculate = True
    cfg.observables.any_muon_spin_correlation.up_start = 0
    cfg.observables.any_muon_spin_correlation.up_end = 9
    cfg.observables.any_muon_spin_correlation.down_start = 9
    cfg.observables.any_muon_spin_correlation.down_end = 17
    cfg.observables.any_muon_spin_correlation.muon_start = -1
    cfg.observables.any_muon_spin_correlation.muon_end = -1
    cfg.observables.any_muon_spin_correlation.fixed_muon = (
        2.222035,
        -1.861475,
        -0.393255,
    )
    cfg.observables.any_muon_spin_correlation.total = 17

    train.train(cfg)

    # read_checkpoints.load_last_n_checkpoints(f"{WORK_DIR}/ethyl_electric_inference_{slurm_job_id}", 1000)
