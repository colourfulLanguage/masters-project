import methyl_electric_config

import read_checkpoints
from ferminet import train

import os

WORK_DIR = "../../runs"

if __name__ == "__main__":
    cfg = methyl_electric_config.get_config()
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    cfg.log.save_path = (
        f"{WORK_DIR}/methyl_electric_inference_{12438954}_{slurm_job_id}_120_000"
    )
    cfg.log.restore_path = f"{WORK_DIR}/methyl_electric_{12438954}"
    cfg.optim.iterations = 120_000
    cfg.log.save_frequency = 5000
    cfg.optim.optimizer = "none"

    cfg.observables.any_muon_spin_correlation.calculate = True
    cfg.observables.any_muon_spin_correlation.up_start = 0
    cfg.observables.any_muon_spin_correlation.up_end = 5
    cfg.observables.any_muon_spin_correlation.down_start = 5
    cfg.observables.any_muon_spin_correlation.down_end = 9
    cfg.observables.any_muon_spin_correlation.muon_start = -1
    cfg.observables.any_muon_spin_correlation.muon_end = -1
    cfg.observables.any_muon_spin_correlation.fixed_muon = (-0.02268, 0.72225, -0.00302)
    cfg.observables.any_muon_spin_correlation.total = 9

    train.train(cfg)
