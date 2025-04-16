import methyl_config

from ferminet import train

import os

WORK_DIR = "../../runs"

if __name__ == "__main__":
    cfg = methyl_config.get_config()
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    cfg.log.save_path = f"{WORK_DIR}/methyl_inference_{10864758}_{slurm_job_id}_120_000"
    cfg.log.restore_path = f"{WORK_DIR}/methyl_{10864758}"

    cfg.optim.iterations = 120_000
    cfg.log.save_frequency = 5000
    cfg.optim.optimizer = "none"

    cfg.observables.any_muon_spin_correlation.calculate = True
    cfg.observables.any_muon_spin_correlation.up_start = 0
    cfg.observables.any_muon_spin_correlation.up_end = 5
    cfg.observables.any_muon_spin_correlation.down_start = 5
    cfg.observables.any_muon_spin_correlation.down_end = 9
    cfg.observables.any_muon_spin_correlation.muon_start = 9
    cfg.observables.any_muon_spin_correlation.muon_end = 10
    cfg.observables.any_muon_spin_correlation.fixed_muon = None
    cfg.observables.any_muon_spin_correlation.total = 10

    train.train(cfg)
