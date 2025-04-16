import methyl_quantum_config

import read_checkpoints
from ferminet import train

import os

WORK_DIR = "../../runs"

if __name__ == "__main__":
    cfg = methyl_quantum_config.get_config()
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    cfg.log.save_path = (
        f"{WORK_DIR}/methyl_quantum_muon_inference_{12632287}_{slurm_job_id}_2000"
    )
    cfg.log.restore_path = f"{WORK_DIR}/methyl_quantum_muon_{12632287}"

    cfg.optim.iterations = 2000
    cfg.log.save_frequency = 1
    cfg.optim.optimizer = "none"

    # cfg.observables.any_muon_spin_correlation.calculate = True
    # cfg.observables.any_muon_spin_correlation.up_start = 0
    # cfg.observables.any_muon_spin_correlation.up_end = 5
    # cfg.observables.any_muon_spin_correlation.down_start = 5
    # cfg.observables.any_muon_spin_correlation.down_end = 9
    # cfg.observables.any_muon_spin_correlation.muon_start = 11
    # cfg.observables.any_muon_spin_correlation.muon_end = 12
    # cfg.observables.any_muon_spin_correlation.fixed_muon = None
    # cfg.observables.any_muon_spin_correlation.total = 12

    train.train(cfg)

    read_checkpoints.load_last_n_checkpoints(cfg.log.save_path, 2000)
