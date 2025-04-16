import ethyl_config

from ferminet import train

import os

WORK_DIR = "../../runs"

if __name__ == "__main__":
    cfg = ethyl_config.get_config()
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    cfg.log.save_path = f"{WORK_DIR}/ethyl_inference_{11783960}_{slurm_job_id}_100_000"
    cfg.log.restore_path = f"{WORK_DIR}/ethyl_{11783960}"

    cfg.optim.iterations = 100_000
    cfg.log.save_frequency = 10_000
    cfg.optim.optimizer = "none"

    cfg.observables.any_muon_spin_correlation.calculate = True
    cfg.observables.any_muon_spin_correlation.up_start = 0
    cfg.observables.any_muon_spin_correlation.up_end = 9
    cfg.observables.any_muon_spin_correlation.down_start = 9
    cfg.observables.any_muon_spin_correlation.down_end = 17
    cfg.observables.any_muon_spin_correlation.muon_start = 17
    cfg.observables.any_muon_spin_correlation.muon_end = 18
    cfg.observables.any_muon_spin_correlation.fixed_muon = None
    cfg.observables.any_muon_spin_correlation.total = 18

    train.train(cfg)

    # read_checkpoints.load_last_n_checkpoints(f"{WORK_DIR}/ethyl_inference_{slurm_job_id}", 2000)
