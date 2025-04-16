import os
import ethyl_quantum_config
from ferminet import train


WORK_DIR = "../../runs"


if __name__ == "__main__":
    # get slurm env variable
    slurm_job_id = os.environ.get("SLURM_JOB_ID")

    cfg = ethyl_quantum_config.get_config()
    cfg.log.save_path = f"{WORK_DIR}/ethyl_quantum_muon_{12555706}/"
    cfg.log.restore_path = f"{WORK_DIR}/ethyl_quantum_muon_{12555706}/"
    cfg.optim.iterations = 1000000

    train.train(cfg)
