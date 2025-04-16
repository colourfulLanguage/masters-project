import os

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["JAX_ENABLE_X64"] = "1"

from ferminet.utils import system
from ferminet import base_config
from ferminet.constants import MUON_MASS

def get_config():
    # Get default options.
    cfg = base_config.default()
    # Set up molecule
    cfg.system.particles = (5, 4, 1)

    cfg.system.molecule = [
        system.Atom("C", (1.90428, 1.34926, 0.28818), units="bohr"),
        system.Atom("H", (2.30584, 3.34047, 0.54178), units="bohr"),
        system.Atom("H", (3.42929, -0.01587, 0.32636), units="bohr"),
    ]

    # Set training hyperparameters
    cfg.batch_size = 4096
    cfg.pretrain.iterations = 0

    cfg.system.charges = (-1, -1, 1)
    cfg.system.masses = (1.0, 1.0, MUON_MASS)

    cfg.optim.lr.rate = 0.01
    cfg.optim.clip_median = True

    cfg.optim.laplacian = "folx"

    return cfg
