import os

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["JAX_ENABLE_X64"] = "1"

from ferminet.utils import system
from ferminet import base_config

import os

def get_config():
    # Get default options.
    cfg = base_config.default()
    # Set up molecule
    cfg.system.electrons = (9, 8)

    ## centered positions
    #cfg.system.molecule = [
    #    system.Atom("C", (-1.37222, 0.08457, -0.30103), units="bohr"),
    #    system.Atom("C", (1.37222, -0.08457, 0.30103), units="bohr"),
    #    system.Atom("H", (-2.01171, 0.79398, -2.11631), units="bohr"),
    #    system.Atom("H", (-2.77517, -0.77546, 0.92370), units="bohr"),
    #    system.Atom("H", (1.71729, -0.03204, 2.34440), units="bohr"),
    #    system.Atom("H", (2.22204, -1.86148, -0.39325), units="bohr"),
    #    system.Atom("H", (2.44843, 1.45746, -0.57146), units="bohr"),
    #]

    cfg.system.molecule = [
        system.Atom("C", (-1.38989641, 0.0, 0.0), units="bohr"),
        system.Atom("C", (1.38989641, 0.0, 0.0), units="bohr"),
        system.Atom("H", (2.23233584, 0.0, 2.12775911), units="bohr"),
        system.Atom("H", (2.15196298, 1.75784889, 1.03364194), units="bohr"),
        system.Atom("H", (2.15196298, -1.75784889, 1.03364194), units="bohr"),
        system.Atom("H", (-2.52548893, 1.81732791, 0), units="bohr"),
        system.Atom("H", (-2.52548893, -1.81732791, 0), units="bohr"),
    ]
    
    # Set training hyperparameters
    cfg.batch_size = 4096
    cfg.pretrain.iterations = 0

    #cfg.optim.lr.rate = 0.01
    cfg.optim.clip_median = True

    cfg.network.network_type = "gcmnnwf"
    cfg.network.gcmnnwf.charges = (-1, -1)
    cfg.network.gcmnnwf.masses = (1.0, 1.0)
    cfg.network.gcmnnwf.spins = (1, -1)
    cfg.network.gcmnnwf.kinds = (0, 0)
    cfg.network.psiformer.use_layer_norm = True

    cfg.system.make_local_energy_fn = "ferminet.af.hamiltonian.local_energy"
    cfg.system.make_local_energy_kwargs = {
        "species_charges": cfg.network.gcmnnwf.charges,
        "species_masses": cfg.network.gcmnnwf.masses,
    }

    cfg.log.save_frequency = 1000

    return cfg
