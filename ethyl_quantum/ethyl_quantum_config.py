import os

os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["JAX_ENABLE_X64"] = "1"

from ferminet.utils import system
from ferminet import base_config


def get_config():
    # Get default options.
    cfg = base_config.default()
    # Set up molecule
    cfg.system.electrons = (9, 8, 4, 1)

    # centered positions
    cfg.system.molecule = [
        system.Atom("C", (-1.4074, 0, 0), units="bohr"),
        system.Atom("C", (1.4074, 0, 0), units="bohr")
    ]

    # Set training hyperparameters
    cfg.batch_size = 4096
    cfg.pretrain.iterations = 0
    cfg.mcmc.burn_in= 0

    cfg.mcmc.steps = 15
    cfg.mcmc.move_width = 0.2
    cfg.optim.lr.rate = 0.2

    # put a spin up & down on each H, and then the rest on the carbons with the spare spin up on the left carbon.
    init_means = []
    # spin up on protons 
    init_means += (-2.46181114, -1.75257843,  0.15536245)
    init_means += (-2.46163842,  1.75279276,  0.15536245)
    init_means += (2.17774293,  1.67338158, -0.95053236)
    init_means += (2.17742813, -1.68148488, -0.93644871)
    # spin up on the muon
    init_means += (2.1942334 , 0.0083198 , 1.93454097)
    # spin up on carbons
    init_means += (-1.4074, 0, 0)
    init_means += (-1.4074, 0, 0)
    init_means += (1.4074, 0, 0)
    init_means += (1.4074, 0, 0)
    # spin down on carbons
    init_means += (1.4074, 0, 0)
    init_means += (1.4074, 0, 0)
    init_means += (1.4074, 0, 0)
    init_means += (1.4074, 0, 0)
    init_means += (-1.4074, 0, 0)
    init_means += (-1.4074, 0, 0)
    init_means += (-1.4074, 0, 0)
    init_means += (-1.4074, 0, 0)
    # protons
    init_means += (-2.46181114, -1.75257843,  0.15536245)
    init_means += (-2.46163842,  1.75279276,  0.15536245)
    init_means += (2.17774293,  1.67338158, -0.95053236)
    init_means += (2.17742813, -1.68148488, -0.93644871)
    # muon
    init_means += (2.1942334 , 0.0083198 , 1.93454097)

    cfg.mcmc.init_means = init_means
    cfg.mcmc.init_width = 0.1
    cfg.optim.clip_median = True

    cfg.network.network_type = "gcmnnwf"
    cfg.network.gcmnnwf.charges = (-1, -1, 1, 1)
    cfg.network.gcmnnwf.masses = (1.0, 1.0, 1836.153, 206.768)
    cfg.network.gcmnnwf.spins = (1, -1, 1, 1)
    cfg.network.gcmnnwf.kinds = (0, 0, 1, 2)
    cfg.network.gcmnnwf.exchanges = (system.Exchange.FERMI, system.Exchange.FERMI, system.Exchange.BOLTZMANN, system.Exchange.BOLTZMANN)

    cfg.network.psiformer.use_layer_norm = True

    cfg.system.make_local_energy_fn = "ferminet.af.hamiltonian.local_energy"
    cfg.system.make_local_energy_kwargs = {
        "species_charges": cfg.network.gcmnnwf.charges,
        "species_masses": cfg.network.gcmnnwf.masses,
    }

    cfg.log.save_frequency = 1000

    return cfg
