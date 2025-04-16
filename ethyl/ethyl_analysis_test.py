import numpy as np
import matplotlib.pyplot as plt
import os

jnp = np
WORK_DIR = "../../runs"


UP = 9
DOWN = 8
ELECTRONS = UP + DOWN
TOTAL = ELECTRONS + 1


def observable(position_data):

    nbins = 4000
    lim = 20

    spin_up_bin = np.zeros(nbins)
    spin_down_bin = np.zeros(nbins)

    for i in range(position_data.shape[0]):
        data = position_data[i : i + 1, :, :, :]

        data = data.reshape(4096, TOTAL * 3)

        spin_up_electron_samples = data[:, : UP * 3].reshape(4096, -1, 3)
        spin_down_electron_samples = data[:, UP * 3 : ELECTRONS * 3].reshape(
            4096, -1, 3
        )
        muon_samples = data[:, ELECTRONS * 3 :].reshape(4096, -1, 3)

        spin_up_electron_muon_seperations_cart = spin_up_electron_samples - jnp.tile(
            muon_samples, (1, UP, 1)
        )
        spin_down_electron_muon_seperations_cart = (
            spin_down_electron_samples - jnp.tile(muon_samples, (1, DOWN, 1))
        )

        spin_up_electron_muon_seperations = jnp.linalg.norm(
            spin_up_electron_muon_seperations_cart, axis=2
        )
        spin_down_electron_muon_seperations = jnp.linalg.norm(
            spin_down_electron_muon_seperations_cart, axis=2
        )

        single_spin_up_bin, _ = jnp.histogram(
            spin_up_electron_muon_seperations, bins=nbins, range=(0, lim)
        )
        single_spin_down_bin, _ = jnp.histogram(
            spin_down_electron_muon_seperations, bins=nbins, range=(0, lim)
        )

        spin_up_bin += single_spin_up_bin
        spin_down_bin += single_spin_down_bin

    return spin_up_bin.astype(int), spin_down_bin.astype(int)


def new_standard(position_data):

    position_data = position_data.reshape(-1, 4096, 3 * TOTAL)
    slice_num = position_data.shape[0]

    spin_up_electron_samples = position_data[:, :, : UP * 3].reshape(
        slice_num, 4096, -1, 3
    )
    spin_down_electron_samples = position_data[:, :, UP * 3 : ELECTRONS * 3].reshape(
        slice_num, 4096, -1, 3
    )
    muon_samples = position_data[:, :, ELECTRONS * 3 :].reshape(slice_num, 4096, -1, 3)

    spin_up_electron_muon_seperations_cart = spin_up_electron_samples - np.tile(
        muon_samples, (1, 1, UP, 1)
    )
    spin_down_electron_muon_seperations_cart = spin_down_electron_samples - np.tile(
        muon_samples, (1, 1, DOWN, 1)
    )

    spin_up_electron_muon_seperations = np.linalg.norm(
        spin_up_electron_muon_seperations_cart, axis=3
    )
    spin_down_electron_muon_seperations = np.linalg.norm(
        spin_down_electron_muon_seperations_cart, axis=3
    )

    max_range = 20
    bins = 4000

    spin_up_bin, bin_edges = np.histogram(
        spin_up_electron_muon_seperations, bins=bins, range=(0, max_range)
    )
    spin_down_bin, _ = np.histogram(
        spin_down_electron_muon_seperations, bins=bins, range=(0, max_range)
    )

    return spin_up_bin, spin_down_bin


if __name__ == "__main__":
    position_data = np.load(f"{WORK_DIR}/ethyl_inference_11630830/inference_out.npy")[
        :, :, :, :
    ]

    observable_position_data = np.load(
        f"{WORK_DIR}/ethyl_inference_11630830_ethyl_muon_spin_correlation.npy"
    ).astype(int)
    print("Position data shape", position_data.shape)

    observable_spin_up_bin, observable_spin_down_bin = observable(position_data)
    new_standard_spin_up_bin, new_standard_spin_down_bin = new_standard(position_data)

    print("UP:")
    print(
        "Observable",
        "\n",
        observable_spin_up_bin[0:100],
        "\n",
        np.sum(observable_spin_up_bin),
        "\n",
        observable_spin_up_bin.shape,
    )
    print(
        "New Standard",
        "\n",
        new_standard_spin_up_bin[0:100],
        "\n",
        np.sum(new_standard_spin_up_bin),
        "\n",
        new_standard_spin_up_bin.shape,
    )
    print(
        "Pre-Calculated Observable",
        "\n",
        observable_position_data[0][0:100],
        "\n",
        np.sum(observable_position_data),
        "\n",
        observable_position_data.shape,
    )
