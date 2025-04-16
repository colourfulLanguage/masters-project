import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import physical_constants
from scipy.optimize import curve_fit
import os


WORK_DIR = "../../runs"


atom_positions = {
    "C_b": (-1.4074, 0, 0),
    "C_a": (1.4074, 0, 0),
    "H1": (2.1942334, 0.0083198, 1.93454097),
    "H2": (2.17742813, -1.68148488, -0.93644871),
    "H3": (2.17774293, 1.67338158, -0.95053236),
    "H4": (-2.46163842, 1.75279276, 0.15536245),
    "H5": (-2.46181114, -1.75257843, 0.15536245),
}

atom_colors = {
    "C_b": "red",
    "C_a": "red",
    "H1": "orange",
    "H2": "blue",
    "H3": "blue",
    "H4": "blue",
    "H5": "blue",
}


# Function to add atom markers
def add_atom_markers(ax, projection, atoms=atom_positions):
    for label, pos in atoms.items():
        if projection == "xy":
            x, y = pos[0], pos[1]
            ax.plot(x, y, "o", label=label, markersize=8, color=atom_colors[label])
        elif projection == "xz":
            x, z = pos[0], pos[2]
            ax.plot(x, z, "o", label=label, markersize=8, color=atom_colors[label])
        elif projection == "yz":
            y, z = pos[1], pos[2]
            ax.plot(y, z, "o", label=label, markersize=8, color=atom_colors[label])
    # ax.legend(loc='upper right')


def plot_muon_density_projections(muon_hist, proton_hist, max_range):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Calculate projections
    xy_projection_muon = np.sum(muon_hist, axis=2).T  # Project along Z
    xz_projection_muon = np.sum(muon_hist, axis=1).T  # Project along Y
    yz_projection_muon = np.sum(muon_hist, axis=0).T  # Project along X

    xy_projection_proton = np.sum(proton_hist, axis=2).T  # Project along Z
    xz_projection_proton = np.sum(proton_hist, axis=1).T  # Project along Y
    yz_projection_proton = np.sum(proton_hist, axis=0).T  # Project along X

    # Plot XY projection
    im_xy_muon = axs[0].imshow(
        xy_projection_muon,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap="plasma",
        alpha=0.5,
    )
    im_xy_proton = axs[0].imshow(
        xy_projection_proton,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap="viridis",
        alpha=0.5,
    )
    axs[0].set_title("Muon and Proton XY Projection")
    axs[0].set_xlabel("X (bohr)")
    axs[0].set_ylabel("Y (bohr)")
    add_atom_markers(axs[0], "xy")

    # Plot XZ projection
    im_xz_muon = axs[1].imshow(
        xz_projection_muon,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap="plasma",
        alpha=0.5,
    )
    im_xz_proton = axs[1].imshow(
        xz_projection_proton,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap="viridis",
        alpha=0.5,
    )
    axs[1].set_title("Muon and Proton XZ Projection")
    axs[1].set_xlabel("X (bohr)")
    axs[1].set_ylabel("Z (bohr)")
    add_atom_markers(axs[1], "xz")

    # Plot YZ projection
    im_yz_muon = axs[2].imshow(
        yz_projection_muon,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap="plasma",
        alpha=0.5,
    )
    im_yz_proton = axs[2].imshow(
        yz_projection_proton,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap="viridis",
        alpha=0.5,
    )
    axs[2].set_title("Muon and Proton YZ Projection")
    axs[2].set_xlabel("Y (bohr)")
    axs[2].set_ylabel("Z (bohr)")
    add_atom_markers(axs[2], "yz")

    plt.tight_layout()
    plt.savefig("plots/quantum_ethyl_muon_proton.png")
    plt.close()


# Function to plot total electron density projections
def plot_electron_density_projections(electron_density_hist, max_range):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Calculate projections
    xy_projection = np.sum(electron_density_hist, axis=2).T  # Project along Z
    xz_projection = np.sum(electron_density_hist, axis=1).T  # Project along Y
    yz_projection = np.sum(electron_density_hist, axis=0).T  # Project along X

    # Plot XY projection
    im_xy = axs[0].imshow(
        xy_projection,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap="viridis",
    )
    axs[0].set_title("XY Projection")
    axs[0].set_xlabel("X (bohr)")
    axs[0].set_ylabel("Y (bohr)")
    add_atom_markers(axs[0], "xy")
    cbar_xy = plt.colorbar(im_xy, ax=axs[0])
    cbar_xy.set_label("Electron Density")

    # Plot XZ projection
    im_xz = axs[1].imshow(
        xz_projection,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap="viridis",
    )
    axs[1].set_title("XZ Projection")
    axs[1].set_xlabel("X (bohr)")
    axs[1].set_ylabel("Z (bohr)")
    add_atom_markers(axs[1], "xz")
    cbar_xz = plt.colorbar(im_xz, ax=axs[1])
    cbar_xz.set_label("Electron Density")

    # Plot YZ projection
    im_yz = axs[2].imshow(
        yz_projection,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap="viridis",
    )
    axs[2].set_title("YZ Projection")
    axs[2].set_xlabel("Y (bohr)")
    axs[2].set_ylabel("Z (bohr)")
    add_atom_markers(axs[2], "yz")
    cbar_yz = plt.colorbar(im_yz, ax=axs[2])
    cbar_yz.set_label("Electron Density")

    plt.tight_layout()
    plt.savefig("plots/quantum_ethyl_muon_electron.png")
    plt.close()


# Function to plot spin density projections
def plot_spin_density_projections(spin_density_hist, max_range):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Calculate projections
    xy_projection = np.sum(spin_density_hist, axis=2).T  # Project along Z
    xz_projection = np.sum(spin_density_hist, axis=1).T  # Project along Y
    yz_projection = np.sum(spin_density_hist, axis=0).T  # Project along X

    # Plot XY projection
    im_xy = axs[0].imshow(
        xy_projection,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap="coolwarm",
    )
    axs[0].set_title("XY Projection")
    axs[0].set_xlabel("X (bohr)")
    axs[0].set_ylabel("Y (bohr)")
    add_atom_markers(axs[0], "xy")
    cbar_xy = plt.colorbar(im_xy, ax=axs[0])
    cbar_xy.set_label("Spin Density")

    # Plot XZ projection
    im_xz = axs[1].imshow(
        xz_projection,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap="coolwarm",
    )
    axs[1].set_title("XZ Projection")
    axs[1].set_xlabel("X (bohr)")
    axs[1].set_ylabel("Z (bohr)")
    add_atom_markers(axs[1], "xz")
    cbar_xz = plt.colorbar(im_xz, ax=axs[1])
    cbar_xz.set_label("Spin Density")

    # Plot YZ projection
    im_yz = axs[2].imshow(
        yz_projection,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap="coolwarm",
    )
    axs[2].set_title("YZ Projection")
    axs[2].set_xlabel("Y (bohr)")
    axs[2].set_ylabel("Z (bohr)")
    add_atom_markers(axs[2], "yz")
    cbar_yz = plt.colorbar(im_yz, ax=axs[2])
    cbar_yz.set_label("Spin Density")

    plt.tight_layout()
    plt.savefig("plots/quantum_ethyl_muon_spin.png")
    plt.close()


def make_3d_plots(
    spin_up_electron_samples, spin_down_electron_samples, proton_samples, muon_samples
):

    UP = 9
    DOWN = 8
    PROTONS = 4
    MUONS = 1
    TOTAL = UP + DOWN + PROTONS + MUONS

    spin_density_bins = 100
    spin_density_max_range = 9

    # reshape into 3D arrays
    spin_up_electron_samples = spin_up_electron_samples.reshape(-1, 3)
    spin_down_electron_samples = spin_down_electron_samples.reshape(-1, 3)
    proton_samples = proton_samples.reshape(-1, 3)
    muon_samples = muon_samples.reshape(-1, 3)

    # Create 3D histogram
    spin_up_hist, _ = np.histogramdd(
        sample=spin_up_electron_samples,
        bins=spin_density_bins,
        range=[[-spin_density_max_range, spin_density_max_range]] * 3,
    )
    spin_down_hist, _ = np.histogramdd(
        sample=spin_down_electron_samples,
        bins=spin_density_bins,
        range=[[-spin_density_max_range, spin_density_max_range]] * 3,
    )
    proton_hist, _ = np.histogramdd(
        sample=proton_samples,
        bins=spin_density_bins,
        range=[[-spin_density_max_range, spin_density_max_range]] * 3,
    )
    muon_hist, _ = np.histogramdd(
        sample=muon_samples,
        bins=spin_density_bins,
        range=[[-spin_density_max_range, spin_density_max_range]] * 3,
    )

    electron_density_hist = spin_up_hist + spin_down_hist
    spin_density_hist = spin_up_hist - spin_down_hist

    plot_spin_density_projections(spin_density_hist, spin_density_max_range)
    plot_electron_density_projections(electron_density_hist, spin_density_max_range)
    plot_muon_density_projections(muon_hist, proton_hist, spin_density_max_range)


def run():

    position_data = np.load(
        f"{WORK_DIR}/ethyl_quantum_muon_inference_12555706/inference_out.npy"
    )[:25, :, :, :]

    print("Position data shape", position_data.shape)

    total_points = (
        position_data.shape[0] * position_data.shape[1] * position_data.shape[2]
    )

    print("Total points", total_points)

    UP = 9
    DOWN = 8
    PROTONS = 4
    MUONS = 1
    TOTAL = UP + DOWN + PROTONS + MUONS

    position_data = position_data.reshape(-1, 4096, 3 * TOTAL)
    slice_num = position_data.shape[0]

    spin_up_electron_samples = position_data[:, :, : UP * 3].reshape(
        slice_num, 4096, -1, 3
    )
    spin_down_electron_samples = position_data[:, :, UP * 3 : (UP + DOWN) * 3].reshape(
        slice_num, 4096, -1, 3
    )
    proton_samples = position_data[
        :, :, (UP + DOWN) * 3 : (UP + DOWN + PROTONS) * 3
    ].reshape(slice_num, 4096, -1, 3)
    muon_samples = position_data[:, :, (UP + DOWN + PROTONS) * 3 :].reshape(
        slice_num, 4096, -1, 3
    )

    print("Mean positions and std deviations")
    print("Spin up electrons", np.mean(spin_up_electron_samples, axis=(0, 1)))
    print("Spin down electrons", np.mean(spin_down_electron_samples, axis=(0, 1)))
    print("Protons", np.mean(proton_samples, axis=(0, 1)))
    print("Muons", np.mean(muon_samples, axis=(0, 1)))

    make_3d_plots(
        spin_up_electron_samples,
        spin_down_electron_samples,
        proton_samples,
        muon_samples,
    )

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

    max_range = 20 // 20
    bins = 4000 // 20

    spin_up_bin, bin_edges = np.histogram(
        spin_up_electron_muon_seperations, bins=bins, range=(0, max_range)
    )
    spin_down_bin, _ = np.histogram(
        spin_down_electron_muon_seperations, bins=bins, range=(0, max_range)
    )

    spin_diff_bin = spin_up_bin - spin_down_bin

    bin_normalisation = (4 / 3) * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)

    bin_heights_normalised = spin_diff_bin / (bin_normalisation * (total_points))

    print("start of bin heights normalised", bin_heights_normalised[0:10])

    def exp_func(r, a, b):
        return a * np.exp(-b * r)

    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    y_data = bin_heights_normalised

    FIT_CUTOFF = 30

    bin_centers_fit = bin_centers[FIT_CUTOFF:]
    y_data_fit = y_data[FIT_CUTOFF:]

    # Fit the data with the 5th-order polynomial function
    popt, pcov = curve_fit(exp_func, bin_centers_fit, y_data_fit)

    # Generate fine points for plotting the fitted polynomial
    r_fine = np.linspace(0, max_range, 1000)
    y_fitted = exp_func(r_fine, *popt)

    # Plot the original data and the fitted polynomial
    plt.plot(bin_centers, y_data, "o", label="Data")
    plt.plot(r_fine, y_fitted, "-", label="5th-Order Polynomial Fit")
    plt.xlabel("Separation (bohr)")
    plt.ylabel("Normalized Counts")
    plt.title("Muon-Electron Separation with 5th-Order Polynomial Fit")
    plt.vlines(
        bin_centers[FIT_CUTOFF],
        0,
        1,
        color="red",
        linestyle="--",
    )
    plt.ylim(-0.1, 0.1)
    plt.legend()
    plt.savefig(f"plots/quantum_ethyl_muon_separation_fit")
    plt.close()

    # Interpolate at r=0
    y_at_r0 = exp_func(0, *popt)

    print(f"Interpolated value at r=0: {y_at_r0}")

    # Constants

    y_in_SI = y_at_r0 / ((physical_constants["Bohr radius"][0]) ** 3)

    mu_e = physical_constants["electron mag. mom."][0]  # J·T^{-1} = J A S^2 KG^-1
    mu_mu = physical_constants["muon mag. mom."][0]  # J·T^{-1} = J A S^2 KG^-1
    mu_0 = physical_constants["vacuum mag. permeability"][0]  # N·A^{-2}
    hbar = physical_constants["reduced Planck constant"][0]  # J·s

    g_e = 2.002_319_304_360_92  # Electron g-factor
    g_mu = 2.002_331_841_23  # Muon g-factor

    gamma_e = (g_e * mu_e) / (hbar)  # units of T-1 S-1 = KG-1 S1 A1
    gamma_mu = (g_mu * mu_mu) / (hbar)  # units of  T-1 S-1 = KG-1 S1 A1

    # N = KG M S^-2

    # gamma_e * gamma_mu = A2 S2 KG-2
    # gamma_e * gamma_mu * mu_0 = N S2 KG-2 = KG M

    # gamma_e * gamma_mu * mu_0 * hbar * |psi|^2 = J S KG M-2

    # Fermi contact coupling constant A
    A_constant = (mu_0 * 2 * hbar) / (3)
    A_value = A_constant * gamma_e * gamma_mu * y_in_SI
    A_value_freq = A_value / (2 * np.pi)
    print(f"Fermi contact coupling A GHz : {A_value_freq/1e9}")
    print(gamma_e, gamma_mu, A_constant)

    # dispose of the figure to avoid memory leaks
    plt.close()


if __name__ == "__main__":
    pass
    run()
