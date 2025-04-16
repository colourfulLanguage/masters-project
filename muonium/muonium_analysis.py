import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import physical_constants
from scipy.optimize import curve_fit

WORK_DIR = "../../runs"


if __name__ == "__main__":

    # Load the position data
    position_data = np.load(
        f"{WORK_DIR}/muonium_inference_observable_10867089/inference_out.npy"
    )[:, :, :, :]
    # position_data = np.load(f"{WORK_DIR}/muonium_andres_inference_10891672/inference_out.npy")

    total_points = (
        position_data.shape[0] * position_data.shape[1] * position_data.shape[2]
    )

    slice_num = position_data.shape[0]
    electron_samples = position_data[:, :, :, :3].reshape(slice_num, 4096, -1, 3)
    muon_samples = position_data[:, :, :, 3:].reshape(slice_num, 4096, -1, 3)

    muon_electron_seperations_cart = electron_samples - muon_samples
    muon_electron_seperations = np.linalg.norm(muon_electron_seperations_cart, axis=3)

    n_bins = 4000 // 20
    max_radius = 20 // 20

    bin_heights, bin_edges = np.histogram(
        muon_electron_seperations, bins=n_bins, range=(0, max_radius)
    )
    bin_normalisation = (4 / 3) * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)
    bin_heights_normalised = bin_heights / (bin_normalisation * total_points)

    electron_x = electron_samples[:, :, :, 0].flatten()
    electron_y = electron_samples[:, :, :, 1].flatten()

    # Extract muon positions (second particle)
    muon_x = muon_samples[:, :, :, 0].flatten()
    muon_y = muon_samples[:, :, :, 1].flatten()

    plt.figure()
    plt.plot(
        (bin_edges[1:] + bin_edges[:-1]) / 2, bin_heights_normalised, marker="o", ls=""
    )
    plt.xlabel("Separation (bohr)")
    plt.ylabel("Normalised Counts")
    plt.title("Muon-Electron Separation Normalised Histogram")
    plt.savefig("plots/muonium_separation_normalised.png")

    plt.figure()
    plt.plot((bin_edges[1:] + bin_edges[:-1]) / 2, bin_heights, marker="o", ls="")
    plt.xlabel("Separation (bohr)")
    plt.ylabel("Normalised Counts")
    plt.title("Muon-Electron Separation Normalised Histogram")
    plt.savefig("plots/muonium_seperation.png")
    # fit an exponential to the bin_heights_normalised to interpolate at r=0

    # Define the exponential function to fit
    def exp_func(r, A, B):
        B = 2 * 0.9951857
        return A * np.exp(-B * r)

    # Bin centers
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    y_data = bin_heights_normalised

    # Perform the curve fitting
    popt, pcov = curve_fit(exp_func, bin_centers, bin_heights_normalised)

    # Extract fitting parameters
    A_fit, B_fit = popt

    # Generate fitted data for plotting
    r_fine = np.linspace(0, max_radius, 1000)
    print("Optimised", popt)
    y_fitted = exp_func(r_fine, *popt)

    # Plot the original data and the fitted function
    plt.figure()
    plt.plot(bin_centers, y_data, "o", label="Data")
    plt.plot(r_fine, y_fitted, "-", label="Fitted Exponential")
    plt.xlabel("Separation (bohr)")
    plt.ylabel("Normalized Counts")
    plt.title("Muon-Electron Separation with Exponential Fit")
    plt.legend()
    plt.savefig("plots/muonium_separation_fit.png")

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
    g_mu = -2.002_331_841_23  # Muon g-factor

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
    print(f"Fermi contact coupling A Hz : {A_value_freq}")
    print(gamma_e, gamma_mu, A_constant)
