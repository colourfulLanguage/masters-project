import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import physical_constants
from scipy.optimize import curve_fit
import os

WORK_DIR = "../../runs"


def run():

    seperation_hists = np.load(
        f"{WORK_DIR}/ethyl_electric_inference_11974154_12823456_100_000/any_muon_spin_correlation.npy"
    )

    seperation_hists = seperation_hists.astype(int)

    spin_up_bin = seperation_hists[0]
    spin_down_bin = seperation_hists[1]

    print("Start of spin up bin", spin_up_bin[0:10])

    bin_edges = np.linspace(0, 1, 4001 // 20)

    iterations = (np.sum(seperation_hists)) / (4096 * 17)
    samples = iterations * 4096

    print(seperation_hists.shape)
    print(np.sum(seperation_hists))
    print(iterations)

    bins = seperation_hists.shape[1]
    max_range = 20

    _, bin_edges = np.histogram(seperation_hists[0], bins=bins, range=(0, max_range))

    spin_diff_bin = spin_up_bin - spin_down_bin

    bin_normalisation = (4 / 3) * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)

    bin_heights_normalised = spin_diff_bin / (bin_normalisation * (samples))

    print("Start of bin heights normalised", bin_heights_normalised[0:10])

    new_max_range = 0.5
    idx = np.argmax(bin_edges > new_max_range)
    new_bin_edges = bin_edges[: idx + 1]
    new_bin_heights_normalised = bin_heights_normalised[:idx]

    spin_diff_bin_errors = np.sqrt(spin_up_bin + spin_down_bin)
    bin_heights_errors = spin_diff_bin_errors / (bin_normalisation * (samples))
    new_bin_heights_errors = bin_heights_errors[:idx]

    bin_centers = (new_bin_edges[1:] + new_bin_edges[:-1]) / 2
    y_data = new_bin_heights_normalised

    FIT_CUTOFF = 0

    # Exclude the 10 smallest values from the fit
    bin_centers_fit = bin_centers[FIT_CUTOFF:]
    y_data_fit = new_bin_heights_normalised[FIT_CUTOFF:]
    y_data_fit_errors = new_bin_heights_errors[FIT_CUTOFF:]

    def fit_func(r, a, c):
        # 5th order poly. first linear term is fixed by cusp conditio
        b = -0.995187 * 2 * 1
        return a + b * r + c * r**2  # + d*r**3 + e*r**4 + f*r**5

    # kde scipy?

    log_y_data_fit = np.nan_to_num(np.log(y_data_fit), nan=0.0, posinf=0, neginf=0)
    log_y_data_fit_errs = np.nan_to_num(
        np.abs(y_data_fit_errors / y_data_fit),
        nan=100000,
        posinf=100000,
        neginf=-100000,
    )

    print(log_y_data_fit)
    print(log_y_data_fit_errs)

    # Fit the data with the 5th-order polynomial function
    popt, pcov = curve_fit(
        fit_func,
        bin_centers_fit,
        log_y_data_fit,
        sigma=log_y_data_fit_errs,
        absolute_sigma=True,
    )

    errs = np.sqrt(np.diag(pcov))
    print("parameters and errors")
    print(popt)
    print(errs)

    # Generate fine points for plotting the fitted polynomial
    r_fine = np.linspace(0, new_max_range, 1000)
    y_fitted = fit_func(r_fine, *popt)

    # Plot the log plot.
    plt.figure()
    plt.errorbar(
        bin_centers_fit,
        log_y_data_fit,
        yerr=log_y_data_fit_errs,
        fmt="o",
        label="Data with Poisson Errors",
    )
    plt.plot(r_fine, y_fitted, "-", label="5th-Order Polynomial Fit", zorder=10)
    plt.xlabel("Separation (bohr)")
    plt.ylabel("log(g(r))")
    plt.vlines(
        bin_centers[FIT_CUTOFF],
        0,
        1,
        colors="r",
        linestyles="dashed",
        label="Fit Cutoff",
    )
    plt.ylim(-5, 0)
    plt.savefig(f"plots/electric_ethyl_separation_fit_observable_log.png")

    # Plot the original data and the fitted polynomial
    plt.figure()
    plt.errorbar(
        bin_centers,
        y_data,
        yerr=new_bin_heights_errors,
        fmt="o",
        label="Data with Poisson Errors",
    )
    plt.plot(r_fine, np.exp(y_fitted), "-", label="5th-Order Polynomial Fit", zorder=10)
    plt.xlabel("Separation (bohr)")
    plt.ylabel("g(r)")
    plt.title("Ethyl unpaired electron-muon correlation function g(r)")
    plt.vlines(
        bin_centers[FIT_CUTOFF],
        0,
        1,
        colors="r",
        linestyles="dashed",
        label="Fit Cutoff",
    )
    plt.ylim(0.01, 0.05)
    # plt.legend()
    plt.savefig(f"plots/electric_ethyl_separation_fit_observable.png")

    # Interpolate at r=0
    y_at_r0 = np.exp(fit_func(0, *popt))
    y_at_r0_param = np.exp(popt[0])
    y_at_r0_err = y_at_r0_param * errs[0]
    y_at_r0_err_percent = (y_at_r0_err / y_at_r0_param) * 100
    print(
        f"Interpolated value at r=0: {y_at_r0} {y_at_r0_param} err: {y_at_r0_err} {y_at_r0_err_percent}%"
    )

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
    # dispose of the figure to avoid memory leaks
    # plt.close()
    volume_integrated_spin_diff = np.sum(
        bin_heights_normalised * bin_normalisation[: len(bin_heights_normalised)]
    )
    print("Volume integrated spin difference", volume_integrated_spin_diff)


if __name__ == "__main__":
    run()
