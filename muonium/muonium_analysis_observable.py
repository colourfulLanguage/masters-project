import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import physical_constants
from scipy.optimize import curve_fit

WORK_DIR = "/leonardo_work/FUAL8_NWPAV/jamie/runs/"

def run():
    # Load muonium data - assuming data structure is [spin_up, spin_down]
    seperation_hists = np.load(f"{WORK_DIR}/muonium_inference_14231933/any_muon_spin_correlation.npy")
    # convert to integer
    seperation_hists = seperation_hists.astype(int)
    
    spin_up_bin = seperation_hists[0]
     
    print("Start of spin up bin", spin_up_bin[0:10])
    
    iterations = np.sum(spin_up_bin)/(4096)
    samples = iterations * 4096
    
    bins = 4000
    max_range = 20
    
    # Create histogram bins
    _, bin_edges = np.histogram(
        spin_up_bin, bins=bins, range=(0, max_range)
    )
    
    # Calculate spin difference
    spin_diff_bin = spin_up_bin
    
    # Calculate bin normalization (spherical shell volume)
    bin_normalisation = (4 / 3) * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)
    
    # Normalize the spin difference
    bin_heights_normalised = spin_diff_bin / (bin_normalisation * samples)
    
    print("Start of bin heights normalised", bin_heights_normalised[0:10])
    
    # Set cutoff for analysis
    new_max_range = 0.5  # Consistent with previous cutoff
    idx = np.argmax(bin_edges > new_max_range)
    new_bin_edges = bin_edges[:idx+1]
    new_bin_heights_normalised = bin_heights_normalised[:idx]
    
    # Calculate errors based on Poisson statistics
    spin_diff_bin_errors = np.sqrt(spin_up_bin)
    bin_heights_errors = spin_diff_bin_errors / (bin_normalisation * samples)
    new_bin_heights_errors = bin_heights_errors[:idx]
    
    # Basic plots
    plt.figure()
    plt.plot(
        (bin_edges[1:] + bin_edges[:-1]) / 2, bin_heights_normalised, marker="o", ls=""
    )
    plt.xlabel("Separation (bohr)")
    plt.ylabel("Normalised Counts")
    plt.title("Muon-Electron Separation Normalised Histogram")
    plt.savefig("plots/muonium_separation_normalised_observable.png")
    
    # Prepare data for fitting
    bin_centers = (new_bin_edges[1:] + new_bin_edges[:-1]) / 2
    y_data = new_bin_heights_normalised
    
    # Define fitting starting point - can be adjusted
    FIT_CUTOFF = 0
    
    # Filter data for fitting
    bin_centers_fit = bin_centers[FIT_CUTOFF:]
    y_data_fit = y_data[FIT_CUTOFF:]
    y_data_fit_errors = new_bin_heights_errors[FIT_CUTOFF:]
    
    # Define fitting function (exponential in log space)
    def fit_func(r, a,  c):
        b = -0.995187 * 2 * 1
        return a + b*r + c*r**2
   
    # Convert to log space for fitting
    # Assuming our data is mostly positive for muonium case
    # If not, we may need to adjust this approach
    log_y_data_fit = np.log(np.abs(y_data_fit))
    # Error propagation from linear to log space
    log_y_data_fit_errs = y_data_fit_errors / np.abs(y_data_fit)
    print(log_y_data_fit_errs)

    # Handle any NaN or inf values
    mask = np.isfinite(log_y_data_fit) & np.isfinite(log_y_data_fit_errs)
    bin_centers_fit = bin_centers_fit[mask]
    log_y_data_fit = log_y_data_fit[mask]
    log_y_data_fit_errs = log_y_data_fit_errs[mask]
    
    # Fit the data
    popt, pcov = curve_fit(fit_func, bin_centers_fit, log_y_data_fit, 
                          )#absolute_sigma=True, sigma=log_y_data_fit_errs)
    errs = np.sqrt(np.diag(pcov))
    
    # Generate points for plotting
    r_fine = np.linspace(0, new_max_range, 1000)
    y_fitted = np.exp(fit_func(r_fine, *popt))
    
    # Plot the data and fit
    plt.figure(figsize=(10, 6))
    plt.errorbar(bin_centers, y_data, yerr=new_bin_heights_errors, fmt='o', 
                label="Data with Poisson errors", alpha=0.6)
    plt.plot(r_fine, y_fitted, "r-", linewidth=2, 
            label=f"Exponential fit (log space)", zorder=10)
    
    # Add vertical line at cutoff
    plt.axvline(x=new_max_range, color='red', linestyle='--', alpha=0.5, 
               label=f'Cutoff: {new_max_range} bohr')
    
    plt.xlabel("Separation (bohr)")
    plt.ylabel("Normalized Spin Correlation")
    plt.title("Muon-Electron Separation with Exponential Fit")
    plt.legend()
    plt.savefig("plots/muonium_separation_fit_observable.png")
    
    # Interpolate at r=0
    y_at_r0 = np.exp(fit_func(0, *popt))
    y_at_r0_param = np.exp(popt[0])  # This is just A in A*exp(-B*r)
    y_at_r0_err = y_at_r0_param * errs[0]  # Error propagation
    y_at_r0_err_percent = (y_at_r0_err / y_at_r0_param) * 100
    
    print(f"Interpolated value at r=0: {y_at_r0} ({y_at_r0_err} error, {y_at_r0_err_percent:.2f}%)")
    
    # Calculate physical quantities
    y_in_SI = y_at_r0 / ((physical_constants["Bohr radius"][0]) ** 3)
    
    mu_e = physical_constants["electron mag. mom."][0]  # J·T^{-1}
    mu_mu = physical_constants["muon mag. mom."][0]  # J·T^{-1}
    mu_0 = physical_constants["vacuum mag. permeability"][0]  # N·A^{-2}
    hbar = physical_constants["reduced Planck constant"][0]  # J·s
    
    g_e = 2.002_319_304_360_92  # Electron g-factor
    g_mu = -2.002_331_841_23  # Muon g-factor
    
    gamma_e = (g_e * mu_e) / (hbar)  # units of T-1 S-1
    gamma_mu = (g_mu * mu_mu) / (hbar)  # units of T-1 S-1
    
    # Fermi contact coupling constant A
    A_constant = (mu_0 * 2 * hbar) / (3)
    A_value = A_constant * gamma_e * gamma_mu * y_in_SI
    A_value_freq = A_value / (2 * np.pi)
    
    print(f"Fermi contact coupling A Hz: {A_value_freq}")
    print(f"Fermi contact coupling A GHz: {A_value_freq/1e9}")
    
    # Calculate volume integrated spin difference
    volume_integrated_spin_diff = np.sum(bin_heights_normalised * bin_normalisation[:len(bin_heights_normalised)])
    print("Volume integrated spin difference:", volume_integrated_spin_diff)

if __name__ == "__main__":
    run()