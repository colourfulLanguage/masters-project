_PATHS = {
    "methyl": {
        "methyl_electric": {
            "path": "methyl_electric_inference_12438954_14053051_120_000/any_muon_spin_correlation.npy",
            "label": "Classical Muon",
        },
        "methyl_quantum": {
            "path": "methyl_quantum_muon_inference_12632287_13893337_100_000/any_muon_spin_correlation.npy",
            "label": "Full Quantum",
        },
        "methyl": {
            "path": "methyl_inference_10864758_14052890_100_000/any_muon_spin_correlation.npy",
            "label": "Quantum Muon",
        },
        # "methyl_pertubed": {
        #    "path": "methyl_extended_inference_12995641_14076179_100_000/any_muon_spin_correlation.npy",
        #    "label": "Quantum Perturbed",
        # },
        # "methyl_electric_pertubed": {
        #    "path": "methyl_electric_extended_inference_12995673_14076152_100_000/any_muon_spin_correlation.npy",
        #    "label": "Classical Perturbed",
        # },
    },
    "ethyl": {
        "ethyl_electric": {
            "path": "ethyl_electric_inference_11974154_14052766_30_000/any_muon_spin_correlation.npy",
            "label": "Classical Muon",
        },
        "ethyl": {
            "path": "ethyl_inference_11783960_14052450_30_000/any_muon_spin_correlation.npy",
            "label": "Quantum Muon",
        },
    },
    "muonium": {
        "muonium": {
            "path": "muonium_inference_14231933/any_muon_spin_correlation.npy",
            "label": "Quantum Muon",
        },
    },
}

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.constants import physical_constants
from scipy.optimize import curve_fit
import os

# Set a professional publication style for physics research with larger fonts
plt.style.use("seaborn-v0_8-paper")  # Professional paper style
mpl.rcParams.update(
    {
        "font.family": "serif",  # Serif fonts are standard in scientific publications
        "font.serif": [
            "DejaVu Serif",
        ],
        "font.size": 14,  # Increased base font size
        "axes.linewidth": 1.0,  # Slightly thicker axes lines
        "axes.labelsize": 16,  # Larger axis labels
        "axes.titlesize": 18,  # Larger title
        "xtick.labelsize": 14,  # Larger tick labels
        "ytick.labelsize": 14,
        "xtick.direction": "in",  # Ticks point inward
        "ytick.direction": "in",  # Ticks point inward
        "xtick.major.size": 5,  # Longer ticks
        "ytick.major.size": 5,
        "xtick.minor.size": 3,  # Add minor ticks
        "ytick.minor.size": 3,
        "figure.figsize": (10, 7),  # Slightly larger figure
        "figure.dpi": 120,  # Higher DPI for clear figures
        "savefig.dpi": 300,  # High DPI for saving
        "savefig.bbox": "tight",  # Tight bounding box
        "mathtext.fontset": "cm",  # Computer Modern math font
        "legend.frameon": False,  # No frame for legends
        "legend.fontsize": 14,  # Increased legend font size
        "grid.linestyle": ":",  # Dotted grid lines
    }
)

WORK_DIR = "../../runs"
FIT_CUTOFF = 0
MAX_RANGE = 20
NEW_MAX_RANGE = 0.5


def fit_func(r, a, c):
    # 5th order poly. first linear term is fixed by cusp condition
    b = -0.995187 * 2 * 1
    return a + b * r + c * r**2


def process_correlation_data(category, file_path, label):
    # Load the spin correlation data
    seperation_hists = np.load(f"{WORK_DIR}/{file_path}")
    seperation_hists = seperation_hists.astype(int)

    # Extract spin up and down bins
    spin_up_bin = seperation_hists[0]

    # Muonium case only uses spin up data (single electron always spin up)
    if category == "muonium":
        spin_down_bin = np.zeros_like(spin_up_bin)  # Create dummy zeros array
        iterations = np.sum(spin_up_bin) / 4096
        particles = 1  # Only 1 electron in muonium
    else:
        spin_down_bin = seperation_hists[1]
        particles = 9 if category == "methyl" else 17
        iterations = (np.sum(seperation_hists)) / (4096 * particles)

    samples = iterations * 4096

    # Calculate spin difference
    spin_diff_bin = (
        spin_up_bin if category == "muonium" else spin_up_bin - spin_down_bin
    )

    # Calculate bin edges, iterations, samples
    bins = seperation_hists.shape[1]
    _, bin_edges = np.histogram(seperation_hists[0], bins=bins, range=(0, MAX_RANGE))

    # Normalize bins
    bin_normalisation = (4 / 3) * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)
    bin_heights_normalised = spin_diff_bin / (bin_normalisation * samples)

    # Limit range for visualization
    idx = np.argmax(bin_edges > NEW_MAX_RANGE)
    new_bin_edges = bin_edges[: idx + 1]
    new_bin_heights_normalised = bin_heights_normalised[:idx]

    # Calculate errors
    if category == "muonium":
        spin_diff_bin_errors = np.sqrt(spin_up_bin)
    else:
        spin_diff_bin_errors = np.sqrt(spin_up_bin + spin_down_bin)

    bin_heights_errors = spin_diff_bin_errors / (bin_normalisation * samples)
    new_bin_heights_errors = bin_heights_errors[:idx]

    # Calculate bin centers for plotting
    bin_centers = (new_bin_edges[1:] + new_bin_edges[:-1]) / 2

    # Prepare data for fitting
    bin_centers_fit = bin_centers[FIT_CUTOFF:]
    y_data_fit = new_bin_heights_normalised[FIT_CUTOFF:]
    y_data_fit_errors = new_bin_heights_errors[FIT_CUTOFF:]

    # Log transform for fitting
    if category == "ethyl":
        log_y_data_fit = np.nan_to_num(np.log(y_data_fit), nan=0.0, posinf=0, neginf=0)
    elif category == "muonium":
        log_y_data_fit = np.nan_to_num(
            np.log(np.abs(y_data_fit)), nan=0.0, posinf=0, neginf=0
        )
    else:
        log_y_data_fit = np.nan_to_num(np.log(-y_data_fit), nan=0.0, posinf=0, neginf=0)

    log_y_data_fit_errs = np.nan_to_num(
        y_data_fit_errors / np.abs(y_data_fit),
        nan=100000,
        posinf=100000,
        neginf=-100000,
    )

    # Fit the function
    try:
        if category == "muonium":
            # Turn off variance matching for muonium
            popt, pcov = curve_fit(
                fit_func,
                bin_centers_fit,
                log_y_data_fit,
                absolute_sigma=False,
            )
        else:
            popt, pcov = curve_fit(
                fit_func,
                bin_centers_fit,
                log_y_data_fit,
                sigma=log_y_data_fit_errs,
                absolute_sigma=True,
            )
        errs = np.sqrt(np.diag(pcov))
        fit_success = True
    except:
        fit_success = False
        popt = None
        errs = None

    # Calculate volume integrated spin difference
    volume_integrated_spin_diff = np.sum(
        bin_heights_normalised * bin_normalisation[: len(bin_heights_normalised)]
    )

    return {
        "bin_centers": bin_centers,
        "bin_heights": new_bin_heights_normalised,
        "bin_errors": new_bin_heights_errors,
        "fit_params": popt,
        "fit_errors": errs,
        "fit_success": fit_success,
        "volume_integrated": volume_integrated_spin_diff,
        "label": label,
    }


def calculate_physical_constants(y_at_r0):
    # Convert to SI units
    y_in_SI = y_at_r0 / ((physical_constants["Bohr radius"][0]) ** 3)

    # Physical constants
    mu_e = physical_constants["electron mag. mom."][0]
    mu_mu = physical_constants["muon mag. mom."][0]
    mu_0 = physical_constants["vacuum mag. permeability"][0]
    hbar = physical_constants["reduced Planck constant"][0]

    g_e = 2.002_319_304_360_92  # Electron g-factor
    g_mu = 2.002_331_841_23  # Muon g-factor

    gamma_e = (g_e * mu_e) / (hbar)
    gamma_mu = (g_mu * mu_mu) / (hbar)

    # Fermi contact coupling constant A
    A_constant = (mu_0 * 2 * hbar) / (3)
    A_value = A_constant * gamma_e * gamma_mu * y_in_SI
    A_value_freq = A_value / (2 * np.pi)

    return A_value_freq / 1e6  # Return in MHz


def get_conversion_factor():
    # Calculate the conversion factor from g(r) to MHz (hyperfine constant)
    # Use g(r)=1.0 as reference to get the scaling factor
    return calculate_physical_constants(1.0)


def format_value_with_error(value, error):
    """Format value with error in bracket notation with appropriate significant figures.
    Example: 123.456 ± 0.789 becomes 123.5(8)"""

    if value == 0 or error == 0:
        return f"{value:.3g}"

    # Determine the position of the last significant digit in value (for 3 sig figs)
    exponent = int(np.floor(np.log10(abs(value))))
    last_digit_pos = exponent - 2  # For 3 significant figures

    # Round value to 3 significant figures
    rounded_value = round(value, -last_digit_pos)

    # Scale error to the last digit position and round to 1 significant figure
    scaled_error = error * 10 ** (-last_digit_pos)
    rounded_error = int(np.ceil(scaled_error))  # Round up to be conservative

    # Format the output string
    if last_digit_pos >= 0:
        # For values >= 1
        value_str = f"{rounded_value:.0f}"
        return f"{value_str}({rounded_error})"
    else:
        # For values < 1, we need to handle decimal places
        decimals = -last_digit_pos
        value_format = f"{{:.{decimals}f}}"
        value_str = value_format.format(rounded_value)
        return f"{value_str}({rounded_error})"


def run():
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    # Process all data first
    category_results = {}

    # Process each category (methyl, ethyl, muonium)
    for category, systems in _PATHS.items():
        category_results[category] = []

        for system_name, system_info in systems.items():
            file_path = system_info["path"]
            label = system_info["label"]

            print(f"Processing {label} from {file_path}")

            try:
                result = process_correlation_data(category, file_path, label)
                category_results[category].append(result)

                # Calculate and print physical constants if fit was successful
                if result["fit_success"]:
                    # For muonium use absolute value
                    if category == "muonium":
                        y_at_r0 = np.exp(result["fit_params"][0])
                    elif category == "ethyl":
                        y_at_r0 = np.exp(result["fit_params"][0])
                    else:
                        y_at_r0 = -np.exp(result["fit_params"][0])

                    fermi_coupling = calculate_physical_constants(abs(y_at_r0))
                    print(
                        f"{label} - Fermi contact coupling (MHz): {fermi_coupling:.4f}"
                    )
                    print(
                        f"{label} - Volume integrated spin difference: {result['volume_integrated']:.6f}"
                    )

            except Exception as e:
                print(f"Error processing {label}: {str(e)}")

    # Create a separate plot for each category
    for category, results in category_results.items():
        fig, ax = plt.subplots(figsize=(10, 7))

        # Use color cycle to get consistent colors for each system
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

        # Store handles and labels for custom legend
        legend_handles = []
        legend_labels = []

        for i, result in enumerate(results):
            color = colors[i % len(colors)]  # Use same color for data and fit

            # Calculate hyperfine constant if fit is successful
            hyperfine_value = None
            hyperfine_error = None
            g_at_r0 = None
            g_at_r0_err = None

            if result["fit_success"]:
                # Calculate g(r) at r=0 and its error
                if category == "ethyl":
                    g_at_r0 = np.exp(result["fit_params"][0])
                    g_at_r0_err = g_at_r0 * result["fit_errors"][0]
                elif category == "muonium":
                    g_at_r0 = np.exp(result["fit_params"][0])
                    g_at_r0_err = g_at_r0 * result["fit_errors"][0]
                else:
                    g_at_r0 = -np.exp(result["fit_params"][0])
                    g_at_r0_err = abs(g_at_r0 * result["fit_errors"][0])

                hyperfine_value = calculate_physical_constants(abs(g_at_r0))
                hyperfine_error = abs(hyperfine_value * (g_at_r0_err / g_at_r0))

            # Plot data points (but don't include in legend)
            ax.plot(
                result["bin_centers"],
                result["bin_heights"],
                "o",
                markersize=2,
                alpha=0.7,
                color=color,
            )

            # If fitting successful, plot the fit line with hyperfine constant in label
            if result["fit_success"]:
                r_fine = np.linspace(0, NEW_MAX_RANGE, 1000)
                y_fitted = fit_func(r_fine, *result["fit_params"])

                # Create label with hyperfine constant using bracket notation with subscript
                legend_label = result["label"]
                if hyperfine_value is not None and hyperfine_error is not None:
                    if category == "muonium":
                        # Use 4 significant figures for muonium with error
                        exponent = int(np.floor(np.log10(abs(hyperfine_value))))
                        last_digit_pos = exponent - 3  # For 4 significant figures

                        # Round value to 4 significant figures
                        rounded_value = round(hyperfine_value, -last_digit_pos)

                        # Scale error to the last digit position and round
                        scaled_error = hyperfine_error * 10 ** (-last_digit_pos)
                        rounded_error = int(
                            np.ceil(scaled_error)
                        )  # Round up to be conservative

                        # Format the output string
                        if last_digit_pos >= 0:
                            value_str = f"{rounded_value:.0f}"
                        else:
                            decimals = -last_digit_pos
                            value_format = f"{{:.{decimals}f}}"
                            value_str = value_format.format(rounded_value)

                        formatted_value = f"{value_str}({rounded_error})"
                        legend_label += f" ($A_i$ = {formatted_value} MHz)"
                    else:
                        formatted_value = format_value_with_error(
                            hyperfine_value, hyperfine_error
                        )
                        legend_label += f" ($A_i$ = {formatted_value} MHz)"

                if category == "ethyl":
                    (line,) = ax.plot(
                        r_fine,
                        np.exp(y_fitted),
                        "-",
                        color=color,
                        linewidth=1,
                    )

                    # Add error bar at r=0 for ethyl - no point, just the error bar
                    ax.errorbar(
                        0,
                        g_at_r0,
                        yerr=g_at_r0_err,
                        fmt="none",  # No marker, just the error bar
                        color=color,
                        capsize=1,  # Small horizontal caps
                        elinewidth=1,
                        zorder=10,
                    )
                elif category == "muonium":
                    (line,) = ax.plot(
                        r_fine,
                        np.exp(y_fitted),
                        "-",
                        color=color,
                        linewidth=1,
                    )

                    # Add error bar at r=0 for muonium
                    ax.errorbar(
                        0,
                        g_at_r0,
                        yerr=g_at_r0_err,
                        fmt="none",  # No marker, just the error bar
                        color=color,
                        capsize=1,  # Small horizontal caps
                        elinewidth=1,
                        zorder=10,
                    )
                else:
                    (line,) = ax.plot(
                        r_fine,
                        -np.exp(y_fitted),
                        "-",
                        color=color,
                        linewidth=1,
                    )

                    # Add error bar at r=0 for methyl - no point, just the error bar
                    ax.errorbar(
                        0,
                        g_at_r0,
                        yerr=g_at_r0_err,
                        fmt="none",  # No marker, just the error bar
                        color=color,
                        capsize=1,  # Small horizontal caps
                        elinewidth=1,
                        zorder=10,
                    )

                # Add only the line to the legend
                legend_handles.append(line)
                legend_labels.append(legend_label)

        # Set primary axis labels and limits
        ax.set_xlabel("Electron-muon separation r (bohr)")
        ax.set_ylabel("g(r)")
        if category == "muonium":
            ax.set_title(f"{category.capitalize()} Muon Spin Correlation Function")
        else:
            ax.set_title(f"{category.capitalize()} Muon Spin Correlation Functions")

        # Set specific y-axis limits for each category
        if category == "methyl":
            ax.set_ylim(-0.0175, -0.0025)
        elif category == "ethyl":
            ax.set_ylim(0.01, 0.04)
        elif category == "muonium":
            # Set fixed y-limits for muonium
            ax.set_ylim(0.10, 0.35)
        else:
            ax.set_ylim(-0.03, 0.03)

        ax.grid(alpha=0.3)

        # Add custom legend with only the fit lines
        if legend_handles:
            ax.legend(legend_handles, legend_labels, loc="best")

        plt.tight_layout()

        # Save both PNG and PDF versions
        plt.savefig(f"plots/{category}_muon_spin_correlations.png", dpi=300)

        print(f"{category.capitalize()} plot saved.")


if __name__ == "__main__":
    run()
