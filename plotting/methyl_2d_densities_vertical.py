import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Patch

# Set a professional publication style for physics research
plt.style.use("seaborn-v0_8-paper")  # Professional paper style

# Set up better plotting defaults
mpl.rcParams.update(
    {
        "font.family": "serif",  # Use serif font
        "font.serif": ["DejaVu Serif"],  # Computer Modern font
        "text.usetex": False,  # Don't use LaTeX rendering
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

# Constants and data that were previously imported
WORK_DIR = "../../runs"

# Atom positions in bohr
atom_positions = {
    "C": (1.90428, 1.34926, 0.28818),
    "H1": (2.30584, 3.34047, 0.54178),
    "H2": (3.42929, -0.01587, 0.32636),
    "H3": (-0.02268, 0.72225, -0.00302),
}

# Centered at the carbon atom
carbon_position = atom_positions["C"]
centered_positions = {
    atom: (
        pos[0] - carbon_position[0],
        pos[1] - carbon_position[1],
        pos[2] - carbon_position[2],
    )
    for atom, pos in atom_positions.items()
}

# Updated atom colors with all hydrogens in standard blue
atom_colors = {
    "C": "black",
    "H1": "blue",  # Back to standard "blue"
    "H2": "blue",
    "H3": "blue",
}


# Function to add atom markers
def add_atom_markers(ax, projection, atoms=centered_positions, exclude=None):
    if exclude is None:
        exclude = []

    for label, pos in atoms.items():
        if label in exclude:
            continue

        if projection == "xy":
            x, y = pos[0], pos[1]
            if label == "C":
                ax.plot(x, y, "o", markersize=8, color=atom_colors[label])
            else:  # Hydrogen atoms - add outline
                ax.plot(
                    x,
                    y,
                    "o",
                    markersize=8,
                    markerfacecolor=atom_colors[label],
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                )
        elif projection == "xz":
            x, z = pos[0], pos[2]
            if label == "C":
                ax.plot(x, z, "o", markersize=8, color=atom_colors[label])
            else:  # Hydrogen atoms - add outline
                ax.plot(
                    x,
                    z,
                    "o",
                    markersize=8,
                    markerfacecolor=atom_colors[label],
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                )
        elif projection == "yz":
            y, z = pos[1], pos[2]
            if label == "C":
                ax.plot(y, z, "o", markersize=8, color=atom_colors[label])
            else:  # Hydrogen atoms - add outline
                ax.plot(
                    y,
                    z,
                    "o",
                    markersize=8,
                    markerfacecolor=atom_colors[label],
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                )


def plot_density_projections(
    muon_density_hist, spin_density_hist, max_range, output_filename
):
    """
    Plot muon density and spin density on the same axes for each projection plane
    Both densities shown as colored plots with different colormaps
    Vertical layout with colorbars on the right
    """
    # Create figure with appropriate dimensions - wider and taller now
    fig = plt.figure(figsize=(10, 18))  # Slightly wider to accommodate 3 columns

    fig.subplots_adjust()

    # Main grid: 1x3 (one row, three columns)
    main_gs = plt.GridSpec(
        1, 5, width_ratios=[10, 1, 1, 7, 1], figure=fig, wspace=0, hspace=0
    )

    # First column: 3 vertically stacked projection plots
    left_gs = GridSpecFromSubplotSpec(3, 1, subplot_spec=main_gs[0, 0])

    # Second column: 2 rows - positive spin and muon density
    middle_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=main_gs[0, 2])

    # Third column: 2 rows - negative spin and legend
    right_gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=main_gs[0, 4])

    # Create main plot axes (left column, stacked vertically)
    ax_xy = fig.add_subplot(left_gs[0, 0])  # Top - XY projection
    ax_xz = fig.add_subplot(left_gs[1, 0])  # Middle - XZ projection
    ax_yz = fig.add_subplot(left_gs[2, 0])  # Bottom - YZ projection
    axs = [ax_xy, ax_xz, ax_yz]

    # Create colorbar axes (middle and right columns)
    ax_pos_spin = fig.add_subplot(middle_gs[0, 0])  # Middle top - positive spin
    ax_muon = fig.add_subplot(middle_gs[1, 0])  # Middle bottom - muon density

    ax_neg_spin = fig.add_subplot(right_gs[0, 0])  # Right top - negative spin
    ax_legend = fig.add_subplot(right_gs[1, 0])  # Right bottom - legend

    # Set consistent font sizes
    plt.rcParams.update({"font.size": 18})
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    LARGE_SIZE = 22

    plt.rc("font", size=SMALL_SIZE)
    plt.rc("axes", titlesize=LARGE_SIZE)
    plt.rc("axes", labelsize=MEDIUM_SIZE)
    plt.rc("xtick", labelsize=SMALL_SIZE)
    plt.rc("ytick", labelsize=SMALL_SIZE)
    plt.rc("legend", fontsize=MEDIUM_SIZE)

    # Calculate projections
    # Muon density projections
    m_xy_projection = np.sum(muon_density_hist, axis=2).T  # Project along Z
    m_xz_projection = np.sum(muon_density_hist, axis=1).T  # Project along Y
    m_yz_projection = np.sum(muon_density_hist, axis=0).T  # Project along X

    # Spin density projections
    s_xy_projection = np.sum(spin_density_hist, axis=2).T  # Project along Z
    s_xz_projection = np.sum(spin_density_hist, axis=1).T  # Project along Y
    s_yz_projection = np.sum(spin_density_hist, axis=0).T  # Project along X

    # Normalize muon density for better visualization and masking
    m_xy_norm = (
        m_xy_projection / np.max(m_xy_projection)
        if np.max(m_xy_projection) > 0
        else m_xy_projection
    )
    m_xz_norm = (
        m_xz_projection / np.max(m_xz_projection)
        if np.max(m_xz_projection) > 0
        else m_xz_projection
    )
    m_yz_norm = (
        m_yz_projection / np.max(m_yz_projection)
        if np.max(m_yz_projection) > 0
        else m_yz_projection
    )

    # Create masks for near-zero values only for muon density
    zero_threshold = 0.05  # Adjust as needed
    m_xy_mask = np.abs(m_xy_norm) < zero_threshold
    m_xz_mask = np.abs(m_xz_norm) < zero_threshold
    m_yz_mask = np.abs(m_yz_norm) < zero_threshold

    # Create alpha arrays for muon density (1 = fully visible, 0 = fully transparent)
    m_xy_alpha = np.ones_like(m_xy_norm)
    m_xy_alpha[m_xy_mask] = 0

    m_xz_alpha = np.ones_like(m_xz_norm)
    m_xz_alpha[m_xz_mask] = 0

    m_yz_alpha = np.ones_like(m_yz_norm)
    m_yz_alpha[m_yz_mask] = 0

    # Set a light background color for all plots
    bg_color = "lightgrey"
    for ax in axs:
        ax.set_facecolor(bg_color)

    # Hide axes for the legend subplot (we'll use it for custom legend)
    ax_legend.axis("off")

    # Define colormap for spin density visualization
    spin_cmap = "coolwarm"
    muon_cmap = "magma_r"

    # Find min/max from XZ and YZ projections for spin density
    spin_min = min(np.min(s_xz_projection), np.min(s_yz_projection))
    spin_max = max(np.max(s_xz_projection), np.max(s_yz_projection))

    # Calculate global min/max for muon density across all projections
    muon_min = min(
        np.min(m_xy_projection), np.min(m_xz_projection), np.min(m_yz_projection)
    )
    muon_max = max(
        np.max(m_xy_projection), np.max(m_xz_projection), np.max(m_yz_projection)
    )

    # XY projection (top)
    # Spin density with normalization
    im_s_xy = ax_xy.imshow(
        s_xy_projection,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap=spin_cmap,
        alpha=0.8,
        vmin=spin_min,
        vmax=spin_max,
    )

    # Muon density with normalization
    im_m_xy = ax_xy.imshow(
        m_xy_projection,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap=muon_cmap,
        alpha=m_xy_alpha * 0.6,
        vmin=muon_min,
        vmax=muon_max,
    )

    ax_xy.set_title("XY Projection")
    ax_xy.set_xlabel("X (bohr)")
    ax_xy.set_ylabel("Y (bohr)")
    add_atom_markers(ax_xy, "xy", exclude=["H3"])
    ax_xy.grid(alpha=0.3, linestyle=":")

    # XZ projection (middle)
    # Spin density with normalization
    im_s_xz = ax_xz.imshow(
        s_xz_projection,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap=spin_cmap,
        alpha=0.8,
        vmin=spin_min,
        vmax=spin_max,
    )

    # Muon density with normalization
    im_m_xz = ax_xz.imshow(
        m_xz_projection,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap=muon_cmap,
        alpha=m_xz_alpha * 0.6,
        vmin=muon_min,
        vmax=muon_max,
    )

    ax_xz.set_title("XZ Projection")
    ax_xz.set_xlabel("X (bohr)")
    ax_xz.set_ylabel("Z (bohr)")
    add_atom_markers(ax_xz, "xz", exclude=["H3"])
    ax_xz.grid(alpha=0.3, linestyle=":")

    # YZ projection (bottom)
    # Spin density with normalization
    im_s_yz = ax_yz.imshow(
        s_yz_projection,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap=spin_cmap,
        alpha=0.8,
        vmin=spin_min,
        vmax=spin_max,
    )

    # Muon density with normalization
    im_m_yz = ax_yz.imshow(
        m_yz_projection,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap=muon_cmap,
        alpha=m_yz_alpha * 0.6,
        vmin=muon_min,
        vmax=muon_max,
    )

    ax_yz.set_title("YZ Projection")
    ax_yz.set_xlabel("Y (bohr)")
    ax_yz.set_ylabel("Z (bohr)")
    add_atom_markers(ax_yz, "yz", exclude=["H3"])
    ax_yz.grid(alpha=0.3, linestyle=":")

    # Get actual min and max values for spin density across all projections
    all_projections = np.concatenate(
        [
            s_xy_projection.flatten(),
            s_xz_projection.flatten(),
            s_yz_projection.flatten(),
        ]
    )

    min_value = np.min(all_projections)
    max_value = np.max(all_projections)

    # Round up to nearest 100
    min_value_rounded = np.ceil(abs(min_value) / 100) * 100
    max_value_rounded = np.ceil(max_value / 100) * 100

    # Create custom colormaps for positive and negative parts of coolwarm
    # Get the colormap colors
    coolwarm = plt.cm.coolwarm
    coolwarm_colors = coolwarm(np.linspace(0, 1, 256))

    # Create a colormap for just the positive values (middle to right of coolwarm)
    positive_colors = coolwarm_colors[128:, :]  # From center (white) to red
    positive_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "positive_cmap", positive_colors
    )

    # Create a colormap for just the negative values (left to middle of coolwarm)
    negative_colors = coolwarm_colors[:129, :]  # From blue to center (white)
    negative_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "negative_cmap", negative_colors
    )

    # Create normalization objects
    norm_positive = mpl.colors.Normalize(vmin=0, vmax=max_value_rounded)
    norm_negative = mpl.colors.Normalize(vmin=-min_value_rounded, vmax=0)

    # Create ScalarMappables for the colorbars
    sm_positive = plt.cm.ScalarMappable(cmap=positive_cmap, norm=norm_positive)
    sm_negative = plt.cm.ScalarMappable(cmap=negative_cmap, norm=norm_negative)

    # Create a ScalarMappable for the muon colorbar that represents all projections
    norm_muon = mpl.colors.Normalize(vmin=muon_min, vmax=muon_max)
    sm_muon = plt.cm.ScalarMappable(cmap=muon_cmap, norm=norm_muon)

    # Add vertical colorbars
    fontsize = 18  # For colorbar labels
    tick_fontsize = 16  # For tick numbers

    # Positive spin density colorbar - single line text with increased padding
    cbar_pos = fig.colorbar(sm_positive, cax=ax_pos_spin, orientation="vertical")
    cbar_pos.set_label(
        "Positive Spin Density (Arb. Units)", fontsize=fontsize, labelpad=15
    )  # Added labelpad
    cbar_pos.ax.tick_params(labelsize=tick_fontsize)

    # Negative spin density colorbar - single line text with increased padding
    cbar_neg = fig.colorbar(sm_negative, cax=ax_neg_spin, orientation="vertical")
    cbar_neg.set_label(
        "Negative Spin Density (Arb. Units)", fontsize=fontsize, labelpad=15
    )  # Added labelpad
    cbar_neg.ax.tick_params(labelsize=tick_fontsize)

    # Muon density colorbar - single line text with increased padding
    cbar_muon = fig.colorbar(sm_muon, cax=ax_muon, orientation="vertical")
    cbar_muon.set_label(
        "Muon Density (Arb. Units)", fontsize=fontsize, labelpad=15
    )  # Added labelpad
    cbar_muon.ax.tick_params(labelsize=tick_fontsize)

    # Add a legend
    legend_elements = [
        Patch(
            facecolor="#FF4444",
            edgecolor="#FF4444",
            alpha=0.8,
            label="Spin Density (+)",
        ),
        Patch(
            facecolor="#4444FF",
            edgecolor="#4444FF",
            alpha=0.8,
            label="Spin Density (-)",
        ),
        Patch(facecolor="purple", edgecolor="purple", alpha=0.6, label="Muon Density"),
        # Add simplified atom markers - just C and H
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="black",
            markersize=10,
            label="C",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=10,
            label="H",
        ),
    ]

    # Place the legend in the bottom right section with better layout
    # Make it a single column vertical legend
    legend = ax_legend.legend(
        handles=legend_elements,
        loc="center",
        fontsize=fontsize,
        frameon=False,
        ncol=1,  # Single column
        handletextpad=0.5,
        labelspacing=1.0,
    )

    # Add grid with alpha=0.3
    for ax in axs:
        ax.grid(alpha=0.3)

    # Add title centered over the entire figure using fig.text instead of suptitle
    # First remove any existing suptitle if present
    if fig._suptitle is not None:
        fig._suptitle.set_visible(False)

    # Adjust spacing between subplots
    plt.subplots_adjust(
        hspace=0.3,  # Space between the projection plots
        wspace=0.3,  # Reduced to zero space between plots and colorbars
    )

    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close("all")


def run():
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    # Load position data
    position_data = np.load(f"{WORK_DIR}/methyl_inference_11780517/inference_out.npy")
    print("Position data shape", position_data.shape)

    # Reshape for processing
    position_data = position_data.reshape(-1, 30)

    # Extract electron and muon positions
    spin_up_electron_samples_raw = position_data[:, : 5 * 3].reshape(-1, 3)
    spin_down_electron_samples_raw = position_data[:, 5 * 3 : 9 * 3].reshape(-1, 3)
    muon_samples_raw = position_data[:, 9 * 3 :].reshape(-1, 3)

    # Center around carbon atom
    spin_up_electron_samples = spin_up_electron_samples_raw - np.tile(
        atom_positions["C"], (len(spin_up_electron_samples_raw), 1)
    )
    spin_down_electron_samples = spin_down_electron_samples_raw - np.tile(
        atom_positions["C"], (len(spin_down_electron_samples_raw), 1)
    )
    muon_samples = muon_samples_raw - np.tile(
        atom_positions["C"], (len(muon_samples_raw), 1)
    )

    # Histogram parameters
    bins = 100
    max_range = 3

    # Create 3D histograms
    spin_up_hist, _ = np.histogramdd(
        spin_up_electron_samples,
        bins=bins,
        range=[[-max_range, max_range]] * 3,
    )
    spin_down_hist, _ = np.histogramdd(
        spin_down_electron_samples,
        bins=bins,
        range=[[-max_range, max_range]] * 3,
    )
    muon_hist, _ = np.histogramdd(
        muon_samples,
        bins=bins,
        range=[[-max_range, max_range]] * 3,
    )

    # Calculate spin density
    spin_density_hist = spin_up_hist - spin_down_hist

    # Plot the density projections
    plot_density_projections(
        muon_hist, spin_density_hist, max_range, "plots/methyl_density_projections.png"
    )

    print("Density projection plots created successfully!")


if __name__ == "__main__":
    run()
