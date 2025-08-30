import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
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
    """
    # Create a more compact figure
    fig = plt.figure(figsize=(14, 13))  # Increased height slightly

    # Create grid spec with adjusted spacing
    gs = fig.add_gridspec(2, 2, wspace=0.0, hspace=0.25)  # Increased hspace

    # Create axes for each projection
    ax_xz = fig.add_subplot(gs[0, 0])  # Top left: XZ
    ax_yz = fig.add_subplot(gs[0, 1])  # Top right: YZ
    ax_xy = fig.add_subplot(gs[1, 0])  # Bottom left: XY

    # Legend/colorbar panel
    colorbar_panel = fig.add_subplot(gs[1, 1])
    colorbar_panel.axis("off")  # Hide axes for this panel

    # Group axes for easier iteration
    axs = [ax_xy, ax_xz, ax_yz]

    # Set consistent font sizes
    TITLE_SIZE = 20  # For subplot titles
    AXIS_LABEL_SIZE = 18  # For axis labels
    TICK_SIZE = 16  # For tick labels
    COLORBAR_TITLE_SIZE = 18  # Same as subplot titles
    COLORBAR_TICK_SIZE = 16  # Same as axis tick sizes

    plt.rcParams.update({"font.size": 16})
    plt.rc("axes", titlesize=TITLE_SIZE)
    plt.rc("axes", labelsize=AXIS_LABEL_SIZE)
    plt.rc("xtick", labelsize=TICK_SIZE)
    plt.rc("ytick", labelsize=TICK_SIZE)
    plt.rc("legend", fontsize=AXIS_LABEL_SIZE)

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

    # Define colormap for spin density visualization
    spin_cmap = "coolwarm"
    muon_cmap = "magma_r"

    # Find min/max from XZ and YZ projections ONLY for spin density
    spin_min = min(np.min(s_xz_projection), np.min(s_yz_projection))
    spin_max = max(np.max(s_xz_projection), np.max(s_yz_projection))

    # Calculate global min/max for muon density across all projections
    muon_min = min(
        np.min(m_xy_projection), np.min(m_xz_projection), np.min(m_yz_projection)
    )
    muon_max = max(
        np.max(m_xy_projection), np.max(m_xz_projection), np.max(m_yz_projection)
    )

    # Round values for cleaner colorbar display
    spin_min_rounded = np.floor(spin_min / 100) * 100
    spin_max_rounded = np.ceil(spin_max / 100) * 100

    # Create the normalization for spin density plotting
    norm_spin = mpl.colors.Normalize(vmin=spin_min_rounded, vmax=spin_max_rounded)
    norm_muon = mpl.colors.Normalize(vmin=muon_min, vmax=muon_max)

    # XY projection (bottom left)
    im_s_xy = ax_xy.imshow(
        s_xy_projection,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap=spin_cmap,
        alpha=0.8,
        norm=norm_spin,  # Use the same normalization
    )

    im_m_xy = ax_xy.imshow(
        m_xy_projection,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap=muon_cmap,
        alpha=m_xy_alpha * 0.6,
        norm=norm_muon,
    )

    ax_xy.set_title("XY Projection")
    ax_xy.set_xlabel("X (Bohr)")
    ax_xy.set_ylabel("Y (Bohr)")
    ax_xy.yaxis.labelpad = -5
    add_atom_markers(ax_xy, "xy", exclude=["H3"])
    ax_xy.grid(alpha=0.3, linestyle=":")

    # XZ projection (top left)
    im_s_xz = ax_xz.imshow(
        s_xz_projection,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap=spin_cmap,
        alpha=0.8,
        norm=norm_spin,  # Use the same normalization
    )

    im_m_xz = ax_xz.imshow(
        m_xz_projection,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap=muon_cmap,
        alpha=m_xz_alpha * 0.6,
        norm=norm_muon,
    )

    ax_xz.set_title("XZ Projection")
    ax_xz.set_xlabel("X (Bohr)")
    ax_xz.set_ylabel("Z (Bohr)")
    ax_xz.yaxis.labelpad = -5
    add_atom_markers(ax_xz, "xz", exclude=["H3"])
    ax_xz.grid(alpha=0.3, linestyle=":")

    # YZ projection (top right)
    im_s_yz = ax_yz.imshow(
        s_yz_projection,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap=spin_cmap,
        alpha=0.8,
        norm=norm_spin,  # Use the same normalization
    )

    im_m_yz = ax_yz.imshow(
        m_yz_projection,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap=muon_cmap,
        alpha=m_yz_alpha * 0.6,
        norm=norm_muon,
    )

    ax_yz.set_title("YZ Projection")
    ax_yz.set_xlabel("Y (Bohr)")
    ax_yz.set_ylabel("Z (Bohr)")
    ax_yz.yaxis.labelpad = -5
    add_atom_markers(ax_yz, "yz", exclude=["H3"])
    ax_yz.grid(alpha=0.3, linestyle=":")

    # Get the colormap
    coolwarm = plt.cm.coolwarm

    # Extract the colors from coolwarm that correspond to our data range
    # This ensures the colorbars show exactly the same colors as the plots
    zero_point = -spin_min_rounded / (
        spin_max_rounded - spin_min_rounded
    )  # Normalized position of zero

    # Adjust bounds to match colors in the coolwarm colormap
    neg_max = abs(spin_min_rounded)
    pos_max = spin_max_rounded

    # Create custom positive and negative colormaps
    # that exactly match the full coolwarm when split at zero
    zero_idx = int(256 * zero_point)
    negative_colors = coolwarm(np.linspace(0, zero_point, zero_idx))
    positive_colors = coolwarm(np.linspace(zero_point, 1, 256 - zero_idx))

    # Create the colormaps
    negative_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "negative_cmap", negative_colors
    )
    positive_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "positive_cmap", positive_colors
    )

    # Create normalization for the colorbars
    norm_positive = mpl.colors.Normalize(vmin=0, vmax=pos_max)

    # For the negative colorbar, we'll reverse the normalization to get correct order
    norm_negative = mpl.colors.Normalize(vmin=0, vmax=neg_max)

    # Create ScalarMappables
    sm_positive = plt.cm.ScalarMappable(cmap=positive_cmap, norm=norm_positive)
    sm_negative = plt.cm.ScalarMappable(cmap=negative_cmap, norm=norm_negative)
    sm_muon = plt.cm.ScalarMappable(cmap=muon_cmap, norm=norm_muon)

    # Create colorbars with vertical spacing
    pos_cbar_ax = colorbar_panel.inset_axes([0.02, 0.9, 0.9, 0.08])  # Top colorbar
    neg_cbar_ax = colorbar_panel.inset_axes([0.02, 0.60, 0.9, 0.08])  # Middle colorbar
    muon_cbar_ax = colorbar_panel.inset_axes([0.02, 0.30, 0.9, 0.08])  # Bottom colorbar

    # Add the colorbars with updated font sizes
    cbar_pos = fig.colorbar(sm_positive, cax=pos_cbar_ax, orientation="horizontal")
    cbar_pos.set_label(
        "Positive Spin Density (Arb. Units)", fontsize=COLORBAR_TITLE_SIZE
    )
    cbar_pos.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)

    # Create negative colorbar with correct ordering
    cbar_neg = fig.colorbar(sm_negative, cax=neg_cbar_ax, orientation="horizontal")

    # Modify tick labels to show negative values in correct order
    ticks = cbar_neg.get_ticks()
    cbar_neg.ax.set_xticklabels([f"-{int(tick)}" for tick in reversed(ticks)])

    # Reverse the ticks for the negative colorbar
    neg_cbar_ax.invert_xaxis()

    cbar_neg.set_label(
        "Negative Spin Density (Arb. Units)", fontsize=COLORBAR_TITLE_SIZE
    )
    cbar_neg.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)

    cbar_muon = fig.colorbar(sm_muon, cax=muon_cbar_ax, orientation="horizontal")
    cbar_muon.set_label("Muon Density (Arb. Units)", fontsize=COLORBAR_TITLE_SIZE)
    cbar_muon.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)

    # Legend elements with newlines in labels
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="black",
            markersize=10,
            label="Carbon\nNucleus",
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
            label="Hydrogen\nNucleus",
        ),
    ]

    # Position legend even lower in the panel
    colorbar_panel.legend(
        handles=legend_elements,
        loc="center",
        bbox_to_anchor=(0.45, 0.075),  # Moved lower to 0.075
        ncol=2,
        fontsize=AXIS_LABEL_SIZE,
        frameon=False,
        handletextpad=0.2,
        columnspacing=0.8,
        labelspacing=0.2,
    )

    # Adjust the overall layout first
    plt.tight_layout()

    # Add title with newline after "Methyl Radical"
    fig.text(
        0.5,
        0.98,
        "Muoniated Methyl Radical:\nProjections of Spin Density and Muon Density",
        fontsize=22,
        ha="center",
        va="top",
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
