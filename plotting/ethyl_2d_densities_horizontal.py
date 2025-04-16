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

# Constants and data paths
WORK_DIR = "../../runs"

# Atom positions in bohr
atom_positions = {
    "C1": (1.39935, 1.73874, 1.93792),
    "C2": (4.14380, 1.56961, 2.53999),
    "H1": (0.75986, 2.44815, 0.12264),
    "H2": (-0.00359, 0.87872, 3.16265),
    "H3": (4.48886, 1.62214, 4.58335),
    "H4": (4.99361, -0.20730, 1.84570),
    "H5": (5.22000, 3.11163, 1.66750),
}

# Calculate center between two carbon atoms
CENTER = (np.array(atom_positions["C1"]) + np.array(atom_positions["C2"])) / 2

# Center atom positions
centered_positions = {
    atom: np.array(pos) - CENTER for atom, pos in atom_positions.items()
}

# Updated atom colors with all hydrogens in standard blue
atom_colors = {
    "C1": "black",
    "C2": "black",
    "H1": "blue",  # Back to standard "blue"
    "H2": "blue",
    "H3": "blue",
    "H4": "blue",
    "H5": "blue",
}


# Function to transform coordinates
def transform(samples):
    # Apply a rotation matrix to align with the C1-C2 axis
    # First center at C1
    samples = samples - np.array(centered_positions["C1"])

    c1_to_c2 = np.array(atom_positions["C2"]) - np.array(atom_positions["C1"])
    h1_to_h2 = np.array(atom_positions["H2"]) - np.array(atom_positions["H1"])

    # Define new coordinate system
    e1 = c1_to_c2 / np.linalg.norm(c1_to_c2)

    b_proj_onto_e1 = np.dot(h1_to_h2, e1)
    b_perp = h1_to_h2 - (b_proj_onto_e1 * e1)
    e2 = b_perp / np.linalg.norm(b_perp)

    e3 = np.cross(e1, e2)

    # Rotation matrix
    M = np.column_stack((e1, e2, e3)).T

    # Apply rotation
    rotated_samples = (M @ samples.T).T

    # Center between C1 and C2
    c1_to_c2_distance = np.linalg.norm(
        np.array(atom_positions["C1"]) - np.array(atom_positions["C2"])
    )
    translated = rotated_samples - (np.array([c1_to_c2_distance, 0, 0])) / 2

    return translated


# Transform atom positions
transformed_positions = {
    atom: transform(np.array([pos])).flatten()
    for atom, pos in centered_positions.items()
}

print(transformed_positions)


# Function to add atom markers
def add_atom_markers(ax, projection, atoms=transformed_positions, exclude=None):
    if exclude is None:
        exclude = []

    for label, pos in atoms.items():
        if label in exclude:
            continue

        if projection == "xy":
            x, y = pos[0], pos[1]
            if label.startswith("C"):  # Carbon atoms
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
            if label.startswith("C"):  # Carbon atoms
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
            if label.startswith("C"):  # Carbon atoms
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
    fig, axs = plt.subplots(1, 3, figsize=(18, 12))  # Further increased figure height

    # Set consistent font sizes - increased by another 2 points
    plt.rcParams.update({"font.size": 18})  # Increased from 16
    SMALL_SIZE = 18  # Increased from 16
    MEDIUM_SIZE = 20  # Increased from 18
    LARGE_SIZE = 22  # Increased from 20

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=LARGE_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize

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

    # Find min/max from XZ and YZ projections for spin density (excluding XY)
    spin_min = min(np.min(s_xz_projection), np.min(s_yz_projection))
    spin_max = max(np.max(s_xz_projection), np.max(s_yz_projection))

    print("Spin min", spin_min)
    print("Spin max", spin_max)

    # Calculate global min/max for muon density across all projections
    muon_min = min(
        np.min(m_xy_projection), np.min(m_xz_projection), np.min(m_yz_projection)
    )
    muon_max = max(
        np.max(m_xy_projection), np.max(m_xz_projection), np.max(m_yz_projection)
    )

    # XY projection
    # Spin density with normalization from XZ and YZ projections
    im_s_xy = axs[0].imshow(
        s_xy_projection,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap=spin_cmap,
        alpha=0.8,
        vmin=spin_min,
        vmax=spin_max,
    )

    # Muon density with consistent normalization
    im_m_xy = axs[0].imshow(
        m_xy_projection,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap=muon_cmap,
        alpha=m_xy_alpha * 0.6,
        vmin=muon_min,
        vmax=muon_max,
    )

    axs[0].set_title("XY Projection")
    axs[0].set_xlabel("X (bohr)")
    axs[0].set_ylabel("Y (bohr)")
    axs[0].yaxis.labelpad = -5
    add_atom_markers(axs[0], "xy", exclude=["H4"])
    axs[0].grid(alpha=0.3, linestyle=":")
    axs[0].set_xticks(np.arange(-3.5, 4.0, 1.0))

    # XZ projection
    # Spin density with consistent normalization
    im_s_xz = axs[1].imshow(
        s_xz_projection,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap=spin_cmap,
        alpha=0.8,
        vmin=spin_min,
        vmax=spin_max,
    )

    # Muon density with consistent normalization
    im_m_xz = axs[1].imshow(
        m_xz_projection,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap=muon_cmap,
        alpha=m_xz_alpha * 0.6,
        vmin=muon_min,
        vmax=muon_max,
    )

    axs[1].set_title("XZ Projection")
    axs[1].set_xlabel("X (bohr)")
    axs[1].set_ylabel("Z (bohr)")
    axs[1].yaxis.labelpad = -5
    add_atom_markers(axs[1], "xz", exclude=["H4"])
    axs[1].grid(alpha=0.3, linestyle=":")
    axs[1].set_xticks(np.arange(-3.5, 4.0, 1.0))

    # YZ projection
    # Spin density with consistent normalization
    im_s_yz = axs[2].imshow(
        s_yz_projection,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap=spin_cmap,
        alpha=0.8,
        vmin=spin_min,
        vmax=spin_max,
    )

    # Muon density with consistent normalization
    im_m_yz = axs[2].imshow(
        m_yz_projection,
        extent=[-max_range, max_range, -max_range, max_range],
        origin="lower",
        cmap=muon_cmap,
        alpha=m_yz_alpha * 0.6,
        vmin=muon_min,
        vmax=muon_max,
    )

    axs[2].set_title("YZ Projection")
    axs[2].set_xlabel("Y (bohr)")
    axs[2].set_ylabel("Z (bohr)")
    axs[2].yaxis.labelpad = -5
    add_atom_markers(axs[2], "yz", exclude=["H4"])
    axs[2].grid(alpha=0.3, linestyle=":")
    axs[2].set_xticks(np.arange(-3.5, 4.0, 1.0))

    # Get actual min and max values for spin density - use only XZ and YZ projections
    # to match what's actually shown in the plots
    all_projections = np.concatenate(
        [
            s_xz_projection.flatten(),
            s_yz_projection.flatten(),
            # s_xy_projection.flatten(),  # Removed XY projection
        ]
    )

    min_value = np.min(all_projections)
    max_value = np.max(all_projections)

    # Round up to nearest 100
    min_value_rounded = np.ceil(abs(min_value) / 100) * 100
    max_value_rounded = np.ceil(max_value / 100) * 100

    # Also get global min/max for muon density across all projections
    all_muon_projections = np.concatenate(
        [
            m_xy_projection.flatten(),
            m_xz_projection.flatten(),
            m_yz_projection.flatten(),
        ]
    )
    muon_min = np.min(all_muon_projections)
    muon_max = np.max(all_muon_projections)

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

    # Define position parameters
    cbar_height = 0.03
    vertical_spacing = (
        0.08  # Increased from 0.06 to add more space between spin colorbars
    )

    # Lengthen the colorbars to nearly reach the middle from each side
    left_cbar_width = 0.45
    right_cbar_width = 0.45

    # Move bottom position - only slightly away from figures
    bottom_position = 0.34  # More subtle adjustment from 0.36

    # Left side colorbars start further left and extend more toward the middle
    left_start = 0.03

    # Left side for spin density colorbars
    left_pos_bar = fig.add_axes(
        [left_start, bottom_position, left_cbar_width, cbar_height]
    )
    left_neg_bar = fig.add_axes(
        [
            left_start,
            bottom_position - (cbar_height + vertical_spacing),
            left_cbar_width,
            cbar_height,
        ]
    )

    # Right side colorbars start closer to the middle
    right_start = 0.52

    # Right side for muon density colorbar
    right_muon_bar = fig.add_axes(
        [right_start, bottom_position, right_cbar_width, cbar_height]
    )

    # Calculate legend position - centered with respect to the negative spin density colorbar
    legend_vertical_position = (
        bottom_position - (cbar_height + vertical_spacing) - 0.01
    )  # Reduced offset to keep legend centered

    # Add horizontal colorbars with larger font sizes
    fontsize = 20  # For colorbar labels
    tick_fontsize = 16  # Further reduced size for tick numbers (from 18 to 16)

    # Spin density colorbars (left side)
    cbar_pos = fig.colorbar(sm_positive, cax=left_pos_bar, orientation="horizontal")
    cbar_pos.set_label("Positive Spin Density (Arb. Units)", fontsize=fontsize)
    cbar_pos.ax.tick_params(labelsize=tick_fontsize)

    cbar_neg = fig.colorbar(sm_negative, cax=left_neg_bar, orientation="horizontal")
    cbar_neg.set_label("Negative Spin Density (Arb. Units)", fontsize=fontsize)
    cbar_neg.ax.tick_params(labelsize=tick_fontsize)

    # Muon density colorbar (right side) - use the global scalar mappable instead of just XY data
    cbar_muon = fig.colorbar(sm_muon, cax=right_muon_bar, orientation="horizontal")
    cbar_muon.set_label("Muon Density (Arb. Units)", fontsize=fontsize)
    cbar_muon.ax.tick_params(labelsize=tick_fontsize)

    # Add a legend with both density types and atoms
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
            markeredgecolor="black",  # Add black outline in legend
            markeredgewidth=0.8,
            markersize=10,
            label="C",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markeredgecolor="black",  # Add black outline in legend
            markeredgewidth=0.8,
            markersize=10,
            label="H",
        ),
    ]

    # Place the legend with two rows for better layout
    fig.legend(
        handles=legend_elements,
        loc="center",
        bbox_to_anchor=(0.735, legend_vertical_position),
        ncol=3,  # Use 3 columns for 5 elements (3 in first row, 2 in second)
        fontsize=fontsize,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.0,
        labelspacing=0.3,
    )

    # Add grid with alpha=0.3
    for ax in axs:
        ax.grid(alpha=0.3)

    # Add super title positioned close to the figure
    fig.suptitle(
        "Muoniated Ethyl Radical: Projections of Spin Density and Muon Density",
        fontsize=26,
        y=0.97,
    )  # Using the same parameters as methyl plot

    # Apply tight layout first for general spacing
    plt.tight_layout()

    # Then adjust specific spacing parameters with the same values as methyl plot
    plt.subplots_adjust(
        bottom=0.40,  # Same spacing as methyl plot
        wspace=0.12,  # Same spacing as methyl plot
        top=0.92,  # Same spacing as methyl plot
    )

    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close("all")


def run():
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    # Load position data - ethyl has 9 spin up and 8 spin down electrons
    position_data = np.load(f"{WORK_DIR}/ethyl_inference_11810814/inference_out.npy")[
        :1000, :, :, :
    ]
    print("Position data shape", position_data.shape)

    # Determine dimensions from the data
    UP = 9
    DOWN = 8
    ELECTRONS = UP + DOWN
    TOTAL = ELECTRONS + 1

    # Reshape for processing
    position_data = position_data.reshape(-1, 4096, 3 * TOTAL)
    slice_num = position_data.shape[0]

    # Extract electron and muon positions
    spin_up_electron_samples_raw = position_data[:, :, : UP * 3].reshape(-1, 3)
    spin_down_electron_samples_raw = position_data[
        :, :, UP * 3 : ELECTRONS * 3
    ].reshape(-1, 3)
    muon_samples_raw = position_data[:, :, ELECTRONS * 3 :].reshape(-1, 3)

    print("Extracting and transforming coordinates...")

    # Apply transformation to get coordinates in the proper reference frame
    spin_up_electron_samples = transform(spin_up_electron_samples_raw)
    spin_down_electron_samples = transform(spin_down_electron_samples_raw)
    muon_samples = transform(muon_samples_raw)

    # Histogram parameters
    bins = 100
    max_range = 3.5

    print("Creating histograms...")

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

    print("Plotting density projections...")

    # Plot the density projections
    plot_density_projections(
        muon_hist, spin_density_hist, max_range, "plots/ethyl_density_projections.png"
    )

    print("Ethyl density projection plots created successfully!")


if __name__ == "__main__":
    run()
