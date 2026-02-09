"""
Ring Mask Utilities for Fourier-Domain Watermarking

Defines the annular regions in Fourier space where watermarks are embedded.
Following Tree-Ring, we use concentric rings that are invariant to rotation.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class RingMask:
    """
    Defines the Fourier-domain region for watermark embedding.

    The mask specifies which Fourier coefficients are modulated by the
    spreading code. Uses concentric rings centered at DC for rotation
    invariance.

    Attributes:
        height: Height of Fourier transform (typically latent height)
        width: Width of Fourier transform (typically latent width)
        r_min: Inner radius of annular region (pixels)
        r_max: Outer radius of annular region (pixels)
        num_rings: Number of concentric rings for code mapping
        mask: Boolean array of shape (height, width), True where embedding occurs
        ring_indices: List of (row, col) tuples for each ring
        positions: All (row, col) positions in the mask
    """
    height: int
    width: int
    r_min: float
    r_max: float
    num_rings: int
    mask: np.ndarray = None
    ring_indices: List[List[Tuple[int, int]]] = None
    positions: List[Tuple[int, int]] = None

    def __post_init__(self):
        """Compute mask and ring assignments after initialization."""
        self._compute_mask()
        self._assign_rings()

    def _compute_mask(self):
        """Compute the binary mask for the annular region."""
        # Create coordinate grids centered at DC (center of FFT)
        cy, cx = self.height // 2, self.width // 2
        y, x = np.ogrid[:self.height, :self.width]

        # Distance from center
        dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

        # Annular mask
        self.mask = (dist >= self.r_min) & (dist <= self.r_max)

        # Store all positions
        self.positions = list(zip(*np.where(self.mask)))

    def _assign_rings(self):
        """Assign mask positions to rings for code modulation."""
        cy, cx = self.height // 2, self.width // 2

        # Compute radius for each position
        self.ring_indices = [[] for _ in range(self.num_rings)]

        for y, x in self.positions:
            dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
            # Map distance to ring index
            ring_idx = int((dist - self.r_min) / (self.r_max - self.r_min) * self.num_rings)
            ring_idx = min(ring_idx, self.num_rings - 1)  # Clamp to valid range
            self.ring_indices[ring_idx].append((y, x))

    def get_ring_positions(self, ring_idx: int) -> List[Tuple[int, int]]:
        """Get all (y, x) positions belonging to a specific ring."""
        return self.ring_indices[ring_idx]

    def get_num_positions(self) -> int:
        """Total number of Fourier positions in the mask."""
        return len(self.positions)

    def get_positions_per_ring(self) -> List[int]:
        """Number of positions in each ring."""
        return [len(ring) for ring in self.ring_indices]

    def get_sign_pattern(self) -> np.ndarray:
        """
        Generate sign pattern for conjugate symmetry preservation.

        For the inverse FFT to produce real output, we need:
            Z[u, v] = conj(Z[-u, -v])

        This pattern ensures we only modify one of each conjugate pair.
        """
        sign = np.ones((self.height, self.width), dtype=np.float32)
        cy, cx = self.height // 2, self.width // 2

        for y, x in self.positions:
            # Determine if this position or its conjugate is the "primary"
            # Use lexicographic ordering: (y, x) < (-y, -x) mod (H, W)
            conj_y = (2 * cy - y) % self.height
            conj_x = (2 * cx - x) % self.width

            if (y, x) > (conj_y, conj_x):
                sign[y, x] = -1  # This is the conjugate, will be set by symmetry

        return sign

    def to_dict(self) -> dict:
        """Serialize mask parameters (not the arrays)."""
        return {
            "height": self.height,
            "width": self.width,
            "r_min": self.r_min,
            "r_max": self.r_max,
            "num_rings": self.num_rings,
            "num_positions": self.get_num_positions(),
            "positions_per_ring": self.get_positions_per_ring(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RingMask":
        """Reconstruct from serialized parameters."""
        return cls(
            height=d["height"],
            width=d["width"],
            r_min=d["r_min"],
            r_max=d["r_max"],
            num_rings=d["num_rings"],
        )


def create_default_mask(
    latent_size: int = 64,
    r_min: float = 4,
    r_max: float = 20,
    code_length: int = 64,
) -> RingMask:
    """
    Create a ring mask with default parameters for SD v1.5.

    Args:
        latent_size: Size of latent space (64 for SD v1.5)
        r_min: Inner radius of annular region
        r_max: Outer radius of annular region
        code_length: Number of rings (should match spreading code length)

    Returns:
        RingMask configured for the given parameters
    """
    return RingMask(
        height=latent_size,
        width=latent_size,
        r_min=r_min,
        r_max=r_max,
        num_rings=code_length,
    )


def visualize_mask(mask: RingMask, save_path: Optional[str] = None):
    """
    Visualize the ring mask for debugging.

    Args:
        mask: RingMask to visualize
        save_path: If provided, save figure to this path
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Binary mask
    axes[0].imshow(mask.mask, cmap='gray')
    axes[0].set_title(f'Ring Mask (r=[{mask.r_min}, {mask.r_max}])')
    axes[0].set_xlabel('Frequency (x)')
    axes[0].set_ylabel('Frequency (y)')

    # Ring assignments (colored by ring index)
    ring_viz = np.zeros((mask.height, mask.width), dtype=np.float32)
    for ring_idx, positions in enumerate(mask.ring_indices):
        for y, x in positions:
            ring_viz[y, x] = ring_idx + 1

    axes[1].imshow(ring_viz, cmap='tab20')
    axes[1].set_title(f'Ring Assignments ({mask.num_rings} rings)')
    axes[1].set_xlabel('Frequency (x)')
    axes[1].set_ylabel('Frequency (y)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    # Test mask creation
    mask = create_default_mask(latent_size=64, code_length=64)
    print(f"Mask info: {mask.to_dict()}")

    # Visualize
    visualize_mask(mask, save_path=None)
