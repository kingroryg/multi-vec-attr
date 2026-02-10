"""
Spreading Code Generation for CDW

Implements Walsh-Hadamard, Gold, and random code families for multi-vendor
watermarking. Each code family has different properties:

- Walsh-Hadamard: Perfect orthogonality, supports exactly L vendors for length L
- Gold: Near-orthogonal, supports L+2 vendors for length L=2^n-1
- Random: No orthogonality guarantees, for baseline comparison
"""

import numpy as np
from enum import Enum
from typing import List, Tuple, Optional
from scipy.linalg import hadamard


class CodeFamily(Enum):
    """Supported spreading code families."""
    WALSH_HADAMARD = "walsh_hadamard"
    GOLD = "gold"
    RANDOM = "random"


def generate_walsh_hadamard_codes(num_codes: int, code_length: Optional[int] = None) -> np.ndarray:
    """
    Generate Walsh-Hadamard orthogonal codes.

    Walsh-Hadamard codes are rows of the Hadamard matrix, providing
    perfect orthogonality: c_i @ c_j = 0 for i != j.

    Args:
        num_codes: Number of codes to generate.
        code_length: Desired code length. Must be a power of 2 and >= num_codes.
                     If None, uses smallest power of 2 >= num_codes.

    Returns:
        codes: np.ndarray of shape (num_codes, code_length) with values in {-1, +1}

    Raises:
        ValueError: If num_codes > code_length or code_length > 4096
    """
    # Determine code length
    if code_length is None:
        # Find smallest power of 2 >= num_codes
        code_length = 1
        while code_length < num_codes:
            code_length *= 2
    else:
        # Validate code_length is power of 2
        if code_length & (code_length - 1) != 0:
            # Round up to nearest power of 2
            cl = 1
            while cl < code_length:
                cl *= 2
            code_length = cl

    if code_length > 4096:
        raise ValueError(f"code_length={code_length} exceeds practical limit of 4096")

    if num_codes > code_length:
        raise ValueError(f"num_codes={num_codes} exceeds code_length={code_length}")

    # Generate Hadamard matrix
    H = hadamard(code_length)

    # Return first num_codes rows (all are orthogonal)
    codes = H[:num_codes].astype(np.float32)

    return codes


def _generate_msequence(n: int, taps: List[int]) -> np.ndarray:
    """
    Generate a maximum-length sequence (m-sequence) using LFSR.

    Args:
        n: Number of register bits (sequence length = 2^n - 1)
        taps: Feedback tap positions (0-indexed from LSB)

    Returns:
        seq: np.ndarray of length 2^n - 1 with values in {-1, +1}
    """
    length = (1 << n) - 1
    register = np.ones(n, dtype=np.int32)  # Initialize to all 1s
    sequence = np.zeros(length, dtype=np.int32)

    for i in range(length):
        sequence[i] = register[-1]

        # Compute feedback (XOR of tapped positions)
        feedback = 0
        for tap in taps:
            feedback ^= register[tap]

        # Shift register
        register = np.roll(register, 1)
        register[0] = feedback

    # Convert {0, 1} to {-1, +1}
    return 2 * sequence - 1


# Preferred pairs of m-sequences for Gold code generation
# Format: (n, taps1, taps2) where n is register length
GOLD_PREFERRED_PAIRS = {
    5: ([4, 2, 0], [4, 3, 2, 1, 0]),      # L = 31
    6: ([5, 0], [5, 4, 1, 0]),             # L = 63
    7: ([6, 0], [6, 5, 4, 0]),             # L = 127
    8: ([7, 5, 2, 1, 0], [7, 6, 5, 4, 0]), # L = 255
    9: ([8, 4, 0], [8, 6, 5, 4, 0]),       # L = 511
    10: ([9, 3, 0], [9, 8, 5, 4, 0]),      # L = 1023
}


def generate_gold_codes(num_codes: int, n: Optional[int] = None) -> np.ndarray:
    """
    Generate Gold codes with bounded cross-correlation.

    Gold codes are generated from preferred pairs of m-sequences.
    For length L = 2^n - 1, the family contains L + 2 codes with
    cross-correlation bounded by 2^((n+2)/2) + 1.

    Args:
        num_codes: Number of codes to generate
        n: Register length (determines code length L = 2^n - 1).
           If None, smallest n giving enough codes is chosen.

    Returns:
        codes: np.ndarray of shape (num_codes, code_length) with values in {-1, +1}

    Raises:
        ValueError: If num_codes exceeds family size for given n
    """
    # Auto-select n if not provided
    if n is None:
        for candidate_n in sorted(GOLD_PREFERRED_PAIRS.keys()):
            max_codes = (1 << candidate_n) - 1 + 2
            if max_codes >= num_codes:
                n = candidate_n
                break
        if n is None:
            raise ValueError(f"num_codes={num_codes} exceeds maximum Gold family size")

    if n not in GOLD_PREFERRED_PAIRS:
        raise ValueError(f"n={n} not in supported values: {list(GOLD_PREFERRED_PAIRS.keys())}")

    code_length = (1 << n) - 1
    max_family_size = code_length + 2

    if num_codes > max_family_size:
        raise ValueError(f"num_codes={num_codes} exceeds Gold family size {max_family_size} for n={n}")

    taps1, taps2 = GOLD_PREFERRED_PAIRS[n]

    # Generate the two m-sequences
    m1 = _generate_msequence(n, taps1)
    m2 = _generate_msequence(n, taps2)

    # Gold family: m1, m2, and all cyclic shifts of m1 XOR m2
    codes = []
    codes.append(m1)
    if len(codes) < num_codes:
        codes.append(m2)

    for shift in range(code_length):
        if len(codes) >= num_codes:
            break
        m2_shifted = np.roll(m2, shift)
        # XOR in {-1, +1} domain: (a * b) gives -1 where they differ, +1 where same
        gold_code = m1 * m2_shifted
        codes.append(gold_code)

    return np.array(codes, dtype=np.float32)


def generate_random_codes(
    num_codes: int,
    code_length: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate random (non-orthogonal) codes for baseline comparison.

    Random codes have expected cross-correlation of 0 but high variance,
    leading to interference at high vendor counts.

    Args:
        num_codes: Number of codes to generate
        code_length: Length of each code
        seed: Random seed for reproducibility

    Returns:
        codes: np.ndarray of shape (num_codes, code_length) with values in {-1, +1}
    """
    rng = np.random.default_rng(seed)
    codes = rng.choice([-1, 1], size=(num_codes, code_length))
    return codes.astype(np.float32)


def compute_code_statistics(codes: np.ndarray) -> dict:
    """Alias for get_code_properties for backward compatibility."""
    return get_code_properties(codes)


def get_code_properties(codes: np.ndarray) -> dict:
    """
    Compute properties of a code family for analysis.

    Args:
        codes: np.ndarray of shape (num_codes, code_length)

    Returns:
        dict with:
            - num_codes: Number of codes
            - code_length: Length of codes
            - auto_correlation: Mean auto-correlation (should be code_length)
            - max_cross_correlation: Maximum |c_i @ c_j| for i != j
            - mean_cross_correlation: Mean |c_i @ c_j| for i != j
            - orthogonality_score: 1 - (max_cross / code_length), higher is better
    """
    num_codes, code_length = codes.shape

    # Auto-correlation (diagonal of Gram matrix)
    auto_corr = np.sum(codes * codes, axis=1)

    # Cross-correlation (off-diagonal of Gram matrix)
    gram = codes @ codes.T
    mask = ~np.eye(num_codes, dtype=bool)
    cross_corr = np.abs(gram[mask])

    return {
        "num_codes": num_codes,
        "code_length": code_length,
        "auto_correlation_mean": float(np.mean(auto_corr)),
        "auto_correlation_std": float(np.std(auto_corr)),
        "max_cross_correlation": float(np.max(cross_corr)) if len(cross_corr) > 0 else 0,
        "mean_cross_correlation": float(np.mean(cross_corr)) if len(cross_corr) > 0 else 0,
        "orthogonality_score": float(1 - np.max(cross_corr) / code_length) if len(cross_corr) > 0 else 1.0,
    }


def generate_codes(
    code_family: CodeFamily,
    num_codes: int,
    code_length: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Unified interface for code generation.

    Args:
        code_family: Which code family to use
        num_codes: Number of codes to generate
        code_length: Code length (required for random, auto for others)
        seed: Random seed (for random codes)

    Returns:
        codes: np.ndarray of shape (num_codes, code_length)
        properties: dict of code family properties
    """
    if code_family == CodeFamily.WALSH_HADAMARD:
        codes = generate_walsh_hadamard_codes(num_codes, code_length)
    elif code_family == CodeFamily.GOLD:
        codes = generate_gold_codes(num_codes)
    elif code_family == CodeFamily.RANDOM:
        if code_length is None:
            raise ValueError("code_length required for random codes")
        codes = generate_random_codes(num_codes, code_length, seed)
    else:
        raise ValueError(f"Unknown code family: {code_family}")

    properties = get_code_properties(codes)
    properties["family"] = code_family.value

    return codes, properties


if __name__ == "__main__":
    # Test code generation
    print("Testing Walsh-Hadamard codes:")
    wh_codes = generate_walsh_hadamard_codes(8)
    print(f"  Shape: {wh_codes.shape}")
    print(f"  Properties: {get_code_properties(wh_codes)}")

    print("\nTesting Gold codes:")
    gold_codes = generate_gold_codes(10, n=5)
    print(f"  Shape: {gold_codes.shape}")
    print(f"  Properties: {get_code_properties(gold_codes)}")

    print("\nTesting Random codes:")
    random_codes = generate_random_codes(8, 64, seed=42)
    print(f"  Shape: {random_codes.shape}")
    print(f"  Properties: {get_code_properties(random_codes)}")
