from fields import GOLDILOCKS
from rescue_prime_optimized.params import RescuePrimeOptimizedParams
from utils import circulant

# ---------------------------------------------------------------------------
# Rescue Prime Optimized (RPO) instances
# ---------------------------------------------------------------------------

RPO_GOLDILOCKS_T12 = RescuePrimeOptimizedParams(
    p=GOLDILOCKS.p,
    t=12,
    alpha=GOLDILOCKS.alpha,
    alpha_inv=GOLDILOCKS.alpha_inv,
    kappa=128,
    c=4,
    R=7,
    M=circulant([7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8]),
    d=4,
)

RPO_GOLDILOCKS_T16 = RescuePrimeOptimizedParams(
    p=GOLDILOCKS.p,
    t=16,
    alpha=GOLDILOCKS.alpha,
    alpha_inv=GOLDILOCKS.alpha_inv,
    kappa=160,
    c=6,
    R=7,
    M=circulant([256, 2, 1073741824, 2048, 16777216, 128, 8, 16, 524288, 4194304, 1, 268435456, 1, 1024, 2, 8192]),
    d=5,
)
