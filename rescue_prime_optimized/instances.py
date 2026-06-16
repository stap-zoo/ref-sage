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
    d=5,
)
