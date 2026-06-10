from math import ceil

from rescue_prime.params import RescuePrimeParams


class RescuePrimeOptimizedParams(RescuePrimeParams):
    """Same parameters as RescuePrime, with some adaptations:
        - matrix: Circulant MDS matrix instead of the Vandermonde-derived one, chosen so matrix-vector products 
        can be computed fast via Karatsuba or NTT-based polynomial multiplication (the field is NTT-friendly)
        - security margin: similar to RescuePrime, but reduced by one round
    """
    LABEL = "RPO"

    def _init_rounds(self) -> int:
        # RPO paper states that 1 round less compared to RP is fine for proposed instance
        return ceil(1.5 * max(5, self._l1())) - 1

    def _init_mds(self) -> list[list[int]]:
        raise NotImplementedError("RPO's MDS matrix is a fixed circulant matrix from the spec; pass M=circulant([...]) explicitly.")
