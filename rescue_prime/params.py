from rescue.params import RescueParams
from utils import vandermonde_mds_matrix, FieldElementSampler
from complexities import gb_comp
from math import ceil, floor

class RescuePrimeParams(RescueParams):
    """Same parameters as Rescue, with some simplifications:
        - round constants: instead of deriving constants through the block cipher's key schedule with
        the zero key, constants are generated directly by expanding a seed string with SHAKE-256
        - security margin: reduced from 100% to 50%
    """
    LABEL = "Rescue-XLIX"
    
    def _l1(self) -> int:
        """Instance-specific number of rounds that can be attacked by a Gröbner basis attack"""
        nvar = lambda r : self.t * (r-1) + self.d # number of variables/equations
        dcon = lambda r : floor(0.5 * (self.alpha - 1) * self.t * (r - 1) + 2) # extrapolation for observed solving degree
        R = 1
        while gb_comp(dreg=dcon(R), nv=nvar(R), w=2) < self.kappa:
            R += 1
        return R

    def _init_rounds(self) -> int:
        """Round number derivation, including 50% security margin"""
        # Rescue Prime: 50% security margin over the Groebner-basis bound.
        return ceil(1.5 * max(5, self._l1()))

    def _init_mds(self) -> list[list[int]]:
        return vandermonde_mds_matrix(self.p, self.t, self.g, transpose=True)

    def _init_rcons(self) -> list[list[int]]:
        seed = f"{self.LABEL}({self.p},{self.t},{self.c},{self.kappa})".encode("ascii")
        return FieldElementSampler(seed, self.p, xof="shake_256", sampling="mod").grid(2 * self.R, self.t)
