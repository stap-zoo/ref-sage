# hash.py
# GMiMC: GMiMCPerm, GMiMC2Perm (permutation) and its mode functions (GMiMCHash; GMiMC2Hash, GMiMC2Compress).

from gmimc.params import GMiMCParams, GMiMC2Params
from utils.matrix import matvecmul, vecadd, vecsub
from utils.primitive import Permutation, HashFunction, CompressionFunction


class GMiMCPerm(Permutation):
    # ---------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------

    def __init__(self, params: GMiMCParams):
        super().__init__(params)  # F, to_field, from_field, t, p, kappa, toy
        self.alpha = params.alpha

        # Rounds
        self.R = params.R

        # Linear layers
        self.M = params.M
        self.M_inv = params.M_inv

        # Constants
        self.rcons = params.rcons

    # ---------------------------------------------------------------------------
    # Component functions
    # ---------------------------------------------------------------------------

    def nonlinear_layer(self, x_in: list, r: int) -> list:
        x_out = x_in.copy()
        x_0_pow = (x_out[0] + self.rcons[r])**self.alpha
        for i in range(1, self.t):
            x_out[i] += x_0_pow
        return x_out
    
    def nonlinear_layer_inv(self, x_in: list, r: int) -> list:
        x_out = x_in.copy()
        x_0_pow = (x_out[0] + self.rcons[r])**self.alpha
        for i in range(1, self.t):
            x_out[i] -= x_0_pow
        return x_out

    def linear_layer(self, state: list, r: int) -> list:
        return matvecmul(self.M, state)

    def linear_layer_inv(self, state: list, r: int) -> list:
        return matvecmul(self.M_inv, state)

    # ---------------------------------------------------------------------------
    # Pre-/post-round steps (GMiMC does no work outside the loop: identities)
    # ---------------------------------------------------------------------------

    def _pre_rounds(self, state: list) -> list:
        return state

    def _pre_rounds_inv(self, state: list) -> list:
        return state

    def _post_rounds(self, state: list) -> list:
        return state

    def _post_rounds_inv(self, state: list) -> list:
        return state

    # ---------------------------------------------------------------------------
    # Permutation
    # ---------------------------------------------------------------------------

    def permute(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self._pre_rounds(state)
        for r in range(self.R):
            state = self.nonlinear_layer(state, r)
            state = self.linear_layer(state, r)
        return self._post_rounds(state)

    def permute_inv(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        state = self._post_rounds_inv(state)
        for r in reversed(range(self.R)):
            state = self.linear_layer_inv(state, r)
            state = self.nonlinear_layer_inv(state, r)
        return self._pre_rounds_inv(state)

class GMiMC2Perm(GMiMCPerm):
    # ---------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------

    def __init__(self, params: GMiMC2Params):
        super().__init__(params)  # base fields + alpha, R, M, M_inv, rcons
        # Extra linear layers (input/output mixing)
        self.M_IO = params.M_IO
        self.M_IO_inv = params.M_IO_inv

    # ---------------------------------------------------------------------------
    # Modified component functions
    # ---------------------------------------------------------------------------
    def nonlinear_layer(self, x_in: list, r: int) -> list:
        x_out = x_in.copy()
        x_out[0] += self.rcons[r]
        x_0_pow = x_out[0]**self.alpha
        for i in range(1, self.t):
            x_out[i] += x_0_pow
        return x_out
    
    def nonlinear_layer_inv(self, x_in: list, r: int) -> list:
        x_out = x_in.copy()
        x_0_pow = x_out[0]**self.alpha
        x_out[0] -= self.rcons[r]
        for i in range(1, self.t):
            x_out[i] -= x_0_pow
        return x_out
    
    # ---------------------------------------------------------------------------
    # Modified pre-/post-round steps
    # ---------------------------------------------------------------------------

    def _pre_rounds(self, state: list) -> list:
        return matvecmul(self.M_IO, state)

    def _pre_rounds_inv(self, state: list) -> list:
        return matvecmul(self.M_IO_inv, state)

    def _post_rounds(self, state: list) -> list:
        return matvecmul(self.M_IO, state)

    def _post_rounds_inv(self, state: list) -> list:
        return matvecmul(self.M_IO_inv, state)
    

# ---------------------------------------------------------------------------
# Hash / compression functions
#
# The permutations above are JUST permutations. Base GMiMC has a single mode, the
# length-encoded sponge; GMiMC2 additionally defines a Jive compression -- so GMiMC2 carries
# its own two mode functions, and compression exists ONLY for GMiMC2, not for base GMiMC:
#     P  = GMiMCPerm(params);  H  = GMiMCHash(P, params.sponge)        # base GMiMC: sponge only
#     P2 = GMiMC2Perm(params); H2 = GMiMC2Hash(P2, params.sponge)      # GMiMC2 sponge
#                              C2 = GMiMC2Compress(P2, params.comp)    # GMiMC2 Jive (a*d -> d)
# ---------------------------------------------------------------------------

class GMiMCHash(HashFunction):
    SPONGE_KIND = "le"        # length-encoded sponge (fixed-length input)

class GMiMC2Hash(HashFunction):
    SPONGE_KIND = "le"        # length-encoded sponge (fixed-length input)

class GMiMC2Compress(CompressionFunction):
    COMP_KIND = "jive"        # Jive_a compression (comp=dict(a=..)); GMiMC2 only
