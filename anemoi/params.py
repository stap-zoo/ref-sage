# params.py
# ---------------------------------------------------------------------------
# Parameter definition for Anemoi: the AnemoiParams class.
#
# AnemoiParams is the single source of truth for an instance. It sanitizes the
# user-facing parameters and expands them into a fully-specified instance that
# the permutation, hash modes, instances and tests consume. Any value the user
# omits is filled in by the matching _init_* helper (or, for r/c/d, by the shared
# derive_rate_capacity_digest). Settings that depart from the recommended ones
# raise a ParamRecommendationWarning rather than an error.
# ---------------------------------------------------------------------------

# Structural imports
import warnings
from recommendations import ParamRecommendationWarning
from types import SimpleNamespace

# Math specific imports
from math import gcd
from sage.all import GF, Integer

# Custom imports
from utils.complexities import gb_comp
from utils.matrix import circulant, is_mds, pht_matrix, dl_m33_52_matrix, dl_m46_83_matrix, map_nested, invert_matrix
from utils.mode import derive_rate_capacity_digest

# Digits of pi, used to derive the round constants via an open butterfly.
PI_0 = 1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679
PI_1 = 8214808651328230664709384460955058223172535940812848111745028410270193852110555964462294895493038196

# First circulant rows found by the reference's circulant_mds_matrix() search,
# precomputed for faster initialization of large instances (l > 4).
CIRCULANT_MDS_ROWS = {
    5:  [1, 1, 3, 4, 5],
    6:  [1, 1, 3, 4, 5, 6],
    7:  [1, 2, 3, 5, 5, 6, 7],
    8:  [1, 2, 3, 5, 7, 8, 8, 9],
    9:  [1, 3, 5, 6, 8, 9, 9, 10, 11],
    10: [1, 2, 5, 6, 8, 11, 11, 12, 13, 14],
}

# Mx matrices
ANEMOI_Mx = {2: pht_matrix, 3: dl_m33_52_matrix, 4: dl_m46_83_matrix}

class AnemoiParams:
    def __init__(
        self,
        p:         int,
        g:         int,
        alpha:     int = None,
        alpha_inv: int = None,
        l:         int = None,
        t:         int = None,
        R:         int = None,
        M:         list[list[int]] = None,
        r:         int = None,
        c:         int = None,
        d:         int = None,
        kappa:     int = 128,
    ):
        """
        Parameters
        ----------
        p         : field characteristic (prime)
        g         : generator of the multiplicative group of GF(p); beta = g and delta = g^{-1}
        alpha     : Flystel exponent, coprime with p-1; smallest such exponent via _init_alpha if not provided
        alpha_inv : alpha^{-1} mod (p-1); computed via _init_alpha_inv if not provided
        l         : number of columns (parallel Flystel S-boxes); the state size is t = 2*l
        t         : state size (must be even); alternative to l, specify at least one of the two
        R         : number of rounds; derived via _init_rounds if not provided
        M         : lxl MDS matrix for the linear layer; generated via _init_mat if not provided
        r         : rate (number of outer state elements absorbed/squeezed per sponge step); derived if not provided
        c         : capacity (number of inner state elements); derived if not provided
        d         : digest size (number of output elements); derived if not provided
        kappa     : target security level in bits (default 128)
        """

        # Input sanitization
        AnemoiParams._input_sanitization(SimpleNamespace(**{k: v for k, v in locals().items() if k != "self"}))

        # General settings (reconcile the state size: l columns, t = 2*l branches)
        l = l if l is not None else t // 2
        self.p = p
        self.F = GF(p)
        self.l = l
        self.t = 2 * l
        self.kappa = kappa

        # Non-linear layer (open Flystel)
        self.alpha = alpha if alpha is not None else self._init_alpha()
        self.alpha_inv = alpha_inv if alpha_inv is not None else self._init_alpha_inv()
        self.g = self.to_field(g)
        self.QUAD = 3 if self.p == 2 else 2
        self.beta = self.g
        # NOTE paper sets delta=0 and gamma=g^-1 (see https://eprint.iacr.org/2022/840.pdf, page 10), but
        # all reference implementations https://github.com/anemoi-hash/anemoi-hash do the opposite
        #self.gamma = self.g ** (-1)
        #self.delta = self.F.zero()
        self.gamma = self.F.zero()
        self.delta = self.g ** (-1)

        # Hash modes
        self.r, self.c, self.d = derive_rate_capacity_digest(self.kappa, self.t, r, c, d)

        # Rounds
        self.R = R if R is not None else self._init_rounds()

        # Affine layer
        Mx = M if M is not None else self._init_mat()
        self.Mx = map_nested(Mx, self.to_field)
        self.Mx_inv = invert_matrix(self.Mx)

        self.My = self._init_My_from_Mx()
        self.My_inv = invert_matrix(self.My)

        self.C, self.D = self._init_cons()  # C (x-lane) and D (y-lane)

        # Parameter sanitization: validate the fully-constructed (stored/derived) values
        self._parameter_sanitization()

    # ---------------------------------------------------------------------------
    # Small field conversion helpers
    # ---------------------------------------------------------------------------

    def from_field(self, el) -> Integer:
        return Integer(el)

    def to_field(self, n: int):
        return self.F(n)

    # ---------------------------------------------------------------------------
    # Input sanitization and security requirements
    # ---------------------------------------------------------------------------

    @staticmethod
    def _input_sanitization(params):
        """Validate the raw constructor arguments: hard checks raise, recommendation
        deviations warn (ParamRecommendationWarning) but do not raise."""

        # --- Hard checks (must always hold) ---
        if params.p == 2:
            raise NotImplementedError("Characteristic 2 version of Anemoi not implemented")
        if params.l is None and params.t is None:
            raise ValueError("specify at least one of 'l' or 't'")
        if params.t is not None:
            if params.t % 2 != 0:
                raise ValueError(f"t must be even. Got {params.t}")
            if params.l is not None and params.t != 2 * params.l:
                raise ValueError(f"inconsistent 'l' and 't'. Got l={params.l} and t={params.t}, expected t = 2*l")
        if params.alpha is not None and gcd(params.alpha, params.p - 1) != 1:
            raise ValueError("power map does not define a permutation (gcd(alpha, p-1) != 1)")

        # --- Warnings (recommended, not required) ---
        field_bits = int(params.p).bit_length()
        if field_bits < 31:
            warnings.warn(f"TOY VERSION: field is only {field_bits} bits", ParamRecommendationWarning, stacklevel=2)

    def _parameter_sanitization(self):
        """Validate the fully-constructed parameter object (stored/derived values):
        hard checks raise, recommendation deviations warn (ParamRecommendationWarning)."""

        # --- Hard checks (must always hold) ---
        if len(self.Mx) != self.l or any(len(row) != self.l for row in self.Mx):
            raise ValueError(f"Mx must be a {self.l} x {self.l} matrix")
        if len(self.C) != self.R or len(self.D) != self.R:
            raise ValueError(f"C and D must have one row per round: expected {self.R}, got {len(self.C)} and {len(self.D)}")
        if any(len(row) != self.l for row in self.C) or any(len(row) != self.l for row in self.D):
            raise ValueError(f"each C and D row must hold {self.l} elements")

    # ---------------------------------------------------------------------------
    # Derivation helpers (defaults for the optional parameters)
    # ---------------------------------------------------------------------------

    def _init_alpha(self) -> int:
        for alpha in range(3, self.p):
            if gcd(alpha, self.p - 1) == 1:
                return alpha

    def _init_alpha_inv(self) -> int:
        return pow(self.alpha, -1, self.p - 1)

    def _R_charp(self) -> int:
        """Instance-specific number of rounds that can be attacked by a Groebner basis attack.
        The reference models R rounds with nv = 2*l*R variables at regularity degree
        2*l*R + kappa_alpha and demands C(dreg + nv, nv)^2 >= 2^kappa."""
        kappa_alpha = {3: 1, 5: 2, 7: 4, 9: 7, 11: 9}
        assert self.alpha in kappa_alpha
        nvar = lambda R: 2 * self.l * R
        dcon = lambda R: 2 * self.l * R + kappa_alpha[self.alpha]
        R = 1
        while gb_comp(dreg=dcon(R), nv=nvar(R), w=2) < self.kappa:
            R += 1
        R += 2  # security margin considering the P_CICO model
        return R

    def _R_char2(self) -> int:
        """Instance-specific number of rounds that can be attacked by a Groebner basis attack
        in the characteristic-2 setting."""
        raise NotImplementedError("Round number derivation for characteristic 2 version of Anemoi not implemented")

    def _init_rounds(self) -> int:
        """Round number derivation formula according to Equation (2) in https://eprint.iacr.org/2022/840.pdf"""
        R = self._R_char2() if self.p == 2 else self._R_charp()  # to prevent algebraic attacks
        R += min(5, self.l + 1)  # security margin
        return max(8, R)

    def _init_cons(self):
        """C and D are built from the digits of pi using an open butterfly."""
        C, D = [], []
        pi_F_0 = self.to_field(PI_0 % self.p)
        pi_F_1 = self.to_field(PI_1 % self.p)
        for r in range(self.R):
            pi_0_r = pi_F_0 ** r
            C.append([])
            D.append([])
            for i in range(self.l):
                pi_1_i = pi_F_1 ** i
                pow_alpha = (pi_0_r + pi_1_i) ** self.alpha
                C[r].append(self.beta * pi_0_r ** 2 + pow_alpha)
                D[r].append(self.beta * pi_1_i ** 2 + pow_alpha + self.delta)
        return C, D

    def _init_mat(self, max_tries: int = 1000):
        """Anemoi's M_x: Identity for l=1 (Anemoi's diffusion then comes from the PHT),
        low-addition matrices M_2/M_3/M_4 with the smallest working power of g for l <= 4,
        precomputed circulant rows for l > 4."""
        if self.l == 1:
            return [[self.F.one()]]
        if self.l <= 4:
            Mx_g = ANEMOI_Mx[self.l]
            gi = self.g
            for _ in range(max_tries):
                Mx = Mx_g(gi)
                if is_mds(Mx, self.F):
                    return Mx
                gi = gi * self.g
            raise RuntimeError(f"no MDS instance of the DL18 l={self.l} shape found within {max_tries} powers of g")
        if self.l in CIRCULANT_MDS_ROWS:
            return circulant(CIRCULANT_MDS_ROWS[self.l])
        raise NotImplementedError(f"MDS matrix generation not implemented for l={self.l}.")

    def _init_My_from_Mx(self):
        """Mx @ P_rho: column j of the result is column (j-1) mod l of Mx.
        Anemoi's M_y = M_x @ P_rho (apply-to-left-rotated-input as an explicit matrix),
        i.e. each row of Mx rotated right by one."""
        return [row[-1:] + row[:-1] for row in self.Mx]
