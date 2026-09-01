# params.py
# AnemoiParams: the fully-specified parameter set for Anemoi (single source of truth per instance).

# Structural imports
from recommendations import recommend
from types import SimpleNamespace

# Math specific imports
from math import gcd
from sage.all import GF, Integer

# Custom imports
from utils.complexities import gb_comp
from utils.matrix import circulant, circulant_mds_matrix, is_mds, pht_matrix, dl_m33_52_matrix, dl_m46_83_matrix, map_nested, invert_matrix

# Digits of pi, used to derive the round constants via an open butterfly.
PI_0 = 1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679
PI_1 = 8214808651328230664709384460955058223172535940812848111745028410270193852110555964462294895493038196

# First circulant rows found by utils.matrix.circulant_mds_matrix() (the search ported
# from the Anemoi reference), precomputed for faster initialization of large instances
# (l > 4). Pinned here as a cache -- reproduced exactly by circulant_mds_matrix(l)[0].
# Entries are small integers, so these are MDS over every field used by these primitives; 
# _init_mat falls back to a live search for any l not listed.
CIRCULANT_MDS_ROWS = {
    5:  [1, 1, 3, 4, 5],
    6:  [1, 1, 3, 4, 5, 6],
    7:  [1, 2, 3, 5, 5, 6, 7],
    8:  [1, 2, 3, 5, 7, 8, 8, 9],
    9:  [1, 3, 5, 6, 8, 9, 9, 10, 11],
    10: [1, 2, 5, 6, 8, 11, 11, 12, 13, 14],
    11: [1, 2, 6, 7, 9, 12, 13, 14, 14, 16, 17],
    12: [1, 3, 4, 8, 9, 11, 14, 14, 17, 18, 19, 20],
}

# Mx matrices
ANEMOI_Mx = {2: pht_matrix, 3: dl_m33_52_matrix, 4: dl_m46_83_matrix}

class AnemoiParams:
    """Single instance spec for Anemoi, read by the permutation (AnemoiPerm) and the mode
    functions (AnemoiHash / AnemoiPiHash / AnemoiCompress). Carries the permutation parameters
    plus the per-mode parameter dicts `sponge` and `comp`; the permutation ignores them."""

    def __init__(
        self,
        p:         int,
        g:         int = None,
        alpha:     int = None,
        alpha_inv: int = None,
        l:         int = None,
        t:         int = None,
        R:         int = None,
        M:         list[list[int]] = None,
        # Modes of operation: per-mode parameter dicts, or None if the instance defines no
        # such mode. sponge = dict(r=.., c=.., d=..); comp = dict(a=..) (Jive_a) / ...
        sponge:    dict = None,
        comp:      dict = None,
        # Target security level (default 128 bits)
        kappa:     int = 128,
        # Toy instance switch: warns instead of raising on recommendation-level checks
        toy:       bool = False,
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
        sponge    : sponge params dict dict(r, c, d), or None for no sponge
        comp      : compression params dict (e.g. dict(a=2) for Jive_2), or None
        kappa     : target security level in bits (default 128)
        toy       : if True, recommendation-level checks warn instead of raising (default False)
        """

        # Input sanitization
        self._input_sanitization(SimpleNamespace(**{k: v for k, v in locals().items() if k != "self"}))

        # General settings (reconcile the state size: l columns, t = 2*l branches)
        self.p = p
        self.F = GF(p)
        self.t = t if t is not None else 2 * l # _input_sanitization checks that at least one of l or t is provided
        self.l = l if l is not None else self.t // 2
        self.kappa = kappa
        self.toy = toy

        # Modes of operation: per-mode param dicts (consumed by the mode functions, not the
        # permutation). None means the instance does not define that mode.
        self.sponge, self.comp = sponge, comp

        # Non-linear layer (open Flystel)
        self.alpha = alpha if alpha is not None else self._init_alpha()
        self.alpha_inv = alpha_inv if alpha_inv is not None else self._init_alpha_inv()
        self.g = self.to_field(g) if g is not None else self.F.multiplicative_generator()
        self.QUAD = 3 if self.p == 2 else 2
        self.beta = self.g
        # NOTE paper sets delta=0 and gamma=g^-1 (see https://eprint.iacr.org/2022/840.pdf, page 10), but
        # all reference implementations https://github.com/anemoi-hash/anemoi-hash do the opposite
        #self.gamma = self.g ** (-1)
        #self.delta = self.F.zero()
        self.gamma = self.F.zero()
        self.delta = self.g ** (-1)

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
            recommend(f"TOY VERSION: field is only {field_bits} bits", params.toy)

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
        and a circulant MDS matrix for l > 4 -- from the precomputed CIRCULANT_MDS_ROWS
        when listed, otherwise generated live via circulant_mds_matrix()."""
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
        # l > 4: circulant MDS matrix. Use the precomputed row when available, otherwise
        # search for it live (slower, but no l is left unsupported).
        row = CIRCULANT_MDS_ROWS.get(self.l)
        if row is None:
            row = circulant_mds_matrix(self.l)[0]
        return circulant(row)

    def _init_My_from_Mx(self):
        """Mx @ P_rho: column j of the result is column (j-1) mod l of Mx.
        Anemoi's M_y = M_x @ P_rho (apply-to-left-rotated-input as an explicit matrix),
        i.e. each row of Mx rotated right by one."""
        return [row[-1:] + row[:-1] for row in self.Mx]
