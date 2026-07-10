# params.py
# ---------------------------------------------------------------------------
# Parameter definition for Reinforced Concrete: the ReinforcedConcreteParams class.
#
# ReinforcedConcreteParams is the single source of truth for an instance. It
# sanitizes the user-facing parameters and expands them into a fully-specified
# instance that the permutation, hash modes, instances and tests consume. Any
# value the user omits is filled in by the matching _init_* helper (or, for
# r/c/d, by the shared derive_rate_capacity_digest). Settings that depart from
# the recommended ones raise a ParamRecommendationWarning rather than an error.
# ---------------------------------------------------------------------------

# Structural imports
from recommendations import ParamRecommendationWarning
from types import SimpleNamespace

# Math specific imports
import warnings
from math import gcd
from sage.all import GF, Integer, PolynomialRing, is_prime, previous_prime

# Custom imports
from utils.sampler import XOFFieldElementSampler
from utils.lut import invert_LUT, mixed_radix_decompose
from utils.matrix import map_nested, invert_matrix
from utils.mode import derive_rate_capacity_digest


class ReinforcedConcreteParams:
    def __init__(
        self,
        p:         int,
        t:         int,
        alpha:     int,
        si:        list[int],
        R_pre:     int = None,
        R_bars:    int = None,
        R_post:    int = None,
        LUT:       list[int] = None,
        COEFFS:    list[int] = None,
        M:         list[list[int]] = None,
        alpha_inv: int = None,
        rcons:     list[list[int]] = None,
        r:         int = None,
        c:         int = None,
        d:         int = None,
        kappa:     int = 128,
    ):
        """
        Parameters
        ----------
        p         : field characteristic (prime)
        t         : permutation state size
        alpha     : power-map exponent for the first state element in Bricks
        si        : bases for decompose/compose in the Bars layer
        R_pre     : number of Bricks+Concrete rounds before the Bars layer(s); derived via _init_rounds if not provided
        R_bars    : number of Bars+Concrete rounds in the middle; derived via _init_rounds if not provided
        R_post    : number of Bricks+Concrete rounds after the Bars layer(s); derived via _init_rounds if not provided
        LUT       : per-digit lookup table used in Bar; generated via _init_LUT if not provided
        COEFFS    : Bricks polynomial coefficients [a_coeffs, b_coeffs]; generated via _init_COEFFS if not provided
        M         : MDS matrix (txt); generated via _init_mat if not provided
        alpha_inv : alpha^{-1} mod (p-1); computed via _init_alpha_inv if not provided
        rcons     : (R+1)xt round constants; generated via _init_cons (SHAKE128) if not provided
        r         : rate (number of outer state elements absorbed/squeezed per sponge step); derived if not provided
        c         : capacity (number of inner state elements); derived if not provided
        d         : digest size (number of output elements); derived if not provided
        kappa     : target security level in bits (default 128)
        """

        # Input sanitization
        ReinforcedConcreteParams._input_sanitization(SimpleNamespace(**{k: v for k, v in locals().items() if k != "self"}))

        # General settings
        self.p = p
        self.F = GF(p)
        self.t = t
        self.kappa = kappa

        # Non-linear layers: Bricks
        self.alpha = alpha
        self.alpha_inv = alpha_inv if alpha_inv is not None else self._init_alpha_inv()
        COEFFS = COEFFS if COEFFS is not None else self._init_COEFFS()
        self.a_coeffs = map_nested(COEFFS[0], self.to_field)
        self.b_coeffs = map_nested(COEFFS[1], self.to_field)

        # Non-linear layers: Bars
        self.si = list(si)
        LUT = LUT if LUT is not None else self._init_LUT()
        self.LUT = self._pad_LUT(LUT, max(si))
        self.LUT_inv = invert_LUT(self.LUT)

        # Hash modes
        self.r, self.c, self.d = derive_rate_capacity_digest(self.kappa, self.t, r, c, d)

        # Rounds (set before _init_cons, whose derivation depends on R)
        if R_pre is None or R_bars is None or R_post is None:
            R_pre, R_bars, R_post = self._init_rounds()
        self.R_pre = R_pre
        self.R_bars = R_bars
        self.R_post = R_post
        self.R = R_pre + R_bars + R_post

        # Affine layer
        self.M = map_nested(M if M is not None else self._init_mat(), self.to_field)
        self.M_inv = invert_matrix(self.M)

        # Round constants
        self.rcons = map_nested(rcons if rcons is not None else self._init_cons(), self.to_field)

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
            raise NotImplementedError("Characteristic 2 not implemented")
        if params.t < 1:
            raise ValueError(f"state size t must be positive. Got {params.t}")
        if gcd(params.alpha, params.p - 1) != 1:
            raise ValueError("power map does not define a permutation (gcd(alpha, p-1) != 1)")
        if params.LUT is not None and len(params.LUT) > 0xFFFF:
            raise ValueError("LUT must fit in 16 bits (len(LUT) <= 0xFFFF)")
        if params.COEFFS is not None and not (len(params.COEFFS) == 2 and all(len(row) == params.t - 1 for row in params.COEFFS)):
            raise ValueError(f"COEFFS must be [a_coeffs, b_coeffs] each of length t-1 = {params.t - 1}")
        if params.M is not None and not (len(params.M) == params.t and all(len(row) == params.t for row in params.M)):
            raise ValueError(f"M must be a {params.t}x{params.t} matrix")

        # --- Warnings (recommended, not required) ---
        field_bits = int(params.p).bit_length()
        if field_bits < 31:
            warnings.warn(f"TOY VERSION: field is only {field_bits} bits", ParamRecommendationWarning, stacklevel=2)

    def _parameter_sanitization(self):
        """Validate the fully-constructed parameter object (stored/derived values):
        hard checks raise, recommendation deviations warn (ParamRecommendationWarning)."""

        # --- Hard checks (must always hold) ---
        if len(self.M) != self.t or any(len(row) != self.t for row in self.M):
            raise ValueError(f"M must be a {self.t} x {self.t} matrix")
        if len(self.rcons) != self.R + 1 or any(len(row) != self.t for row in self.rcons):
            raise ValueError(f"rcons must be an {self.R + 1} x {self.t} grid (one row per round plus the final one)")
        if len(self.a_coeffs) != self.t - 1 or len(self.b_coeffs) != self.t - 1:
            raise ValueError(f"Bricks coefficients must each have length t-1 = {self.t - 1}")
        if len(self.LUT) < max(self.si):
            raise ValueError(f"LUT must cover the largest base in si ({max(self.si)}); got {len(self.LUT)} entries")

    # ---------------------------------------------------------------------------
    # Derivation helpers (defaults for the optional parameters)
    # ---------------------------------------------------------------------------

    def _init_alpha_inv(self) -> int:
        return pow(self.alpha, -1, self.p - 1)

    def _init_COEFFS(self) -> list[list[int]]:
        """Generate Bricks coefficients (alpha_i, beta_i) for i = 1, ..., t-1.

        Bricks is invertible iff every quadratic z^2 + alpha_i*z + beta_i is non-zero
        for all z in F_p, i.e. the discriminant alpha_i^2 - 4*beta_i is a non-square
        in F_p (RC requires "alpha_i^2 - 4*beta_i is not a quadratic residue").

        We pick the smallest gap g >= 1 for which (alpha, beta) = (1, 1+g) is valid,
        then keep g fixed and scan alpha = 1, 2, 3, ..., collecting the first t-1 pairs
        (alpha, alpha + g) with non-square discriminant. This reproduces the published
        coefficients: gap 1 -> (1,2),(3,4) for BLS12-381 and BN254 (where -7 is a
        non-square), and gap 2 -> (1,3),(2,4) for the ST prime (where -7 is a square,
        so gap 1 is unavailable at the start).

        Returns COEFFS = [a_coeffs, b_coeffs] with a_coeffs = alphas, b_coeffs = betas.
        """
        def valid(a, b):
            return not self.F(a * a - 4 * b).is_square()

        g = 1
        while not valid(1, 1 + g):
            g += 1

        a_coeffs, b_coeffs = [], []
        alpha = 1
        while len(a_coeffs) < self.t - 1:
            beta = alpha + g
            if valid(alpha, beta):
                a_coeffs.append(alpha)
                b_coeffs.append(beta)
            alpha += 1
        return [a_coeffs, b_coeffs]

    def _lut_prime(self) -> int:
        """The prime p' on which the per-digit S-box acts.

        Each digit S-box S_i is a permutation of Z_{s_i} that applies the non-linear
        permutation f to inputs in {0, ..., p'-1} and the identity to the remaining
        inputs {p', ..., s_i - 1}. For this to be a permutation of every Z_{s_i}, p'
        must be <= every digit of (p - 1) and prime. Following RC Section 6.1, p' is
        the largest prime <= v, where  v = min_i v_i and (v_1, ..., v_n) = Decomp(p - 1).
        (For BLS12-381 p' = 659, for BN254 p' = 641, for the ST prime p' = 1013.)
        """
        v = mixed_radix_decompose(self.p - 1, self.si, self.from_field)
        v_min = int(min(v))
        return v_min if is_prime(v_min) else previous_prime(v_min)

    @staticmethod
    def _lut_exponent(p_prime: int) -> int:
        """Smallest d that is a prime of the form 2^n - 1 (a Mersenne prime) with
        gcd(d, p'-1) = 1. The coprimality makes x -> x^d a permutation of F_{p'}."""
        n = 2
        while True:
            d = (1 << n) - 1
            if is_prime(d) and gcd(d, p_prime - 1) == 1:
                return d
            n += 1

    @staticmethod
    def _lut_rounds(p_prime: int, d: int) -> int:
        """r = 2 * ceil(log_d(p')), the number of (X + c_i)^d steps composed into f."""
        k = 0
        while d ** k < p_prime:
            k += 1
        return 2 * k

    def _init_LUT(self, max_trials: int = 1000) -> list[int]:
        """Generate the S-box lookup table f(0), ..., f(p'-1) as described in RC App. A.3
        in https://eprint.iacr.org/2021/1038.pdf.

        f is the non-identity part of every per-digit S-box. It must be a permutation
        of F_{p'} with a high-degree, dense polynomial representation. It is built as a
        keyed-MiMC-style composition

            f(X) = (f_r ° f_{r-1} ° ... ° f_1)(X),    f_i(X) = (X + c_i)^d  in F_{p'}[X],

        with d the smallest Mersenne prime coprime to p'-1, r = 2*ceil(log_d(p')), and
        random constants c_i. We resample the c_i until f reaches the maximum degree
        p'-2 (permutation polynomials cannot reach p'-1) and full density of p'-1
        non-zero coefficients.

        Returns the unpadded table of length p'; _pad_LUT extends it with identity
        entries up to max(si).

        NOTE: The constants are drawn deterministically from a SHAKE-128 stream seeded from the
        instance, so the table is reproducible; it is NOT expected to match a specific published
        table, whose original (undisclosed) random constants differ. In particular, so concrete
        sampling method was described in the paper.
        """
        p_prime = self._lut_prime()
        Fp = GF(p_prime)
        d = self._lut_exponent(p_prime)
        r = self._lut_rounds(p_prime, d)

        R = PolynomialRing(Fp, "X")
        X = R.gen()
        reduction = X ** p_prime - X  # reduce to the degree-<p' function representative

        n_bytes = (self.p.bit_length() + 7) // 8
        seed = b"ReinforcedConcrete" + self.p.to_bytes(n_bytes, "little")
        sampler = XOFFieldElementSampler(seed=seed, p=p_prime, xof="shake_128", sampling="bitmask")

        for _ in range(max_trials):
            f = X
            for _ in range(r):
                c = sampler.next()
                f = ((f + c) ** d) % reduction

            if f.degree() == p_prime - 2 and len(f.coefficients()) == p_prime - 1:
                return [int(f(Fp(x))) for x in range(p_prime)]

        raise RuntimeError(f"Failed to find a dense, maximum-degree S-box polynomial for p'={p_prime} within {max_trials} trials.")

    def _init_cons(self) -> list[list[int]]:
        n_bytes = (self.p.bit_length() + 7) // 8
        seed = b"ReinforcedConcrete" + self.p.to_bytes(n_bytes, "little")
        return XOFFieldElementSampler(seed=seed, p=self.p, xof="shake_128", sampling="bitmask").grid(self.R + 1, self.t)

    def _init_rounds(self) -> tuple[int, int, int]:
        """Return the (R_pre, R_bars, R_post) round split.
        TODO: implement the round-number criterion of the Reinforced Concrete
        paper (https://eprint.iacr.org/2021/1038, Section 6); until then the
        split must be passed explicitly."""
        raise NotImplementedError("Error: Not implemented -- round number derivation for Reinforced Concrete")

    def _init_mat(self) -> list[list[int]]:
        """Return the t x t MDS matrix.
        TODO: implement the paper's circulant construction (M = circ(2, 1, 1) for
        t = 3, https://eprint.iacr.org/2021/1038 Section 5); until then M must be
        passed explicitly."""
        raise NotImplementedError("Error: Not implemented -- MDS matrix generation for Reinforced Concrete")

    @staticmethod
    def _pad_LUT(LUT: list[int], max_si: int) -> list[int]:
        out = list(LUT)
        for i in range(len(LUT), max_si):
            out.append(i)
        return out
