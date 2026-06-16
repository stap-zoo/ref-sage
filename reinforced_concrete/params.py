from sage.all import GF, Integer, PolynomialRing, is_prime, previous_prime

from utils import XOFFieldElementSampler, invert_LUT, map_to_field, invert_matrix

class ReinforcedConcreteParams:
    def __init__(
        self,
        p:         int,
        t:         int,
        R_pre:     int,
        R_bars:    int,
        R_post:    int,
        alpha:     int,
        si:        list[int],
        LUT:       list[int],
        COEFFS:    list[int],
        M:         list[list[int]],
        d:         int,
        r:         int = None,
        c:         int = None,
        alpha_inv: int = None,
        rcons:     list[list[int]] = None,
        kappa:     int = 128,
    ):
        """
        Parameters
        ----------
        p         : field characteristic (prime)
        t         : permutation state size
        R_pre     : number of Bricks+Concrete rounds before the Bars layer(s)
        R_bars    : number of Bars+Concrete rounds in the middle (after R_pre, before R_post)
        R_post    : number of Bricks+Concrete rounds after the Bars layer(s)
        alpha     : power-map exponent for the first state element in Bricks
        si        : bases for decompose/compose in the Bars layer
        LUT       : per-digit lookup table used in Bar (same LUT can be applied to all chunks due to padding in _pad_LUT)
        COEFFS    : Bricks polynomial coefficients
        M         : MDS matrix (txt)
        d         : digest size (number of output elements)
        r         : rate (number of outer state elements absorbed/squeezed per sponge step)
        c         : capacity (number of inner state elements)
        alpha_inv : alpha^{-1} mod (p-1); computed if not provided
        rcons     : Rxt round constants; generated via SHAKE128 if not provided
        kappa     : target security level in bits (default 128)
        """
        assert len(LUT) <= 0xFFFF
        assert len(COEFFS) == 2 and all(len(row) == t - 1 for row in COEFFS)
        assert len(M) == t and all(len(row) == t for row in M) if M is not None else True

        self.p = p
        self.F = GF(p)
        self.t = t
        self.kappa = kappa

        # Rounds
        if R_pre is None or R_bars is None or R_post is None:
            R_pre, R_bars, R_post = self._init_rounds()
        self.R_pre = R_pre
        self.R_bars = R_bars
        self.R_post = R_post
        self.R = R_pre + R_bars + R_post

        # Non-linear layers: Bricks
        self.alpha = alpha
        self.alpha_inv = alpha_inv if alpha_inv is not None else pow(alpha, -1, p - 1)
        COEFFS = COEFFS if COEFFS is not None else self._init_coeffs()
        self.a_coeffs = map_to_field(COEFFS[0], self.to_field)
        self.b_coeffs = map_to_field(COEFFS[1], self.to_field)

        # Non-linear layers: Bars
        self.si = list(si)
        LUT = LUT if LUT is not None else self._init_lut()
        self.LUT = self._pad_LUT(LUT, max(si)) if LUT is not None else self._pad_LUT(self._init_lut(), max(si))
        self.LUT_inv = invert_LUT(self.LUT)

        # Affine layer
        M = M if M is not None else self._init_mds()
        self.M = map_to_field(M, self.to_field)
        self.M_inv = invert_matrix(self.M)

        self.rcons = rcons if rcons is not None else self._init_rcons()
        self.rcons = map_to_field(self.rcons, self.to_field)

        # Hash modes
        self.r = r
        self.c = c
        self.d = d

    # ---------------------------------------------------------------------------
    # Small helpers
    # ---------------------------------------------------------------------------

    def from_field(self, el) -> Integer:
        return Integer(el)

    def to_field(self, n: int):
        return self.F(n)

    def _init_coeffs(self) -> list[list[int]]:
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
 
    def _init_lut(self, max_trials: int = 1000) -> list[int]:
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

    def _init_rcons(self) -> list[list[int]]:
        n_bytes = (self.p.bit_length() + 7) // 8
        seed = b"ReinforcedConcrete" + self.p.to_bytes(n_bytes, "little")
        return XOFFieldElementSampler(seed=seed, p=self.p, xof="shake_128", sampling="bitmask").grid(self.R + 1, self.t)
    
    def _init_rounds(self, R_pre, R_bars, R_post) -> (int,int,int):
        # TODO implement
        raise NotImplementedError("Automatic round number derivation not implemented for RC.")

    def _init_mds(self):
        # TODO implement
        raise NotImplementedError("MDS matrix generation not implemented for RC.")

    @staticmethod
    def _pad_LUT(LUT: list[int], max_si: int) -> list[int]:
        out = list(LUT)
        for i in range(len(LUT), max_si):
            out.append(i)
        return out

