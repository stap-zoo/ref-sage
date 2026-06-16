from sage.all import GF, Integer

from utils import (
    LFSRFieldElementSampler,
    XOFFieldElementSampler,
    cauchy_mds_matrix,
    circulant,
    m4_to_block_circulant_matrix,
    dl_m44_84_matrix,
    ones_plus_diag_matrix,
    map_to_field,
    invert_matrix,
)

# ---------------------------------------------------------------------------
# Grain LFSR settings (used by Poseidon/Poseidon2)
# ---------------------------------------------------------------------------

# The 80-bit Grain LFSR used by Poseidon's generate_parameters_grain.sage (Appendix E of
# https://eprint.iacr.org/2019/458): an 80-bit register with these feedback taps. (The warm-up
# defaults to 2*state_size = 160, matching the reference.)
GRAIN_STATE_SIZE = 80
GRAIN_TAPS = [0, 13, 23, 38, 51, 62]

# A seed is a nothing-up-my-sleeve encoding of the instance. Each field has a *meaning*
# (params -> int, the reference semantics); layouts choose widths and which meanings to carry.
def _const(v):      return lambda P: v
def _field_code(P): return 2 if P.F.characteristic() == 2 else 1
def _sbox(P):       return 1 if P.alpha == -1 else 0        # reference rule: 1 = inverse map, 0 = power map
def _alpha(P):      return 0 if P.alpha == -1 else P.alpha  # added by Ethereum 
def _n(P):          return P.p.bit_length()
def _t(P):          return P.t
def _r_ext(P):      return P.R_ext
def _r_int(P):      return P.R_int

# Layouts: ordered (name, width, meaning). The only irreducible per-layout constant is the
# `shrink` refers to whether the GrainLFSR is self-shrinking or not
_ORIGINAL = {"shrink": True, "fields": [("field", 2, _field_code), ("sbox", 4, _sbox), ("n", 12, _n), ("t", 12, _t), ("R_F", 10, _r_ext), ("R_P", 10, _r_int)]}
_ETHEREUM = {"shrink": False, "fields": [("field", 2,  _const(2)), ("type", 1, _sbox), ("alpha", 5, _alpha), ("n", 10, _n), ("t", 10, _t), ("R_F", 10, _r_ext), ("R_P", 10, _r_int)]}

# Versions: a layout + any deviation from the reference meanings. The _ORIGINAL version
# references https://extgit.isec.tugraz.at/krypto/hadeshash, following the description in the Poseidon paper.
# NOTE: isec/horizenlabs force the s-box marker to 1 even for power maps.
POSEIDON_SEEDING_VERSIONS = {
    "circom":      {"layout": _ORIGINAL},
    "isec":        {"layout": _ORIGINAL, "deviation": {"sbox": _const(1)}},
    "horizenlabs": {"layout": _ORIGINAL, "deviation": {"sbox": _const(1)}},
    "ethereum":    {"layout": _ETHEREUM},
}

# ---------------------------------------------------------------------------
# Hades base parameters
# ---------------------------------------------------------------------------

class HadesParams:
    """Common parameters for Hades-strategy permutations (Poseidon, Poseidon2, Neptune):
    a state of t branches, an S-box of degree alpha, and the round structure.

        external (full) rounds -> internal (partial) rounds -> external (full) rounds

    with R_ext external and R_int internal rounds. The S-box hits all t branches in external
    rounds and only the first `u` branches in internal rounds.

    Each round is ARK -> S-box -> matrix (round constant added *before* the S-box); the round
    loop just indexes `rcons[round_idx]`. The work done once outside the loop -- a leading
    external matrix, output whitening -- lives in the permutation's _pre_rounds / _post_rounds
    (see hades/hash.py). Poseidon/Poseidon2 are ARK-before-S by nature, so `rcons` is exactly
    their published grid; Neptune is S-before-ARK, so it carries a leading zero row in `rcons`
    (its round 0 adds nothing before the S-box) and applies its final constant in _post_rounds.
    There is therefore no rc_init / rc_rounds split.

    Subclasses derive their matrices and round constants from a deterministic sampler via the
    _init_sampler / _init_rcons / _init_M_ext / _init_M_int hooks (any of which may instead be
    supplied explicitly). Matrices and round constants are given as plain integers (or field
    elements) and stored as field elements.
    """

    def __init__(
        self,
        *,
        p: int,
        t: int,
        alpha: int,
        R_ext: int,
        R_int: int,
        r: int,
        c: int,
        d: int,
        version: str = "isec",
        M_ext: list[list] = None,
        M_int: list[list] = None,
        rcons: list[list] = None,
        R_ext_beg: int = None,
        R_ext_end: int = None,
        u: int = 1,
        kappa: int = 128,
    ):
        """
        Parameters
        ----------
        p               : field characteristic (prime)
        t               : permutation state size (branches)
        alpha           : exponent of the power-map S-box (its degree)
        R_ext           : number of external (full) rounds
        R_int           : number of internal (partial) rounds
        r               : rate (number of outer state elements absorbed/squeezed per sponge step)
        c               : capacity (number of inner state elements)
        d               : digest size (number of output elements)
        version         : round-constant / MDS construction strategy (see POSEIDON_VERSIONS)
        M_ext           : external-round matrix (txt); derived via _init_M_ext if not given
        M_int           : internal-round matrix (txt); derived via _init_M_int if not given
        rcons           : per-round constant grid the round loop indexes; derived via _init_rcons if not given
        R_ext_beg       : number of external rounds before the internal rounds; default R_ext // 2
        R_ext_end       : number of external rounds after the internal rounds; default R_ext - R_ext_beg
        u               : number of branches the S-box hits in internal rounds; default 1
        kappa           : target security level in bits (default 128)
        """
        self.p = p
        self.F = GF(p)
        self.t = t
        self.kappa = kappa

        # Rounds
        if R_ext is None or R_int is None:
            R_ext, R_int = self._init_rounds(R_ext, R_int)
        self.R_ext = R_ext
        self.R_int = R_int
        self.R = R_ext + R_int
        self.R_ext_beg = R_ext_beg if R_ext_beg is not None else R_ext // 2
        self.R_ext_end = R_ext_end if R_ext_end is not None else R_ext - self.R_ext_beg

        # Non-linear layer
        self.alpha = alpha
        self.alpha_inv = pow(alpha, -1, p - 1)
        self.u = u

        # Deterministic sampler (the _init_* derivations below consume its stream in order)
        self.version = version
        self.sampler = self._init_sampler()

        # Round constants are drawn first (Poseidon's Grain MDS continues the same stream),
        # then the matrices. `rcons` is the engine grid the round loop indexes (>= R rows;
        # Neptune adds one for its trailing whitening constant).
        self.rcons = map_to_field(rcons if rcons is not None else self._init_rcons(), self.to_field)
        if len(self.rcons) < self.R:
            raise ValueError(f"Expected at least {self.R} round-constant rows, got {len(self.rcons)}")
        self.M_ext = map_to_field(M_ext if M_ext is not None else self._init_M_ext(), self.to_field)
        self.M_int = map_to_field(M_int if M_int is not None else self._init_M_int(), self.to_field)
        self.M_ext_inv = invert_matrix(self.M_ext)
        self.M_int_inv = invert_matrix(self.M_int)

        # Hash modes (required; supplied per instance)
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

    def _init_rounds(self, R_ext, R_int):
        raise NotImplementedError("Virtual function, implement in derived class.")

    def _init_M_ext(self):
        raise NotImplementedError("Virtual function, implement in derived class.")

    def _init_M_int(self):
        raise NotImplementedError("Virtual function, implement in derived class.")

    def _init_rcons(self):
        raise NotImplementedError("Virtual function, implement in derived class.")

    def _init_sampler(self):
        """Build the Grain LFSR for a Poseidon `version` (see POSEIDON_SEEDING_VERSIONS): 
        assemble the primitive-specific 80-bit seed from the instance parameters and select 
        the bit-extraction mode (self-shrinking or raw). """
        try:
            spec = POSEIDON_SEEDING_VERSIONS[self.version]
        except KeyError:
            raise ValueError(f"Unknown Poseidon version {self.version}. Use one of: {', '.join(POSEIDON_VERSIONS)}.")
        
        to_bits   = LFSRFieldElementSampler.to_bits
        layout    = spec["layout"]
        deviation = spec.get("deviation", {})

        seed = []
        for name, width, meaning in layout["fields"]:
            value = deviation.get(name, meaning)(self)   # deviation overrides the reference meaning
            assert 0 <= value < (1 << width), f"{self.version}.{name}={value} exceeds {width} bits"
            seed += to_bits(value, width)
        assert len(seed) <= GRAIN_STATE_SIZE, f"seed {len(seed)} exceeds {GRAIN_STATE_SIZE}"
        seed += [1] * (GRAIN_STATE_SIZE - len(seed)) # padding

        return LFSRFieldElementSampler(seed_bits=seed, p=self.p, taps=GRAIN_TAPS, state_size=GRAIN_STATE_SIZE,
                                    sampling="bitmask", shrink=layout["shrink"])

# ---------------------------------------------------------------------------
# Poseidon
# ---------------------------------------------------------------------------

class PoseidonParams(HadesParams):
    """Poseidon (https://eprint.iacr.org/2019/458): one MDS matrix for both external and
    internal rounds, full-width round constants added before every S-box, no leading matrix.
    The round constants come from the Grain LFSR seeded per `version` (see POSEIDON_VERSIONS);
    the MDS is a Cauchy matrix whose `mds_strategy` is either "sampled" (xs/ys drawn from the
    same Grain stream -- isec/circom) or "fixed" (the deterministic 1/(i-t-j) indices --
    khovratovich/ethereum). Both are overridable by supplying `rcons` / `M` directly."""

    MDS_STRATEGIES = ("sampled", "fixed")

    def __init__(self, *, p, t, alpha, R_ext, R_int, r, c, d,
                 R_ext_beg=None, R_ext_end=None, version="isec", mds_strategy="sampled",
                 M=None, rcons=None, u=1, kappa=128):
        if mds_strategy not in self.MDS_STRATEGIES:
            raise ValueError(f"Unknown mds_strategy {mds_strategy!r}. Use one of {self.MDS_STRATEGIES}.")
        self.mds_strategy = mds_strategy
        super().__init__(p=p, t=t, alpha=alpha, R_ext=R_ext, R_int=R_int, r=r, c=c, d=d, version=version,
                         M_ext=M, M_int=M, rcons=rcons,
                         R_ext_beg=R_ext_beg, R_ext_end=R_ext_end, u=u, kappa=kappa)
        self.M = self.M_ext  # alias (single MDS)

    def _init_rcons(self):
        # Full-width round constants for every round (rejection-sampled from the Grain stream).
        return self.sampler.grid(self.R, self.t)

    def _init_M_ext(self):
        if self.mds_strategy == "fixed":
            # fixed 1/(i-t-j) (khovratovich / "ethereum")
            return cauchy_mds_matrix(self.p, self.t)          
        elif self.mds_strategy == "sampled":
            # draw the Cauchy xs/ys from the Grain stream *after* the round constants. The
            # reference switches from rejection sampling (round constants) to mod-reduction here
            # (create_mds_p), so do the same on the shared sampler before drawing the MDS material.
            self.sampler.set_sampling("mod")
            return cauchy_mds_matrix(self.p, self.t, sampler=self.sampler)
        else:
            raise NotImplementedError(f"Unknown matrix generation strategy {self.mds_strategy}.") 

    def _init_M_int(self):
        return self.M_ext  # single MDS for both external and internal rounds


# ---------------------------------------------------------------------------
# Poseidon2
# ---------------------------------------------------------------------------

class Poseidon2Params(HadesParams):
    """Poseidon2 (https://eprint.iacr.org/2023/323): Distinct external matrix M_E (leading + full rounds) 
    and internal matrix M_I = J + diag(mat_diag) (partial rounds); round constants added to all branches 
    in external rounds and only the first u branches in internal rounds.
    mat_diag is the MAT_DIAG_M_1 vector (defaulted for t in {2,3}, supplied per instance otherwise)."""

    def __init__(self, *, p, t, alpha, R_ext, R_int, r, c, d,
                 R_ext_beg=None, R_ext_end=None, version="isec",
                 M_ext=None, rcons=None, mat_diag=None, u=1, kappa=128):
        self.mat_diag = mat_diag
        super().__init__(p=p, t=t, alpha=alpha, R_ext=R_ext, R_int=R_int, r=r, c=c, d=d, version=version,
                         rcons=rcons, M_ext=M_ext, M_int=None, # M_int is always J + diag(mat_diag)
                         R_ext_beg=R_ext_beg, R_ext_end=R_ext_end, u=u, kappa=kappa)

    def _init_rcons(self):
        """Grid R x t: external rounds draw t constants, internal rounds draw u constants
        (placed on the first u branches, the rest zero), matching the HorizenLabs reference for u=1."""
        rc = []
        for i in range(self.R):
            if self.R_ext_beg <= i < self.R_ext_beg + self.R_int:    # internal round
                rc.append([self.sampler.next() for _ in range(self.u)] + [0] * (self.t - self.u))
            else:                                                    # external round
                rc.append([self.sampler.next() for _ in range(self.t)])
        return rc

    def _init_M_ext(self):
        if self.t == 2:
            return circulant(row=[2, 1])
        if self.t == 3:
            return circulant(row=[2, 1, 1])
        if self.t % 4 == 0:
            M4 = dl_m44_84_matrix(alpha=2)
            if self.t == 4:
                return [[2 * x for x in row] for row in M4]
            return m4_to_block_circulant_matrix(t=self.t, M4=M4)
        raise ValueError("Poseidon2 state size must be 2, 3, or a multiple of 4")

    def _init_M_int(self):
        # M_I = J + diag(mat_diag). The MAT_DIAG_M_1 diagonals for the small state sizes are fixed
        # by the spec; larger sizes are field-specific and must be supplied per instance.
        if self.mat_diag is None:
            if self.t == 2:
                self.mat_diag = [1, 2]
            elif self.t == 3:
                self.mat_diag = [1, 1, 2]
            else:
                # TODO implement like in Neptune?
                raise ValueError(f"mat_diag (MAT_DIAG_M_1) required for Poseidon2 with t={self.t}")
        return ones_plus_diag_matrix(self.mat_diag)


# ---------------------------------------------------------------------------
# Neptune
# ---------------------------------------------------------------------------

class NeptuneParams(HadesParams):
    """Neptune (https://eprint.iacr.org/2021/1695.pdf): External rounds use a quadratic pair-wise S-box (Lai-Massey-like construction) 
    and a split external matrix. The internal rounds use the degree-alpha power map on branch 0 with M_I = J + diag(mu). 
    All constants/matrices are derived deterministically via SHAKE128, as in the IAIK zk-friendly-hash-zoo reference 
    (https://extgit.isec.tugraz.at/krypto/zkfriendlyhashzoo).

    Round order is S -> M -> ARK. Expressed in Hades's ARK -> S -> M form, this means a
    leading zero row in `rcons` (round 0 adds no constant before its S-box) and the final
    constant applied as output whitening in the permutation's _post_rounds; the leading external
    matrix is applied by _pre_rounds."""

    def __init__(self, *, p, t, alpha, R_ext, R_int, r, c, d,
                 R_ext_beg=None, R_ext_end=None, M_ext=None, rcons=None, mat_diag=None, u=1, kappa=128):
        if t % 2 != 0:
            raise ValueError("Neptune state size t must be even")
        self.mat_diag = mat_diag
        super().__init__(p=p, t=t, alpha=alpha, R_ext=R_ext, R_int=R_int, r=r, c=c, d=d,
                         M_ext=M_ext, rcons=rcons,           # M_int is always J + diag(mat_diag)
                         R_ext_beg=R_ext_beg, R_ext_end=R_ext_end, u=u, kappa=kappa)
        # gamma is the next nonzero SHAKE draw after the internal-matrix diagonal mu.
        # Lai-Massey parameters
        self.lm_alpha = self.to_field(1)        # Lai-Massey multiplier; = 1 in the Neptune spec
        self.lm_alpha_inv = pow(self.lm_alpha, -1, p - 1)
        self.lm_beta = self.to_field(1)
        self.lm_gamma = self.to_field(self.sampler.next_nonzero())
        self.lm_M = map_to_field([[2,1],[1,3]], self.to_field)
        self.lm_M_inv = invert_matrix(self.lm_M)

    def _init_sampler(self):
        n_bytes = ((self.p.bit_length() + 63) // 64) * 8
        seed = b"Neptune" + self.p.to_bytes(n_bytes, "little")
        sampler = XOFFieldElementSampler(seed=seed, p=self.p, xof="shake_128", sampling="bitmask")
        return sampler

    def _init_rcons(self):
        # Leading zero row + R full-width rows: round 0 adds nothing before its S-box, and the
        # final row is applied by _post_rounds (output whitening). The R drawn rows match the
        # reference's full-width round constants; the zero row is not drawn from the stream.
        return [[0] * self.t] + self.sampler.grid(self.R, self.t)

    def _init_M_ext(self):
        """Even/odd "split" matrix M with M[2r][2c] = M'[r][c] and M[2r+1][2c+1] = M''[r][c]
        (all other entries zero). M', M'' are fixed circulants for t in {4,8} and otherwise
        sampled from the SHAKE stream (M' then M'')."""
        t, half = self.t, self.t // 2
        if t == 4:
            Mp, Mpp = circulant(row=[2, 1]), circulant(row=[1, 2])
        elif t == 8:
            Mp, Mpp = circulant(row=[3, 2, 1, 1]), circulant(row=[1, 1, 2, 3])
        else:
            Mp = self.sampler.grid(half, half)
            Mpp = self.sampler.grid(half, half)
        M = [[0] * t for _ in range(t)]
        for rr in range(half):
            for col in range(half):
                M[2 * rr][2 * col] = Mp[rr][col]
                M[2 * rr + 1][2 * col + 1] = Mpp[rr][col]
        return M

    def _init_M_int(self):
        # M_I = J + diag(mat_diag). When not supplied, draw the diagonal mu (nonzero) from SHAKE;
        # mat_diag = mu - 1 so the resulting diagonal is mu (off-diagonal entries 1).
        if self.mat_diag is None:
            mu = [self.sampler.next_nonzero() for _ in range(self.t)]
            self.mat_diag = [m - 1 for m in mu]
        return ones_plus_diag_matrix(self.mat_diag)
