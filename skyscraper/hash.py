# hash.py
# ---------------------------------------------------------------------------
# Skyscraper: the permutation (round function) and the hash modes built on it.
#
# Constructed from a fully-specified SkyscraperParams object.
#
# NOTE: the split-and-lookup S-box (_sl_sbox / _sl_sbox_inv, used by nonlinear_layer on
# the first u branches) decomposes a field element into bytes and applies a
# precomputed LUT. It is therefore inherently NON-generic -- it operates on the
# integer representation and cannot run symbolically over a polynomial ring,
# unlike the power-map branch of nonlinear_layer and the AffineLayer. This is a
# deliberate exception to the "generic component" contract.
# ---------------------------------------------------------------------------

from skyscraper.params import SkyscraperParams
from utils.matrix import vecadd, vecsub
from utils.lut import mixed_radix_decompose, mixed_radix_compose
from utils.poly import eval_aos
from utils.mode import compress_davies_meyer


class Skyscraper:
    def __init__(self, params: SkyscraperParams):
        self.F = params.F
        self.to_field = params.to_field
        self.from_field = params.from_field

        # Extension degree and Feistel state size (t = 2*n base-field elements)
        self.n = params.n
        self.t = params.t

        # Square layer
        self.cpolys = params.cpolys
        self.mont_R_inv = params.mont_R_inv

        # Bar layer
        self.si = params.si
        self.m = params.m
        self.rot = params.rot
        self.LUTs = params.LUTs

        # Round schedule and constants
        self.R = params.R
        self.bar_rounds = params.bar_rounds
        self.rcons = params.rcons

        # Hash modes
        self.sponge = params.sponge

    # ---------------------------------------------------------------------------
    # Round functions (each maps one branch of n coordinates to n coordinates)
    # ---------------------------------------------------------------------------

    def is_bar_round(self, r: int) -> bool:
        return r in self.bar_rounds
        
    def is_square_round(self, r: int) -> bool:
        return not self.is_bar_round(r)

    def _square(self, branch: list) -> list:
        """ x -> x^2 * mont_R_inv over GF(p^n): square via the coordinate polynomials,
        then scale every coordinate by the (base-field) Montgomery constant."""
        return [eval_aos(terms, branch) * self.mont_R_inv for terms in self.cpolys]

    def _bar(self, xL: list) -> list:
        """Split-and-lookup S-box (non-generic; see module note):
        Decompose every (extension) field element into n*m digits, apply the LUT to each chunk, 
        cyclically rotate left (over the whole element) by rot, then recompose."""

        def flatten(digits_by_coord: list[list[int]]) -> list[int]:
            """Flatten a list of coordinate digit lists (nxm) into one digit list (n*m)."""
            return [digit for coord_digits in digits_by_coord for digit in coord_digits]
        
        def unflatten(digits_flat: list[int]) -> list[list[int]]:
            """Split a flat digit list (n*m) into a list of coordinate digit lists (nxm)."""
            return [digits_flat[i * self.m:(i + 1) * self.m] for i in range(self.n)]
        
        # Transform each coordinate of the (extension) field elements to its digit representation 
        digits_by_coord = [mixed_radix_decompose(coord, self.si, self.from_field) for coord in xL]
        
        # Apply LUT to each chunk
        digits_by_coord = [[self.LUTs[si][d] for si, d in zip(self.si, coord_digits)] for coord_digits in digits_by_coord]
        
        # Apply rotation to flattened n*m-digit state
        digits_flat = flatten(digits_by_coord)
        digits_flat = digits_flat[self.rot:] + digits_flat[:self.rot]
        digits_by_coord = unflatten(digits_flat)
        
        return [mixed_radix_compose(coord_digits, self.si, self.to_field) for coord_digits in digits_by_coord]

    def round_fun(self, xL: list, r: int) -> list:
        y = self._square(xL) if self.is_square_round(r) else self._bar(xL)
        return vecadd(y, self.rcons[r]) # addition over extension field is component-wise

    # ---------------------------------------------------------------------------
    # Permutation
    # ---------------------------------------------------------------------------

    def permutation(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        xL, xR = list(state[:self.n]), list(state[self.n:])
        for i in range(self.R):
            xL, xR = vecadd(xR, self.round_fun(xL, i)), xL
        return xL + xR

    def permutation_inv(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        xL, xR = list(state[self.n:]), list(state[:self.n]) # undo final twist
        for i in reversed(range(self.R)):
            xL, xR = vecsub(xR, self.round_fun(xL, i)), xL
        return xR + xL # undo last twist

    # ---------------------------------------------------------------------------
    # Hash modes
    # ---------------------------------------------------------------------------

    def compress(self, message: list) -> list:
        """Reference 2n -> n compression: the Davies-Meyer feed-forward perm(in_L || in_R)_L + in_L over the left branch."""
        if len(message) != self.t:
            raise ValueError(f"compress expects {self.t} elements, got {len(message)}")
        return compress_davies_meyer(
            perm=self.permutation,
            x_m=list(message[:self.n]),
            x_c=list(message[self.n:]),
            digest_size=self.n,
            to_field=self.to_field,
        )

    def compress_2_to_1(self, x1: list, x2: list) -> list:
        """Merge two n-element digests into one (Merkle node), via compress."""
        if len(x1) != self.n or len(x2) != self.n:
            raise ValueError(f"Inputs must be digests of {self.n} elements, got {len(x1)} and {len(x2)}.")
        return self.compress(list(x1) + list(x2))

    def hash_sponge(self, data: list) -> list:
        return self.sponge.hash(self.permutation, data)
