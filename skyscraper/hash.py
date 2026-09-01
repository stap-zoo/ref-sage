# hash.py
# Skyscraper: SkyscraperPerm (permutation) and its mode functions (SkyscraperHash, SkyscraperCompress).
# NOTE: uses a lookup-table S-box -- non-generic (cannot run symbolically over a polynomial ring).

from skyscraper.params import SkyscraperParams
from utils.matrix import vecadd, vecsub
from utils.lut import mixed_radix_decompose, mixed_radix_compose
from utils.poly import eval_aos
from utils.primitive import Permutation, HashFunction, CompressionFunction


class SkyscraperPerm(Permutation):
    def __init__(self, params: SkyscraperParams):
        super().__init__(params)  # F, to_field, from_field, t, p, kappa, toy

        # Extension degree (Feistel state size t = 2*n base-field elements)
        self.n = params.n

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

    def permute(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        xL, xR = list(state[:self.n]), list(state[self.n:])
        for i in range(self.R):
            xL, xR = vecadd(xR, self.round_fun(xL, i)), xL
        return xL + xR

    def permute_inv(self, state: list) -> list:
        if len(state) != self.t:
            raise ValueError(f"Invalid state size. Expected {self.t}, got {len(state)}")

        xL, xR = list(state[self.n:]), list(state[:self.n]) # undo final twist
        for i in reversed(range(self.R)):
            xL, xR = vecsub(xR, self.round_fun(xL, i)), xL
        return xR + xL # undo last twist


# ---------------------------------------------------------------------------
# Hash / compression functions
#
# SkyscraperPerm above is JUST the permutation. Each mode wraps a permutation:
#     P = SkyscraperPerm(params)
#     H = SkyscraperHash(P, params.sponge)      # SAFE sponge
#     C = SkyscraperCompress(P, params.comp)    # 2n -> n Davies-Meyer (trunc_n(P(x)+x))
# The reference 2n -> n compression is exactly the truncation mode with arity 2
# (comp=dict(a=2) -> d = t/2 = n): trunc keeps the first n of P(x)+x, i.e. the left branch.
# ---------------------------------------------------------------------------

class SkyscraperHash(HashFunction):
    SPONGE_KIND = "safe"

class SkyscraperCompress(CompressionFunction):
    COMP_KIND = "trunc"       # 2n -> n Davies-Meyer over the left branch (comp=dict(a=2))
