"""Polynomial representation and coordinate polynomials of power maps over (extension) fields.

Representations of a multivariate polynomial:
  AoS  (array of structs):  list[tuple[coeff, exponent_tuple]]
          e.g. [(1, (5,0,0)), (3, (2,2,1))]
  SoA  (struct of arrays):  tuple[list[coeff], list[exponent_tuple]]
          e.g. ([1, 3], [(5,0,0), (2,2,1)])

Exponents are tuples of ints. Coefficient type is controlled by the caller via
a `from_field` callable (applied on the way out of Sage) and a `to_field`
callable (applied on the way back in); both default to identity.
A "point" for evaluation is a sequence whose length matches the arity; entries
may be ints, field elements, or symbolic / polynomial-ring elements.
"""

from sage.all import GF, PolynomialRing, gcd, ZZ, prod
import warnings


# ---------------------------------------------------------------------------
# permutation test
# ---------------------------------------------------------------------------

def is_power_permutation(alpha, f, p):
    """True iff x^alpha is a permutation of F_p[x]/f.

    For f irreducible of degree n this is gcd(alpha, p^n - 1) == 1; the
    per-factor product also covers the squarefree reducible case.
    """
    assert f.is_squarefree()
    return all(gcd(alpha, p**h.degree() - 1) == 1 for h, _e in f.factor())

# ---------------------------------------------------------------------------
# single-polynomial format conversions
# ---------------------------------------------------------------------------

def _normalize_exp(e):
    """Exponent vector (tuple / ETuple / list) -> tuple[int]."""
    return tuple(int(k) for k in e)

def univ_from_list(x, L):
    """reconstruct sum_i L[i] * x^i from a coefficient list L (low -> high)."""
    return sum(xi * x**i for i, xi in enumerate(L))

def poly_to_aos(poly):
    """Sage multivariate polynomial -> AoS: list[(coeff, exps)]."""
    return [(c, _normalize_exp(e)) for c, e in zip(poly.coefficients(), poly.exponents())]

def poly_to_soa(poly):
    """Sage multivariate polynomial -> SoA: (coeffs, exps)."""
    return poly.coefficients(), [_normalize_exp(e) for e in poly.exponents()]

def aos_to_poly(terms, ring=None):
    """AoS -> Sage polynomial.

    The coefficient parent is the source of truth for the base ring: it is read
    from the coefficients themselves. If `ring` is given it is reused, provided
    it has enough generators and its base ring matches the coefficients'
    parent; otherwise a fresh ring over that parent is built.
    """
    if not terms:
        # nothing to infer a base from; fall back to the ring or a trivial default
        return (ring if ring is not None else PolynomialRing(ZZ, 'x0')).zero()

    arity = len(terms[0][1])
    base = terms[0][0].parent()        # base ring inferred from the coefficients

    if ring is not None:
        if ring.base_ring() != base:
            raise ValueError(f"ring base {ring.base_ring()} != coefficient parent {base}")
        if ring.ngens() < arity:
            raise ValueError(f"ring has {ring.ngens()} generators, need {arity}")
        R = ring
    else:
        R = PolynomialRing(base, ['x%d' % i for i in range(arity)])

    return sum((c * R.monomial(*e) for c, e in terms), R.zero())

def soa_to_poly(coeffs, exps, ring=None):
    """SoA -> Sage polynomial. See aos_to_poly."""
    return aos_to_poly(list(zip(coeffs, exps)), ring=ring)

def is_soa(poly):
    """True if `poly` is in SoA form: a (coeffs, exps) pair of equal-length sequences."""
    return (isinstance(poly, tuple) and len(poly) == 2 and hasattr(poly[0], '__len__') and hasattr(poly[1], '__len__'))

def is_aos(poly):
    """True if `poly` is in AoS form: a non-empty list of (coeff, exponent-sequence) pairs."""
    return (isinstance(poly, list) and poly and isinstance(poly[0], tuple) and len(poly[0]) == 2 and hasattr(poly[0][1], '__len__'))

def is_sage(poly):
    """True if `poly` is a Sage polynomial (has the coefficients()/exponents() interface)."""
    return hasattr(poly, 'coefficients') and hasattr(poly, 'exponents')

def to_term_dict(poly):
    if is_soa(poly):
        coeffs, exps = poly
        return {_normalize_exp(e): c for c, e in zip(coeffs, exps)}
    if is_aos(poly):
        return {_normalize_exp(e): c for c, e in poly}
    if is_sage(poly):
        return {_normalize_exp(e): c for c, e in zip(poly.coefficients(), poly.exponents())}
    raise TypeError(f"unsupported polynomial representation: {type(poly)}")

def map_coeffs(poly, fn):
    if is_soa(poly):
        coeffs, exps = poly
        return [fn(c) for c in coeffs], exps
    if is_aos(poly):
        return [(fn(c), e) for c, e in poly]
    raise TypeError(f"unsupported representation: {type(poly)}")

# ---------------------------------------------------------------------------
# evaluation (arithmetic stays in the point's own ring)
# ---------------------------------------------------------------------------

def eval_aos(terms, point):
    """Evaluate an AoS polynomial at `point` (length = arity)."""
    return sum(c * prod(v**k for v, k in zip(point, e) if k) for c, e in terms)

def eval_soa(coeffs, exps, point):
    """Evaluate a SoA polynomial at `point`. See eval_aos."""
    return eval_aos(zip(coeffs, exps), point)

# ---------------------------------------------------------------------------
# diff of polynomials, in any of the supported representations
# ---------------------------------------------------------------------------

def _coeff_parent(d):
    """Parent/type of the coefficients in a normalized term dict, or None if empty.
    Uses Sage's .parent() when available, else the Python type."""
    if not d:
        return None
    c = next(iter(d.values()))
    return c.parent() if hasattr(c, 'parent') else type(c)

def diff_polys(p1, p2, debug=False):
    """Compare two single polynomials (each AoS, SoA, or Sage poly).
    Coefficients are compared as-is (no normalization). Returns
    (n_monomial_diffs, n_coeff_diffs)."""
    A = to_term_dict(p1)
    B = to_term_dict(p2)

    pa, pb = _coeff_parent(A), _coeff_parent(B)
    if pa is not None and pb is not None and pa != pb:
        warnings.warn(
            f"coefficient bases differ: p1 over {pa}, p2 over {pb}; "
            "comparisons may be unreliable across different parents",
            stacklevel=2,
        )

    a_only = set(A) - set(B)
    b_only = set(B) - set(A)
    cdiff = {m: (A[m], B[m]) for m in (set(A) & set(B)) if A[m] != B[m]}

    if debug:
        print("  Monomials only in p1:", sorted(a_only))
        print("  Monomials only in p2:", sorted(b_only))
        print("  Monomials with different coefficients:")
        for m in sorted(cdiff):
            c0, c1 = cdiff[m]
            print(f"    {m}: p1 {c0}, p2 {c1}")

    return len(a_only) + len(b_only), len(cdiff)

def diff_polys_list(source1, source2, debug=False):
    """Compare two lists of coordinate polynomials, coordinate by coordinate."""
    n_mondiff = n_cdiff = 0
    for i, (p1, p2) in enumerate(zip(source1, source2)):
        if debug:
            print(f"Coordinate {i}:")
        m, c = diff_polys(p1, p2, debug=debug)
        n_mondiff += m
        n_cdiff += c
    return n_mondiff, n_cdiff

# ---------------------------------------------------------------------------
# power-map coordinate polynomials over an extension (or prime) field
# ---------------------------------------------------------------------------

def power_map_coordinate_polys(Fn, alpha, debug=False):
    """Coordinate polynomials of the power map  x -> x^alpha  over the field Fn.

    An element of a degree-n extension F_p[X]/f_mod is written in coordinates as
    x0 + x1*X + ... + x_{n-1}*X^{n-1}. Raising it to the alpha-th power (reduced
    mod f_mod) yields another element whose n coordinates are each a polynomial
    in x0..x_{n-1}. Those n coordinate polynomials are what this returns.

    Fn    : prime field GF(p), or an extension F.extension(f_mod, 'X').
    alpha : power-map exponent.
    Returns: a list of n = Fn.degree() polynomials in F_p[x0, ..., x_{n-1}].
             For a prime field (n == 1) the single coordinate is just x0^alpha.
    """
    n = Fn.degree()                  # extension degree (1 for a prime field)
    Fp = Fn.base_ring()              # the underlying base ring F_p

    # Coordinate ring: coefficients of the power map live in F_p[x0..x_{n-1}].
    coord_ring = PolynomialRing(Fp, ['x%d' % i for i in range(n)])

    if n == 1:
        # Prime field: the map is simply x0^alpha, one coordinate.
        coord_polys = [coord_ring.gen(0)**alpha]
    else:
        # Work in (F_p[x0..x_{n-1}])[X], i.e. univariate in X with coefficients
        # drawn from the coordinate ring, so we can carry the symbolic
        # coordinates through the extension arithmetic.
        modulus = Fn.modulus()                       # f_mod, as a poly in F_p[X]
        ext_ring = PolynomialRing(coord_ring, 'X')
        modulus_lifted = ext_ring([coord_ring(c) for c in modulus.list()])

        # Symbolic element  x0 + x1*X + ... + x_{n-1}*X^{n-1}.
        symbolic_el = univ_from_list(ext_ring.gen(), coord_ring.gens())

        # Raise to alpha and reduce mod f_mod; the X-coefficients of the result
        # are the coordinate polynomials.
        powered = symbolic_el**alpha % modulus_lifted
        coord_polys = powered.list()
        coord_polys += [coord_ring(0)] * (n - len(coord_polys))   # pad to n

    if debug:
        _debug_check_power_map(Fn, alpha, coord_polys, n, Fp)

    return coord_polys


def _debug_check_power_map(Fn, alpha, coord_polys, n, Fp):
    """Sanity check: evaluate the coordinate polynomials on a random field
    element and confirm they reproduce x^alpha computed natively in Fn, and
    that the AoS and SoA evaluators agree with the Sage-polynomial evaluation."""
    print("-- power_map_coordinate_polys --")
    print("Fn =", Fn, "\ndegree =", n, "\nalpha =", alpha)
    if n > 1:
        print("modulus =", Fn.modulus(), "\nirreducible:", Fn.modulus().is_irreducible())

    el = Fn.random_element()
    # element's coordinates, padded to length n (high zero coords may be dropped)
    coords = list(el.list()) + [Fp(0)] * (n - len(el.list()))

    # 1) evaluate via the Sage polynomials directly
    evaluated = [poly(*coords) for poly in coord_polys]

    # 2) evaluate via the AoS and SoA forms (same coordinates as input point)
    aos_forms = [poly_to_aos(poly) for poly in coord_polys]
    soa_forms = [poly_to_soa(poly) for poly in coord_polys]
    eval_aos_res = [eval_aos(terms, coords)          for terms        in aos_forms]
    eval_soa_res = [eval_soa(coeffs, exps, coords)   for (coeffs, exps) in soa_forms]

    if n == 1:
        native = el**alpha
        reconstructed = evaluated[0]
    else:
        Fp_X = Fn.modulus().parent()                 # F_p[X]
        native        = univ_from_list(Fp_X.gen(), (el**alpha).list())
        reconstructed = univ_from_list(Fp_X.gen(), evaluated)

    print("g            =", el)
    print("g^alpha      =", native,        "(native field arithmetic)")
    print("g^alpha      =", reconstructed, "(via coordinate polynomials)")
    print("match (poly) :", native == reconstructed)
    # AoS / SoA evaluators must reproduce the Sage-poly coordinate values
    print("match (AoS)  :", eval_aos_res == evaluated)
    print("match (SoA)  :", eval_soa_res == evaluated)
    print('-' * 60)