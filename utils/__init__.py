"""Shared utilities for the primitives in this repo.

Submodules, grouped by concern:
  - matrix          : matrix/vector arithmetic, nested-list mapping, MDS/diffusion matrices
  - sampler         : deterministic field-element samplers for round-constant generation
  - field           : the Field type and predefined field instances
  - lut             : lookup-table helpers for LUT-based constructions
  - mode            : modes of operation (sponge / compression / padding)
  - complexities    : attack-complexity estimators

Import from the relevant submodule, e.g. `from utils.matrix import circulant`.
"""
