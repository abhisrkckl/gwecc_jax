import numpy as np
import jax
import numdifftools as nd
from gwecc_jax.orbital_evolution import (
    e_from_τ,
    τ_from_e,
    n_from_e,
    lbar_from_e,
    γbar1_from_e,
    γbar2_from_e,
    γbar3_from_e,
    evolve_orbit,
)


def test_e_from_τ_inverse():
    # Test that e_from_τ and τ_from_e are inverses of each other
    τ_values = np.linspace(1e-6, 100, 10)
    for τ in τ_values:
        e = e_from_τ(τ)
        τ_reconstructed = τ_from_e(e)
        assert np.isclose(τ, τ_reconstructed, rtol=1e-5), f"Failed for τ={τ}"


def test_τ_from_e_inverse():
    # Test that τ_from_e and e_from_τ are inverses of each other
    e_values = np.linspace(0.01, 0.9, 10)
    for e in e_values:
        τ = τ_from_e(e)
        e_reconstructed = e_from_τ(τ)
        assert np.isclose(e, e_reconstructed, rtol=1e-5), f"Failed for e={e}"


def test_e_from_τ_monotonicity():
    # Test that e_from_τ is a monotonic function of τ
    τ_values = np.linspace(1e-6, 100, 100)
    e_values = e_from_τ(τ_values)
    assert np.all(np.diff(e_values) > 0), "e_from_τ is not monotonic"


def test_τ_from_e_monotonicity():
    # Test that τ_from_e is a monotonic function of e
    e_values = np.linspace(0.01, 0.9, 100)
    τ_values = τ_from_e(e_values)
    assert np.all(np.diff(τ_values) > 0), "τ_from_e is not monotonic"


def test_τ_from_e_derivative():
    # Test that the derivative of τ_from_e is positive
    e_values = np.linspace(0.01, 0.9, 10)
    for e in e_values:
        dτ_de_num = nd.Gradient(τ_from_e)(e)
        dτ_de_jax = jax.grad(τ_from_e)(e)
        assert np.allclose(
            dτ_de_num, dτ_de_jax, rtol=1e-5
        ), "Numerical and JAX derivatives of τ_from_e do not match"


def test_e_τ_derivatives():
    # Test that the derivative of e_from_τ is positive
    e_values = np.linspace(0.01, 0.9, 10)
    for e in e_values:
        u = jax.grad(lambda e: e_from_τ(τ_from_e(e)))(e)
        assert np.allclose(
            u, 1, rtol=1e-6
        ), "JAX derivatives of e_from_τ(τ_from_e(e)) is not 1."


def test_n_from_e_monotonicity():
    # Test that n_from_e is a monotonic function of e
    n0 = 1.0  # arbitrary reference value
    e_values = np.linspace(0.01, 0.9, 100)
    n_values = n_from_e(n0, e_values[0], e_values)
    assert np.all(np.diff(n_values) < 0), "n_from_e is not monotonic"


def test_n_from_e_positive():
    # Test that n_from_e is consistent with its definition
    n0 = 1.0  # arbitrary reference value
    e_values = np.linspace(0.01, 0.9, 10)
    for e in e_values:
        n = n_from_e(n0, e_values[0], e)
        assert n > 0, f"n_from_e returned non-positive value for e={e}"


def test_n_from_e_derivative():
    # Test that the derivative of n_from_e is negative
    n0 = 1.0  # arbitrary reference value
    e_values = np.linspace(0.01, 0.9, 10)
    for e in e_values:
        n = n_from_e(n0, e_values[0], e)
        dn_de_anl = (
            -3
            * n
            / e
            / (1 - e * e)
            * (96 + 292 * e**2 + 37 * e**4)
            / (304 + 121 * e**2)
        )
        dn_de_jax = jax.grad(lambda e: n_from_e(n0, e_values[0], e))(e)
        assert np.allclose(
            dn_de_anl, dn_de_jax, rtol=1e-5
        ), "Analytical and JAX derivatives of n_from_e do not match"


def test_lbar_from_e_monotonicity():
    # Test that lbar_from_e is a monotonic function of e
    e_values = np.linspace(0.01, 0.9, 10)
    lbar_values = lbar_from_e(e_values)
    assert np.all(np.diff(lbar_values) > 0), "lbar_from_e is not monotonic"


def test_lbar_from_e_derivative():
    # Test that the derivative of lbar_from_e is positive
    e_values = np.linspace(0.01, 0.9, 10)
    for e in e_values:
        dlbar_de_num = nd.Gradient(lbar_from_e)(e)
        dlbar_de_jax = jax.grad(lbar_from_e)(e)
        assert np.allclose(
            dlbar_de_num, dlbar_de_jax, rtol=1e-5
        ), "Numerical and JAX derivatives of lbar_from_e do not match"


def test_γbar1_from_e_monotonicity():
    # Test that γbar1_from_e is a monotonic function of e
    e_values = np.linspace(0.01, 0.9, 10)
    γbar1_values = γbar1_from_e(e_values)
    assert np.all(np.diff(γbar1_values) > 0), "γbar1_from_e is not monotonic"


def test_γbar1_from_e_derivative():
    # Test that the derivative of γbar1_from_e is positive
    e_values = np.linspace(0.01, 0.9, 10)
    for e in e_values:
        dγbar1_de_num = nd.Gradient(γbar1_from_e)(e)
        dγbar1_de_jax = jax.grad(γbar1_from_e)(e)
        assert np.allclose(
            dγbar1_de_num, dγbar1_de_jax, rtol=1e-5
        ), "Numerical and JAX derivatives of γbar1_from_e do not match"


def test_γbar2_from_e_monotonicity():
    # Test that γbar2_from_e is a monotonic function of e
    e_values = np.linspace(0.01, 0.9, 10)
    η = 0.25  # arbitrary value for η
    γbar2_values = γbar2_from_e(e_values, η)
    assert np.all(np.diff(γbar2_values) > 0), "γbar2_from_e is not monotonic"


def test_γbar2_from_e_derivative():
    # Test that the derivative of γbar2_from_e is positive
    e_values = np.linspace(0.01, 0.9, 10)
    η = 0.25  # arbitrary value for η
    for e in e_values:
        dγbar2_de_num = nd.Gradient(lambda e: γbar2_from_e(e, η))(e)
        dγbar2_de_jax = jax.grad(lambda e: γbar2_from_e(e, η))(e)
        assert np.allclose(
            dγbar2_de_num, dγbar2_de_jax, rtol=1e-5
        ), "Numerical and JAX derivatives of γbar2_from_e do not match"


def test_γbar3_from_e_finite():
    # Test that γbar3_from_e is finite for e in (0, 1)
    e_values = np.linspace(0.01, 0.9, 10)
    η = 0.25  # arbitrary value for η
    for e in e_values:
        γbar3 = γbar3_from_e(e, η)
        assert np.isfinite(γbar3), f"γbar3_from_e returned non-finite value for e={e}"


def test_γbar3_from_e_derivative():
    # Test that the derivative of γbar3_from_e is positive
    e_values = np.linspace(0.01, 0.9, 10)
    η = 0.25  # arbitrary value for η
    for e in e_values:
        dγbar3_de_num = nd.Gradient(lambda e: γbar3_from_e(e, η))(e)
        dγbar3_de_jax = jax.grad(lambda e: γbar3_from_e(e, η))(e)
        assert np.allclose(
            dγbar3_de_num, dγbar3_de_jax, rtol=1e-5
        ), "Numerical and JAX derivatives of γbar3_from_e do not match"


def test_evolve_orbit_consistency():
    # Test that evolve_orbit returns consistent results
    M = 5000 # s
    η = 0.25
    t0 = 0.0 # s
    e0 = 0.1
    n0 = 1e-8 # rad/s
    l0 = 0.5 # rad
    γ0 = 0.2
    t = 1e6 # s

    n, e, l, γ = evolve_orbit(M, η, t0, e0, n0, l0, γ0, t)

    assert n > n0
    assert e < e0
    assert l > l0
    assert γ > γ0