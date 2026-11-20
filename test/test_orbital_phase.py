from gwecc_jax.orbital_phase import mikkola, kepler
import jax.numpy as jnp
import jax

def test_mikkola_kepler():
    # Test that mikkola solves Kepler's equation
    e_values = jnp.linspace(0.01, 0.9, 10)
    l_values = jnp.linspace(0, 2 * jnp.pi, 10)

    for e in e_values:
        for l in l_values:
            u = mikkola(e, l)
            l_reconstructed = kepler(e, u)
            assert jnp.isclose(l, l_reconstructed, rtol=1e-6), f"Failed for e={e}, l={l}"

def test_mikkola_derivative():
    # Test that the derivative of mikkola is consistent with the derivative of kepler
    e_values = jnp.linspace(0.01, 0.9, 10)
    l_values = jnp.linspace(0, 2 * jnp.pi, 10)

    for e in e_values:
        for l in l_values:
            u = mikkola(e, l)
            du_de = jax.grad(mikkola, argnums=0)(e, l)
            du_dl = jax.grad(mikkola, argnums=1)(e, l)

            denom = 1 - e * jnp.cos(u)
            du_de_expected = jnp.sin(u) / denom
            du_dl_expected = 1 / denom

            assert jnp.isclose(du_de, du_de_expected, rtol=1e-6), f"Failed for e={e}, l={l} (du/de)"
            assert jnp.isclose(du_dl, du_dl_expected, rtol=1e-6), f"Failed for e={e}, l={l} (du/dl)"