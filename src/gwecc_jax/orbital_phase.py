import jax
import jax.numpy as jnp

from .quasi_keplerian import advance_of_periastron, angular_eccentricity


@jax.custom_jvp
def mikkola(e, l):
    """Mikkola's method for solving Kepler's equation.

    Eccentric anomaly as a function of mean anomaly.
    Mikkola 1987.

    Supports broadcasting between e and l.
    """

    e = jnp.asarray(e)
    l = jnp.asarray(l)

    # Handle negative mean anomalies
    sgn = jnp.sign(l)
    l_abs = jnp.abs(l)

    # Reduce mean anomaly to [0, 2*pi)
    ncycles = jnp.floor(l_abs / (2 * jnp.pi))
    l_red = l_abs - 2 * jnp.pi * ncycles

    # Reflect to [0, pi]
    flag = l_red > jnp.pi
    l_red = jnp.where(flag, 2 * jnp.pi - l_red, l_red)

    # Mikkola's initial approximation
    alpha = (1 - e) / (4 * e + 0.5)
    beta = (l_red / 2) / (4 * e + 0.5)

    sqrt_term = jnp.sqrt(alpha**3 + beta**2)

    z = jnp.where(
        beta > 0,
        jnp.cbrt(beta + sqrt_term),
        jnp.cbrt(beta - sqrt_term),
    )

    s = z - alpha / z
    w = s - (0.078 * s**5) / (1 + e)

    E0 = l_red + e * (3 * w - 4 * w**3)

    # Fourth-order refinement
    esu = e * jnp.sin(E0)
    ecu = e * jnp.cos(E0)

    fu = E0 - esu - l_red
    f1u = 1 - ecu
    f2u = esu
    f3u = ecu
    f4u = -esu

    u1 = -fu / f1u
    u2 = -fu / (f1u + f2u * u1 / 2)
    u3 = -fu / (f1u + f2u * u2 / 2 + f3u * u2**2 / 6)
    u4 = -fu / (f1u + f2u * u3 / 2 + f3u * u3**2 / 6 + f4u * u3**3 / 24)

    xi = E0 + u4

    # Undo reflection and restore complete orbital cycles
    sol = jnp.where(flag, 2 * jnp.pi - xi, xi)
    result = sgn * (sol + 2 * jnp.pi * ncycles)

    # Circular-orbit and zero-mean-anomaly limits
    result = jnp.where((e == 0) | (l == 0), l, result)

    return result


@mikkola.defjvp
def mikkola_jvp(primals, tangents):
    e, l = primals
    e_dot, l_dot = tangents

    u = mikkola(e, l)

    denom = 1 - e * jnp.cos(u)

    du_de = jnp.sin(u) / denom
    du_dl = 1 / denom

    u_dot = du_de * e_dot + du_dl * l_dot

    return u, u_dot


def kepler(e, u):
    """Kepler equation. Mean anomaly as a function of eccentric anomaly."""
    return u - e * jnp.sin(u)


@jax.jit
def true_anomaly_diff(e, u):
    """Difference between true anomaly and eccentric anomaly
    as a function of eccentric anomaly.

    Parameters
    ----------
    e : array_like
        Orbital eccentricity.
    u : array_like
        Eccentric anomaly (radians).

    Returns
    -------
    array_like
        Difference between true and eccentric anomalies (radians).
    """

    su = jnp.sin(u)
    cu = jnp.cos(u)

    beta_phi = jnp.where(
        e > 1e-15,
        (1 - jnp.sqrt(1 - e**2)) / e,
        e / 2 + e**3 / 8 + e**5 / 16,
    )

    v_u = 2 * jnp.arctan2(beta_phi * su, 1 - beta_phi * cu)
    v_l = v_u + e * su

    return v_l


def orbital_phase(e, n, l, γ, M, η):
    x = (M * n) ** (2 / 3)
    k = advance_of_periastron(x, e, η)
    x = (M * (1 + k) * n) ** (2 / 3)
    k = advance_of_periastron(x, e, η)

    eφ = angular_eccentricity(x, e, η)

    u = mikkola(e, l)
    v_l = true_anomaly_diff(eφ, u)
    v = v_l + l

    W = (1 + k) * v_l

    φ = γ + l + W
    ω = φ - v

    return u, ω
