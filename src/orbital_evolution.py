import jax
import jax.numpy as jnp
import numpy as np
from interpax import CubicSpline
from scipy.special import hyp2f1

τs, es = np.loadtxt("GWecc.txt", delimiter=" ").T


def make_e_τ_funcs(τs, es):
    e_from_τ_spline = CubicSpline(τs, es)
    τ_from_e_spline = CubicSpline(es, τs)

    emin = jnp.min(es)
    emax = jnp.max(es)
    τmin = jnp.min(τs)
    τmax = jnp.max(τs)

    a = 2 * jnp.sqrt(2) / 5 ** (2362 / 2299) / 17 ** (1181 / 2299)
    b = 2 / jnp.sqrt(1 - emax) - a * τmax

    # e(τ)
    @jax.custom_jvp
    def e_from_τ(τ):
        coeff = 2 ** (559 / 726) * 3 ** (19 / 48) / 19 ** (145 / 242)

        low = coeff * τ ** (19 / 48)
        high = 1 - 4 / (a * τ + b) ** 2
        middle = e_from_τ_spline(τ)

        return jnp.where(τ < τmin, low, jnp.where(τ > τmax, high, middle))

    @e_from_τ.defjvp
    def e_from_τ_jvp(primals, tangents):
        (τ,), (τ_dot,) = primals, tangents

        e = e_from_τ(τ)
        de = (
            (1 - e * e) ** (3 / 2)
            / e ** (29 / 19)
            / (121 * e * e + 304) ** (1181 / 2299)
        )

        return e, de * τ_dot

    # τ(e)
    @jax.custom_jvp
    def τ_from_e(e):
        coeff = 19 * 19 ** (1181 / 2299) / (6 * 2 ** (2173 / 2299))

        low = coeff * e ** (48 / 19)
        high = (2 / jnp.sqrt(1 - e) - b) / a
        middle = τ_from_e_spline(e)

        return jnp.where(e < emin, low, jnp.where(e > emax, high, middle))

    @τ_from_e.defjvp
    def τ_from_e_jvp(primals, tangents):
        (e,), (e_dot,) = primals, tangents

        τ = τ_from_e(e)
        dτ = (
            e ** (29 / 19)
            * (121 * e * e + 304) ** (1181 / 2299)
            / (1 - e * e) ** (3 / 2)
        )

        return τ, dτ * e_dot

    return e_from_τ, τ_from_e


e_from_τ, τ_from_e = make_e_τ_funcs(τs, es)


@jax.jit
def n_from_e(n0, e0, e):
    return (
        n0
        * (e0 / e) ** (18 / 19)
        * ((1 - e**2) / (1 - e0**2)) ** 1.5
        * ((304 + 121 * e0**2) / (304 + 121 * e**2)) ** (1305 / 2299)
    )


@jax.jit
def evolv_coeff_κ(Mch, n0, e0):
    """Coefficient for scaling time. Eq. 20b of Susobhanan+ 2020."""
    return (
        (1 / 15)
        * (Mch * n0) ** (5 / 3)
        * n0
        * e0 ** (48 / 19)
        * (304 + 121 * e0**2) ** (3480 / 2299)
        / (1 - e0**2) ** 4
    )


def evolv_coeff_α(Mch, n0, e0):
    "Coefficient for scaling mean anomaly. Eq. 26a of Susobhanan+ 2020."
    return (
        15
        / Mch ** (5 / 3)
        / n0 ** (5 / 3)
        / e0 ** (30 / 19)
        / (304 + 121 * e0**2) ** (2175 / 2299)
        * (1 - e0**2) ** 2.5
    )


def evolv_coeff_β(Mch, M, n0, e0):
    """Coefficient for scaling the periastron angle at 1PN order.
    Eq. 26b of Susobhanan+ 2020."""
    return (
        45
        / Mch ** (5 / 3)
        * M ** (2 / 3)
        / n0
        / e0 ** (18 / 19)
        / (304 + 121 * e0**2) ** (1305 / 2299)
        * (1 - e0**2) ** 1.5
    )


def evolv_coeff_β2(Mch, M, n0, e0):
    """Coefficient for scaling the periastron angle at 2PN order.
    Eq. A7 of Susobhanan+ 2020."""
    return (
        15
        / (4 * Mch ** (5 / 3))
        * M ** (4 / 3)
        / jnp.cbrt(n0)
        / e0 ** (6 / 19)
        / (304 + 121 * e0**2) ** (435 / 2299)
        * jnp.sqrt(1 - e0**2)
    )


def evolv_coeff_β3(Mch, M, n0, e0):
    """Coefficient for scaling the periastron angle at 3PN order.
    Eq. A13 of Susobhanan+ 2020."""
    return (
        15
        / (128 * Mch ** (5 / 3))
        * (M * M)
        * jnp.cbrt(n0)
        * e0 ** (6 / 19)
        * (304 + 121 * e0**2) ** (435 / 2299)
        / jnp.sqrt(1 - e0**2)
    )


def lbar_from_e(e):
    """Scaled mean anomaly as a function of eccentricity.
    Eq. 30a of Susobhanan+ 2020"""

    coeff = 19 ** (2175 / 2299) / 30 / 2 ** (496 / 2299)
    return (
        coeff
        * e
        * e ** (11 / 19)
        * hyp2f1(124 / 2299, 15 / 19, 34 / 19, -121 * e**2 / 304)
    )


def γbar1_from_e(e):
    """Scaled periastron angle (1PN) as a function of eccentricity.
    Eq. 30b of Susobhanan+ 2020."""
    coeff = 19 ** (1305 / 2299) / 36 / 2 ** (1677 / 2299)
    return (
        coeff * e ** (18 / 19) * hyp2f1(994 / 2299, 9 / 19, 28 / 19, -121 * e**2 / 304)
    )


def γbar2_from_e(e, η):
    """Scaled periastron angle (2PN) as a function of eccentricity
    Eq. A10 of Susobhanan+ 2020"""
    coeff = 3 * 2 ** (1740 / 2299) * 19 ** (435 / 2299)
    return (
        e ** (6 / 19)
        / 336
        * (
            4 * (51 - 26 * η) * (304 + 121 * e**2) ** (435 / 2299)
            + coeff
            * (23 + 2 * η)
            * hyp2f1(3 / 19, 1864 / 2299, 22 / 19, -121 * e**2 / 304)
        )
    )


def γbar3_from_e(e, η):
    """Scaled periastron angle (3PN) as a function of eccentricity.
    Eq. A14 of Susobhanan+ 2020 (Padé approximant)."""
    # γbar3(e) = (γbar3_0 + η*γbar3_1(e) + (η^2)*γbar3_2(e)) / e^(6/19)

    # The functions gbar30 and gbar31 contain Appell F1 functions that are
    # hard to compute. So I am using Padé approximants here. Since this is a
    # 3PN correction the error due to this approximation should be negligible.
    # This probably won't work if e == 0.

    γbar3_0 = -71.19148913450734 + (
        14.212810668086124 * (e * e)
        - 2.7809293422212327 * (e**4)
        - 0.5247248712090506 * (e**6)
    ) / (1.0 + 0.021216346439953203 * (e * e) - 0.08582538305359699 * (e**4))

    γbar3_1 = 75.17536852020817 + (
        -11.720605504532221 * (e * e)
        + 0.9937219102426796 * (e**4)
        + 0.3416818503954963 * (e**6)
    ) / (1.0 + 0.11825867242669967 * (e * e) - 0.05881762636038525 * (e**4))

    γbar3_2 = -3.1640661837558817 + (
        3.109257298381722 * (e * e)
        + 1.0709804529990066 * (e**4)
        + 0.05588268809419086 * (e**6)
    ) / (1.0 + 0.46114315553206664 * (e * e) + 0.041157522000398405 * (e**4))

    return (γbar3_0 + η * γbar3_1 + η * η * γbar3_2) / e ** (6 / 19)

def evolve_orbit(
    M,
    η,
    t0,
    e0,
    n0,
    l0,
    γ0,
    t,
):
    dt = t - t0

    Mch = M * η ** (3 / 5)

    τ0 = τ_from_e(e0)
    κ = evolv_coeff_κ(Mch, n0, e0)
    τ = τ0 - κ * dt 
    e = e_from_τ(τ)
    n = n_from_e(n0, e0, e)

    α = evolv_coeff_α(Mch, n0, e0)
    lbar0 = lbar_from_e(e0)
    lbar = lbar_from_e(e)
    l = l0 + (lbar0 - lbar) * α

    β1 = evolv_coeff_β(Mch, M, n0, e0)
    β2 = evolv_coeff_β2(Mch, M, n0, e0)
    β3 = evolv_coeff_β3(Mch, M, n0, e0)
    γbar10 = γbar1_from_e(e0)
    γbar20 = γbar2_from_e(e0, η)
    γbar30 = γbar3_from_e(e0, η)
    γbar1 = γbar1_from_e(e)
    γbar2 = γbar2_from_e(e, η)
    γbar3 = γbar3_from_e(e, η)
    γ = γ0 + (γbar10 - γbar1) * β1 + (γbar20 - γbar2) * β2 + (γbar30 - γbar3) * β3

    return n, e, l, γ


def de__dt(Mch, e, n):
    return (
        (-1 / 15)
        * (Mch * n) ** (5 / 3)
        * n
        * e
        * (304 + 121 * e**2)
        / (1 - e**2) ** 2.5
    )


def dn__dt(Mch, e, n):
    return (
        (1 / 5)
        * (Mch * n) ** (5 / 3)
        * n**2
        * (96 + 292 * e**2 + 37 * e**4)
        / (1 - e**2) ** 3.5
    )
