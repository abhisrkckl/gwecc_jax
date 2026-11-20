import jax.numpy as jnp

def residual_PQR(e, u):
    su = jnp.sin(u)
    cu = jnp.cos(u)
    c2u = cu * cu - su * su

    P = (jnp.sqrt(1 - e^2) * (c2u - e * cu)) / (1 - e * cu)
    Q = (((e^2 - 2) * cu + e) * su) / (1 - e * cu)
    R = e * su

    return P, Q, R


def residual_A(e, u, ω):
    s2ω = jnp.sin(2*ω)
    c2ω = jnp.cos(2*ω)

    P, Q, R = residual_PQR(e, u)

    sA0 = R
    sA1 = -P * s2ω + Q * c2ω
    sA2 = P * c2ω + Q * s2ω

    return sA0, sA1, sA2

