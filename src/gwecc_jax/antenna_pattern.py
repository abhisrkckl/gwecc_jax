import jax.numpy as jnp


def sky_direction_uvec(ra, dec):
    λ = ra
    β = dec

    n1 = jnp.cos(λ) * jnp.cos(β)
    n2 = jnp.sin(λ) * jnp.cos(β)
    n3 = jnp.sin(β)

    return jnp.array([n1, n2, n3])


def gw_polarization_tensors(ra, dec):
    λ = ra
    β = dec
    e11p = (jnp.sin(λ) ** 2) - (jnp.cos(λ) ** 2) * (jnp.sin(β) ** 2)
    e21p = -jnp.sin(λ) * jnp.cos(λ) * ((jnp.sin(β) ** 2) + 1)
    e31p = jnp.cos(λ) * jnp.sin(β) * jnp.cos(β)
    e12p = -jnp.sin(λ) * jnp.cos(λ) * ((jnp.sin(β) ** 2) + 1)
    e22p = (jnp.cos(λ) ** 2) - (jnp.sin(λ) ** 2) * (jnp.sin(β) ** 2)
    e32p = jnp.sin(λ) * jnp.sin(β) * jnp.cos(β)
    e13p = jnp.cos(λ) * jnp.sin(β) * jnp.cos(β)
    e23p = jnp.sin(λ) * jnp.sin(β) * jnp.cos(β)
    e33p = -(jnp.cos(β) ** 2)

    e11c = jnp.sin(2 * λ) * jnp.sin(β)
    e21c = -jnp.cos(2 * λ) * jnp.sin(β)
    e31c = -jnp.sin(λ) * jnp.cos(β)
    e12c = -jnp.cos(2 * λ) * jnp.sin(β)
    e22c = -jnp.sin(2 * λ) * jnp.sin(β)
    e32c = jnp.cos(λ) * jnp.cos(β)
    e13c = -jnp.sin(λ) * jnp.cos(β)
    e23c = jnp.cos(λ) * jnp.cos(β)
    e33c = 0.0

    ep = jnp.array([[e11p, e12p, e13p], [e21p, e22p, e23p], [e31p, e32p, e33p]])
    ec = jnp.array([[e11c, e12c, e13c], [e21c, e22c, e23c], [e31c, e32c, e33c]])

    return ep, ec


def antenna_pattern(ra_gw, dec_gw, ra_psr, dec_psr):
    phat = sky_direction_uvec(ra_psr, dec_psr)
    nhat = sky_direction_uvec(ra_gw, dec_gw)
    ep, ex = gw_polarization_tensors(ra_gw, dec_gw)

    cosµ = jnp.dot(phat, nhat)

    Fp = jnp.dot(phat, jnp.dot(ep, phat)) * 0.5 / (1 - cosµ)
    Fx = jnp.dot(phat, jnp.dot(ex, phat)) * 0.5 / (1 - cosµ)

    return Fp, Fx, cosμ
