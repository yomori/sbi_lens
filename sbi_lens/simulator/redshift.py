import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax.tree_util import register_pytree_node_class
from jax_cosmo.scipy.integrate import simps
from tensorflow_probability.substrates.jax.math import find_root_chandrupatla

from sbi_lens.simulator.romberg import romb


@register_pytree_node_class
class photoz_bin(jc.redshift.redshift_distribution):
    """Defines a smail distribution with these arguments
    Parameters:
    -----------
    parent_pz:

    zphot_min:

    zphot_max:

    zphot_sig: coefficient in front of (1+z)
    """

    def pz_fn(self, z):
        parent_pz, zphot_min, zphot_max, zphot_sig = self.params
        p = parent_pz(z)

        # Apply photo-z errors
        x = 1.0 / (jnp.sqrt(2.0) * zphot_sig * (1.0 + z))
        res = (
            0.5
            * p
            * (
                jax.scipy.special.erf((zphot_max - z) * x)
                - jax.scipy.special.erf((zphot_min - z) * x)
            )
        )
        return res

    @property
    def gals_per_arcmin2(self):
        parent_pz, zphot_min, zphot_max, zphot_sig = self.params
        return parent_pz._gals_per_arcmin2 * simps(
            lambda t: parent_pz(t), zphot_min, zphot_max, 256
        )

    @property
    def gals_per_steradian(self):
        """Returns the number density of galaxies in steradian"""
        return self.gals_per_arcmin2 * jc.redshift.steradian_to_arcmin2


def subdivide(pz, nbins, zphot_sigma):
    """Divide this redshift bins into sub-bins
    nbins : Number of bins to generate
    bintype : 'eq_dens' or 'eq_size'
    """
    # Compute the redshift boundaries for each bin generated
    zbounds = [0.0]
    bins = []
    n_per_bin = 1.0 / nbins
    for i in range(nbins - 1):
        zbound = find_root_chandrupatla(
            lambda z: romb(pz, 0.0, z) - (i + 1.0) * n_per_bin, zbounds[i], pz.zmax
        ).estimated_root
        zbounds.append(zbound)
        new_bin = photoz_bin(pz, zbounds[i], zbounds[i + 1], zphot_sigma)
        bins.append(new_bin)

    zbounds.append(pz.zmax)
    new_bin = photoz_bin(pz, zbounds[nbins - 1], zbounds[nbins], zphot_sigma)
    bins.append(new_bin)

    return bins
