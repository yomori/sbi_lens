# Full field SBI inference 


# load lsst year 10 settings
from sbi_lens.config import config_lsst_y_10
import pickle
from functools import partial
from tqdm import tqdm
from chainconsumer import ChainConsumer

from numpyro.handlers import trace, seed

import jax
import jax.numpy as jnp

from haiku._src.nets.resnet import ResNet18
import optax
import haiku as hk

import tensorflow_datasets as tfds
import tensorflow as tf

import tensorflow_probability as tfp; tfp = tfp.experimental.substrates.jax
tfb = tfp.bijectors
tfd = tfp.distributions

from sbi_lens.normflow.models import (
  ConditionalRealNVP,
  AffineCoupling
)


print("Loading experimental settings")
N                = config_lsst_y_10.N
map_size         = config_lsst_y_10.map_size
sigma_e          = config_lsst_y_10.sigma_e
gals_per_arcmin2 = config_lsst_y_10.gals_per_arcmin2
nbins            = config_lsst_y_10.nbins
a                = config_lsst_y_10.a
b                = config_lsst_y_10.b
z0               = config_lsst_y_10.z0
params_name      = config_lsst_y_10.params_name
truth            = config_lsst_y_10.truth

# compressor
dim = nb_cosmo_parameters = 6

compressor = hk.transform_with_state(lambda y : ResNet18(dim)(y, is_training=False))

# load compressor params
a_file = open('sbi_lens/data/params_compressor/opt_state_resnet_vmim.pkl', "rb")
opt_state_resnet= pickle.load(a_file)

a_file = open('sbi_lens/data/params_compressor/params_nd_compressor_vmim.pkl', "rb")
parameters_compressor= pickle.load(a_file)





# load observation and saved posterior or run HMC for a giver observed mass map
from sbi_lens.simulator.utils import get_reference_sample_posterior_full_field

print("------------------- Loading files ----------------------")
posterior_ff, obs_mass_map = get_reference_sample_posterior_full_field(
                                                                        run_mcmc=False,
                                                                        N=N,
                                                                        map_size=map_size,
                                                                        gals_per_arcmin2=gals_per_arcmin2,
                                                                        sigma_e=sigma_e
                                                                    )



print("-----------------compress observed mass map----------------------")
obs_mass_map_comressed, _ = compressor.apply(
                                                parameters_compressor,
                                                opt_state_resnet,
                                                None,
                                                obs_mass_map.reshape([1, N, N, nbins])
                                                )

print("-----------------Load datasets----------------------")
# Generating training examples -- this step takes a while
from sbi_lens.gen_dataset import LensingLogNormalDataset

ds = tfds.load(
                'LensingLogNormalDataset/year_10_with_noise_score_density',
                split='train'
              )

ds = ds.repeat()
ds = ds.shuffle(1000)
ds = ds.batch(128)
ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
ds_train = iter(tfds.as_numpy(ds))


print(" ------------------ create neural density estimator -> normalizing flow -------------------------")
from sbi_lens.normflow.models import AffineCoupling, ConditionalRealNVP

def loss_nll(params, mu, batch):
    comp_batch,_ = compressor.apply(
                                    parameters_compressor,
                                    opt_state_resnet,
                                    None,
                                    batch.reshape([-1, N, N, nbins])
                                   )
    return - jnp.mean(log_prob_fn(params, mu, comp_batch))

@jax.jit
def update(params, opt_state, mu, batch):
    """Single SGD update step."""
    loss, grads  = jax.value_and_grad(loss_nll)(
                                                params,
                                                mu,
                                                batch,
                                               )
    
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    return loss, new_params, new_opt_state


bijector_npe = partial(
                       AffineCoupling,
                       layers=[128] * 2,
                       activation=jax.nn.silu
                      )

NF_npe       = partial(
                       ConditionalRealNVP,
                       n_layers=4,
                       bijector_fn=bijector_npe
                      )

nvp_nd        = hk.without_apply_rng(hk.transform(lambda theta,y : NF_npe(dim)(y).log_prob(theta).squeeze()))

log_prob_fn   = lambda params, theta, y : nvp_nd.apply(params, theta, y)

nvp_sample_nd = hk.transform(lambda y : NF_npe(dim)(y).sample(100_000, seed=hk.next_rng_key()))


params = nvp_nd.init(
                     jax.random.PRNGKey(42),
                     0.5 * jnp.ones([1, dim]),
                     0.5 * jnp.ones([1, dim])
                    )

optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)


batch_loss = []
pbar = tqdm(range(3_000))

for batch in pbar:
    ex = next(ds_train)
    if not jnp.isnan(ex['simulation']).any():
        l, params, opt_state = update(
                                      params,
                                      opt_state,
                                      ex['theta'],
                                      ex['simulation'],
                                     )
        batch_loss.append(l)
        pbar.set_description(f"loss {l:.3f}")

sample_nd = nvp_sample_nd.apply(
                                params,
                                rng = jax.random.PRNGKey(43),
                                y   = obs_mass_map_comressed * jnp.ones([100_000, dim])
                            )