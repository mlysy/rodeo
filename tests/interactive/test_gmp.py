import jax
import jax.numpy as jnp
import numpy as np
import jax.random as random
import unittest
import rodeo.gauss_markov as gm
from rodeo.utils import mvncond
from jax.config import config
import utils
config.update("jax_enable_x64", True)

class TestGMP(unittest.TestCase):
    
    def test_gmp(self):
        # parameters for gmp
        key = random.PRNGKey(0)
        key, *subkeys = random.split(key, 4)
        t = 3
        T = 6
        n_mat = 3
        A = random.normal(subkeys[0], (T, n_mat, n_mat))
        b = random.normal(subkeys[1], (T+1, n_mat))
        V = random.normal(subkeys[2], (T+1, n_mat, n_mat))
        V = jax.vmap(lambda vs: vs.dot(vs.T))(V)
        C = jax.vmap(lambda vs: jnp.linalg.cholesky(vs))(V)
        mu_gm, V_gm = gm.gauss_markov_mv(A=A, b=b, C=C)

        # test forward gmp
        mu = jnp.ravel(mu_gm[:t+1])
        Sigma = jnp.reshape(V_gm[:t+1, :, :t+1], ((t+1)*n_mat, (t+1)*n_mat))
        Acond, bcond, Vcond = mvncond(mu, Sigma, icond=jnp.array([True]*n_mat*t + [False]*n_mat))

        # test backward gmp
        mu = jnp.ravel(mu_gm[t:])
        Sigma = jnp.reshape(V_gm[t:, :, t:], ((T-t+1)*n_mat, (T-t+1)*n_mat))
        Acond2, bcond2, Vcond2 = mvncond(mu, Sigma, icond=jnp.array([False]*n_mat + [True]*n_mat*(T-t)))


        self.assertAlmostEqual(utils.rel_err(Acond[:, :-n_mat], np.zeros((n_mat, n_mat*(t-1)))), 0.0)
        self.assertAlmostEqual(utils.rel_err(Acond2[:, n_mat:], np.zeros((n_mat, n_mat*(T-t-1)))), 0.0)


if __name__ == '__main__':
    unittest.main()
