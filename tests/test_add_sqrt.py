import unittest
import jax
import jax.numpy as jnp
from rodeo.utils import add_sqrt

class TestAddSqrt(unittest.TestCase):
    """
    Test if add_sqrt() give same result as expected math result.
    """

    def test_add_sqrt(self):
        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        sqrt_A = jax.random.normal(subkey, shape = (2,2))
        key, subkey = jax.random.split(key)
        sqrt_B = jax.random.normal(subkey, shape = (2,2))
        A = jnp.dot(sqrt_A, sqrt_A.T)
        B = jnp.dot(sqrt_B, sqrt_B.T)
        sqrt_AB = add_sqrt(sqrt_A, sqrt_B)
        self.assertTrue(jnp.allclose(A + B, jnp.dot(sqrt_AB, sqrt_AB.T)))
        
if __name__ == '__main__':
    unittest.main()
    
    
