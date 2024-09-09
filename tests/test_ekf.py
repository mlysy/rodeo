import unittest
import jax
import jax.numpy as jnp
import jax.random as random
import rodeo.kalmantv as ktv
import rodeo.ekf as ekf
from rodeo.utils import mvncond
import utils
# jax.config.update("jax_enable_x64", True)
# --- kalmantv.predict ---------------------------------------------------------

class TestEKF(unittest.TestCase):
    """
    Test if EKF gives the same results as KalmanTV for linear state space model.

    """
    setUp = utils.kalman_setup

    def test_predict(self):
        # theta_{0|0}
        mean_state_past, var_state_past = utils.kalman_theta(
            m=0, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mean_gm, Sigma=self.var_gm
        )
        mean_state_pred1, var_state_pred1 = ktv.predict(
            mean_state_past=mean_state_past,
            var_state_past=var_state_past,
            mean_state=self.mean_state[1],
            wgt_state=self.wgt_state[0],
            var_state=self.var_state[1]
        )

        def mean_fun(mean_state_past, theta):
            return self.wgt_state[0].dot(mean_state_past) + self.mean_state[1]
    
        def var_fun(mean_state_past, theta):
            return self.var_state[1]

        mean_state_pred2, var_state_pred2, wgt_state2 = ekf.predict(
            mean_state_past=mean_state_past,
            var_state_past=var_state_past,
            mean_fun=mean_fun, 
            var_fun=var_fun, 
            theta=None
        )

        self.assertAlmostEqual(utils.rel_err(mean_state_pred1, mean_state_pred2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_pred1, var_state_pred2), 0.0)
        self.assertAlmostEqual(utils.rel_err(self.wgt_state[0], wgt_state2), 0.0)

    def test_update(self):
        # theta_{1|0}
        mean_state_pred, var_state_pred = utils.kalman_theta(
            m=1, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mean_gm, Sigma=self.var_gm
        )
        mean_state_filt1, var_state_filt1 = ktv.update(
            mean_state_pred=mean_state_pred[:self.n_meas],
            var_state_pred=var_state_pred[:self.n_meas, :self.n_meas],
            x_meas=self.x_meas[1],
            mean_meas=self.mean_meas[1],
            wgt_meas=self.wgt_meas[1][:self.n_meas, :self.n_meas],
            var_meas=self.var_meas[1]
        )

        def meas_lpdf(mean_state_pred, x_meas, theta):
            return jnp.sum(
                jax.scipy.stats.multivariate_normal.logpdf(
                    x_meas, 
                    mean=self.wgt_meas[1][:self.n_meas, :self.n_meas].dot(mean_state_pred) + self.mean_meas[1], 
                    cov=self.var_meas[1])
            )
        
        mean_state_filt2, var_state_filt2 = ekf.update(
            mean_state_pred=mean_state_pred[:self.n_meas],
            var_state_pred=var_state_pred[:self.n_meas, :self.n_meas],
            x_meas=self.x_meas[1],
            meas_lpdf=meas_lpdf,
            theta=None
        )
        self.assertAlmostEqual(utils.rel_err(mean_state_filt1, mean_state_filt2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_filt1, var_state_filt2), 0.0)


if __name__ == '__main__':
    unittest.main()
