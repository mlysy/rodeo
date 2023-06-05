import unittest
import jax
import jax.numpy as jnp
import jax.random as random
import rodeo.kalmantv as ktv
import utils
from rodeo.utils import mvncond
# from jax.config import config
# config.update("jax_enable_x64", True)
# --- kalmantv.predict ---------------------------------------------------------

class TestKalmanTVGM(unittest.TestCase):
    """
    Test if KalmanTV gives the same results as Gaussian Markov process.

    """
    setUp = utils.kalman_setup

    def test_predict(self):
        # theta_{0|0}
        mean_state_past, var_state_past = utils.kalman_theta(
            m=0, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mean_gm, Sigma=self.var_gm
        )
        # theta_{1|0}
        mean_state_pred1, var_state_pred1 = utils.kalman_theta(
            m=1, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mean_gm, Sigma=self.var_gm
        )
        mean_state_pred2, var_state_pred2 = ktv.predict(
            mean_state_past=mean_state_past,
            var_state_past=var_state_past,
            mean_state=self.mean_state[1],
            trans_state=self.trans_state[0],
            var_state=self.var_state[1]
        )

        self.assertAlmostEqual(utils.rel_err(mean_state_pred1, mean_state_pred2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_pred1, var_state_pred2), 0.0)

    def test_update(self):
        # theta_{1|0}
        mean_state_pred, var_state_pred = utils.kalman_theta(
            m=1, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mean_gm, Sigma=self.var_gm
        )
        # theta_{1|1}
        mean_state_filt1, var_state_filt1 = utils.kalman_theta(
            m=1, y=self.x_meas[0:2], mu=self.mean_gm, Sigma=self.var_gm
        )
        mean_state_filt2, var_state_filt2 = ktv.update(
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred,
            W=self.trans_meas[1],
            x_meas=self.x_meas[1],
            mean_meas=self.mean_meas[1],
            trans_meas=self.trans_meas[1],
            var_meas=self.var_meas[1]
        )

        self.assertAlmostEqual(utils.rel_err(mean_state_filt1, mean_state_filt2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_filt1, var_state_filt2), 0.0)

    def test_filter(self):
        # theta_{0|0}
        mean_state_past, var_state_past = utils.kalman_theta(
            m=0, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mean_gm, Sigma=self.var_gm
        )
        # theta_{1|0}
        mean_state_pred1, var_state_pred1 = utils.kalman_theta(
            m=1, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mean_gm, Sigma=self.var_gm
        )
        # theta_{1|1}
        mean_state_filt1, var_state_filt1 = utils.kalman_theta(
            m=1, y=self.x_meas[0:2], mu=self.mean_gm, Sigma=self.var_gm
        )
        mean_state_pred2, var_state_pred2, \
            mean_state_filt2, var_state_filt2 = ktv.filter(
                mean_state_past=mean_state_past,
                var_state_past=var_state_past,
                mean_state=self.mean_state[1],
                trans_state=self.trans_state[0],
                var_state=self.var_state[1],
                x_meas=self.x_meas[1],
                mean_meas=self.mean_meas[1],
                trans_meas=self.trans_meas[1],
                var_meas=self.var_meas[1]
            )

        self.assertAlmostEqual(utils.rel_err(mean_state_pred1, mean_state_pred2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_pred1, var_state_pred2), 0.0)
        self.assertAlmostEqual(utils.rel_err(mean_state_filt1, mean_state_filt2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_filt1, var_state_filt2), 0.0)

    def test_filter2(self):
        # theta_{1|1}
        mean_state_past, var_state_past = utils.kalman_theta(
            m=1, y=self.x_meas[0:2], mu=self.mean_gm, Sigma=self.var_gm
        )
        # theta_{2|1}
        mean_state_pred1, var_state_pred1 = utils.kalman_theta(
            m=2, y=self.x_meas[0:2], mu=self.mean_gm, Sigma=self.var_gm
        )
        # theta_{2|2}
        mean_state_filt1, var_state_filt1 = utils.kalman_theta(
            m=2, y=self.x_meas[0:3], mu=self.mean_gm, Sigma=self.var_gm
        )
        mean_state_pred2, var_state_pred2, \
            mean_state_filt2, var_state_filt2 = ktv.filter(
                mean_state_past=mean_state_past,
                var_state_past=var_state_past,
                mean_state=self.mean_state[2],
                trans_state=self.trans_state[1],
                var_state=self.var_state[2],
                x_meas=self.x_meas[2],
                mean_meas=self.mean_meas[2],
                trans_meas=self.trans_meas[2],
                var_meas=self.var_meas[2]
            )

        self.assertAlmostEqual(utils.rel_err(mean_state_pred1, mean_state_pred2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_pred1, var_state_pred2), 0.0)
        self.assertAlmostEqual(utils.rel_err(mean_state_filt1, mean_state_filt2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_filt1, var_state_filt2), 0.0)

    def test_smooth_mv(self):
        # theta_{1|1}
        mean_state_next, var_state_next = utils.kalman_theta(
            m=1, y=self.x_meas, mu=self.mean_gm, Sigma=self.var_gm
        )
        # theta_{0|0}
        mean_state_filt, var_state_filt = utils.kalman_theta(
            m=0, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mean_gm, Sigma=self.var_gm
        )
        # theta_{1|0}
        mean_state_pred, var_state_pred = utils.kalman_theta(
            m=1, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mean_gm, Sigma=self.var_gm
        )
        # theta_{0|1}
        mean_state_smooth1, var_state_smooth1 = utils.kalman_theta(
            m=0, y=self.x_meas, mu=self.mean_gm, Sigma=self.var_gm
        )

        mean_state_smooth2, var_state_smooth2 = ktv.smooth_mv(
            mean_state_next=mean_state_next,
            var_state_next=var_state_next,
            mean_state_filt=mean_state_filt,
            var_state_filt=var_state_filt,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred,
            trans_state=self.trans_state[0]
        )

        self.assertAlmostEqual(utils.rel_err(mean_state_smooth1, mean_state_smooth2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_smooth1, var_state_smooth2), 0.0)

    def test_smooth_sim(self):
        # theta_{0|0}
        mean_state_filt, var_state_filt = utils.kalman_theta(
            m=0, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mean_gm, Sigma=self.var_gm
        )
        # theta_{1|0}
        mean_state_pred, var_state_pred = utils.kalman_theta(
            m=1, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mean_gm, Sigma=self.var_gm
        )
        # theta_{0:1|1}
        mean_state_smooth, var_state_smooth = utils.kalman_theta(
            m=[0, 1], y=self.x_meas, mu=self.mean_gm, Sigma=self.var_gm
        )
        A, b, V = mvncond(
            mu=mean_state_smooth.ravel(),
            Sigma=var_state_smooth.reshape(2*self.n_state, 2*self.n_state),
            icond=jnp.array([False]*self.n_state + [True]*self.n_state)
        )
        mean_state_sim1 = A.dot(self.x_state_next)+b
        # x_state_smooth1 = random.multivariate_normal(self.key, A.dot(self.x_state_next)+b, V)
        #x_state_smooth1 = ktv._state_sim(
        #    mean_state=A.dot(self.x_state_next)+b,
        #    var_state=V,
        #    z_state=self.z_state
        #)

        mean_state_sim2, var_state_sim2 = ktv.smooth_sim(
            x_state_next=self.x_state_next,
            mean_state_filt=mean_state_filt,
            var_state_filt=var_state_filt,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred,
            trans_state=self.trans_state[0]
        )

        self.assertAlmostEqual(utils.rel_err(mean_state_sim1, mean_state_sim2), 0.0)
        self.assertAlmostEqual(utils.rel_err(V, var_state_sim2), 0.0)
        

    def test_smooth(self):
        # theta_{1|1}
        mean_state_next, var_state_next = utils.kalman_theta(
            m=1, y=self.x_meas, mu=self.mean_gm, Sigma=self.var_gm
        )
        # theta_{0|0}
        mean_state_filt, var_state_filt = utils.kalman_theta(
            m=0, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mean_gm, Sigma=self.var_gm
        )
        # theta_{1|0}
        mean_state_pred, var_state_pred = utils.kalman_theta(
            m=1, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mean_gm, Sigma=self.var_gm
        )
        # theta_{0:1|1}
        mean_state_smooth1, var_state_smooth1 = utils.kalman_theta(
            m=[0, 1], y=self.x_meas, mu=self.mean_gm, Sigma=self.var_gm
        )
        A, b, V = mvncond(
            mu=mean_state_smooth1.ravel(),
            Sigma=var_state_smooth1.reshape(2*self.n_state, 2*self.n_state),
            icond=jnp.array([False]*self.n_state + [True]*self.n_state)
        )
        mean_state_sim1 = A.dot(self.x_state_next)+b
        #x_state_smooth1 = ktv._state_sim(
        #    mean_state=A.dot(self.x_state_next)+b,
        #    var_state=V,
        #    z_state=self.z_state
        #)
        # x_state_smooth1 = jax.random.multivariate_normal(self.key, A.dot(self.x_state_next)+b, V)
        mean_state_sim2, var_state_sim2, mean_state_smooth2, var_state_smooth2 = ktv.smooth(
            x_state_next=self.x_state_next,
            mean_state_next=mean_state_next,
            var_state_next=var_state_next,
            mean_state_filt=mean_state_filt,
            var_state_filt=var_state_filt,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred,
            trans_state=self.trans_state[0],
        )
        self.assertAlmostEqual(utils.rel_err(mean_state_sim1, mean_state_sim2), 0.0)
        self.assertAlmostEqual(utils.rel_err(V, var_state_sim2), 0.0)
        self.assertAlmostEqual(utils.rel_err(mean_state_smooth1[0], mean_state_smooth2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_smooth1[0, :, 0, :].squeeze(), var_state_smooth2), 0.0)
    
    def test_smooth_cond(self):
        # theta_{0|0}
        mean_state_filt, var_state_filt = utils.kalman_theta(
            m=0, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mean_gm, Sigma=self.var_gm
        )
        # theta_{1|0}
        mean_state_pred, var_state_pred = utils.kalman_theta(
            m=1, y=jnp.atleast_2d(self.x_meas[0]), mu=self.mean_gm, Sigma=self.var_gm
        )
        # theta_{0:1|1}
        mean_state_smooth, var_state_smooth = utils.kalman_theta(
            m=[0, 1], y=self.x_meas, mu=self.mean_gm, Sigma=self.var_gm
        )
        A, b, V = mvncond(
            mu=mean_state_smooth.ravel(),
            Sigma=var_state_smooth.reshape(2*self.n_state, 2*self.n_state),
            icond=jnp.array([False]*self.n_state + [True]*self.n_state)
        )
        # mean_state_sim1 = A.dot(self.x_state_next)+b
        # x_state_smooth1 = random.multivariate_normal(self.key, A.dot(self.x_state_next)+b, V)
        #x_state_smooth1 = ktv._state_sim(
        #    mean_state=A.dot(self.x_state_next)+b,
        #    var_state=V,
        #    z_state=self.z_state
        #)

        A2, b2, V2 = ktv.smooth_cond(
            mean_state_filt=mean_state_filt,
            var_state_filt=var_state_filt,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred,
            trans_state=self.trans_state[0]
        )

        self.assertAlmostEqual(utils.rel_err(A, A2), 0.0)
        self.assertAlmostEqual(utils.rel_err(b, b2), 0.0)
        self.assertAlmostEqual(utils.rel_err(V, V2), 0.0)

if __name__ == '__main__':
    unittest.main()
