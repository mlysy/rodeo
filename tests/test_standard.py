import unittest
import jax
import jax.numpy as jnp
import rodeo.kalmantv.standard as ktv
import utils


# --- kalmantv.predict ---------------------------------------------------------


class TestKalmanTVGM(unittest.TestCase):
    """
    Test if KalmanTV gives the same results as Gaussian Markov process.

    """
    setUp = utils.kalman_setup

    def test_predict(self):
        mean_state_past, var_state_past, mean_state_pred1, var_state_pred1, _, _ = utils.test_filter(self)
        mean_state_pred2, var_state_pred2 = jax.jit(ktv.predict)(
            mean_state_past=mean_state_past,
            var_state_past=var_state_past,
            mean_state=self.mean_state[1],
            wgt_state=self.wgt_state[0],
            var_state=self.var_state[1]
        )

        self.assertAlmostEqual(utils.rel_err(
            mean_state_pred1, mean_state_pred2), 0.0)
        self.assertAlmostEqual(utils.rel_err(
            var_state_pred1, var_state_pred2), 0.0)
        
        
    def test_update(self):
        _, _, mean_state_pred, var_state_pred,  mean_state_filt1, var_state_filt1 = utils.test_filter(self)
        mean_state_filt2, var_state_filt2 = jax.jit(ktv.update)(
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred,
            x_meas=self.x_meas[1],
            mean_meas=self.mean_meas[1],
            wgt_meas=self.wgt_meas[1],
            var_meas=self.var_meas[1]
        )

        self.assertAlmostEqual(utils.rel_err(
            mean_state_filt1, mean_state_filt2), 0.0)
        self.assertAlmostEqual(utils.rel_err(
            var_state_filt1, var_state_filt2), 0.0)
        

    def test_filter(self):
        mean_state_past, var_state_past, mean_state_pred1, \
            var_state_pred1, mean_state_filt1, var_state_filt1 = utils.test_filter(self)
        mean_state_pred2, var_state_pred2, \
            mean_state_filt2, var_state_filt2 = jax.jit(ktv.filter)(
                mean_state_past=mean_state_past,
                var_state_past=var_state_past,
                mean_state=self.mean_state[1],
                wgt_state=self.wgt_state[0],
                var_state=self.var_state[1],
                x_meas=self.x_meas[1],
                mean_meas=self.mean_meas[1],
                wgt_meas=self.wgt_meas[1],
                var_meas=self.var_meas[1]
            )

        self.assertAlmostEqual(utils.rel_err(
            mean_state_pred1, mean_state_pred2), 0.0)
        self.assertAlmostEqual(utils.rel_err(
            var_state_pred1, var_state_pred2), 0.0)
        self.assertAlmostEqual(utils.rel_err(
            mean_state_filt1, mean_state_filt2), 0.0)
        self.assertAlmostEqual(utils.rel_err(
            var_state_filt1, var_state_filt2), 0.0)


    def test_filter2(self):
        mean_state_past, var_state_past, mean_state_pred1, var_state_pred1, \
            mean_state_filt1, var_state_filt1 = utils.test_filter2(self)
        
        mean_state_pred2, var_state_pred2, \
            mean_state_filt2, var_state_filt2 = jax.jit(ktv.filter)(
                mean_state_past=mean_state_past,
                var_state_past=var_state_past,
                mean_state=self.mean_state[2],
                wgt_state=self.wgt_state[1],
                var_state=self.var_state[2],
                x_meas=self.x_meas[2],
                mean_meas=self.mean_meas[2],
                wgt_meas=self.wgt_meas[2],
                var_meas=self.var_meas[2]
            )

        self.assertAlmostEqual(utils.rel_err(
            mean_state_pred1, mean_state_pred2), 0.0)
        self.assertAlmostEqual(utils.rel_err(
            var_state_pred1, var_state_pred2), 0.0)
        self.assertAlmostEqual(utils.rel_err(
            mean_state_filt1, mean_state_filt2), 0.0)
        self.assertAlmostEqual(utils.rel_err(
            var_state_filt1, var_state_filt2), 0.0)
        
        
    def test_smooth_mv(self):
        mean_state_next, var_state_next, mean_state_filt, var_state_filt, mean_state_pred, var_state_pred, \
            mean_state_smooth1, var_state_smooth1, _, _, _, _ = utils.test_smooth(self)
        
        mean_state_smooth2, var_state_smooth2 = jax.jit(ktv.smooth_mv)(
            mean_state_next=mean_state_next,
            var_state_next=var_state_next,
            mean_state_filt=mean_state_filt,
            var_state_filt=var_state_filt,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred,
            wgt_state=self.wgt_state[0]
        )

        self.assertAlmostEqual(utils.rel_err(
            mean_state_smooth1[0], mean_state_smooth2), 0.0)
        self.assertAlmostEqual(utils.rel_err(
            var_state_smooth1[0, :, 0, :].squeeze(), var_state_smooth2), 0.0)        

        
    def test_smooth_sim(self):
        _, _, mean_state_filt, var_state_filt, mean_state_pred, var_state_pred, \
            _, _, mean_state_sim1, V, _, _ = utils.test_smooth(self)

        mean_state_sim2, var_state_sim2 = jax.jit(ktv.smooth_sim)(
            x_state_next=self.x_state_next,
            mean_state_filt=mean_state_filt,
            var_state_filt=var_state_filt,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred,
            wgt_state=self.wgt_state[0]
        )
        # x_state_smooth1 = random.multivariate_normal(self.key, A.dot(self.x_state_next)+b, V)
        # x_state_smooth1 = ktv._state_sim(
        #    mean_state=A.dot(self.x_state_next)+b,
        #    var_state=V,
        #    z_state=self.z_state
        # )

        self.assertAlmostEqual(utils.rel_err(
            mean_state_sim1, mean_state_sim2), 0.0)
        self.assertAlmostEqual(utils.rel_err(V, var_state_sim2), 0.0)
        
        
    def test_smooth(self):
        mean_state_next, var_state_next, mean_state_filt, var_state_filt, mean_state_pred, var_state_pred, \
            mean_state_smooth1, var_state_smooth1, mean_state_sim1, V, _, _ = utils.test_smooth(self)

        # x_state_smooth1 = ktv._state_sim(
        #    mean_state=A.dot(self.x_state_next)+b,
        #    var_state=V,
        #    z_state=self.z_state
        # )
        # x_state_smooth1 = jax.random.multivariate_normal(self.key, A.dot(self.x_state_next)+b, V)
        mean_state_sim2, var_state_sim2, \
            mean_state_smooth2, var_state_smooth2 = jax.jit(ktv.smooth)(
                x_state_next=self.x_state_next,
                mean_state_next=mean_state_next,
                var_state_next=var_state_next,
                mean_state_filt=mean_state_filt,
                var_state_filt=var_state_filt,
                mean_state_pred=mean_state_pred,
                var_state_pred=var_state_pred,
                wgt_state=self.wgt_state[0],
            )
        self.assertAlmostEqual(utils.rel_err(
            mean_state_sim1, mean_state_sim2), 0.0)
        self.assertAlmostEqual(utils.rel_err(V, var_state_sim2), 0.0)
        self.assertAlmostEqual(utils.rel_err(
            mean_state_smooth1[0], mean_state_smooth2), 0.0)
        self.assertAlmostEqual(utils.rel_err(
            var_state_smooth1[0, :, 0, :].squeeze(), var_state_smooth2), 0.0)
        
        
    def test_smooth_cond(self):
        _, _, mean_state_filt, var_state_filt, mean_state_pred, \
            var_state_pred, _, _, _, V, A, b =  utils.test_smooth(self)
        # mean_state_sim1 = A.dot(self.x_state_next)+b
        # x_state_smooth1 = random.multivariate_normal(self.key, A.dot(self.x_state_next)+b, V)
        # x_state_smooth1 = ktv._state_sim(
        #    mean_state=A.dot(self.x_state_next)+b,
        #    var_state=V,
        #    z_state=self.z_state
        # )

        A2, b2, V2 = jax.jit(ktv.smooth_cond)(
            mean_state_filt=mean_state_filt,
            var_state_filt=var_state_filt,
            mean_state_pred=mean_state_pred,
            var_state_pred=var_state_pred,
            wgt_state=self.wgt_state[0]
        )

        self.assertAlmostEqual(utils.rel_err(A, A2), 0.0)
        self.assertAlmostEqual(utils.rel_err(b, b2), 0.0)
        self.assertAlmostEqual(utils.rel_err(V, V2), 0.0)
        


if __name__ == '__main__':
    unittest.main()
