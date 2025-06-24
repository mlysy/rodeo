import unittest
import jax
import jax.numpy as jnp
import rodeo.kalmantv.square_root as sqrt_ktv
import utils


# --- square-root kalmantv.predict ---------------------------------------------------------


def comparison_check (self, mean1, mean2, var1, var2, places=9):
    """
    Helper function to assert that two values are almost equal with respect to a relative error tolerance.
    """
    self.assertAlmostEqual(utils.rel_err(mean1, mean2),0.0) 
    self.assertAlmostEqual(utils.rel_err(var1, jnp.dot(var2,var2.T)), 0.0, places) 


class TestKalmanTVGM(unittest.TestCase):
    """
    Test if square-root KalmanTV gives the same results as square-root Gaussian Markov process.
    """
    setUp = utils.kalman_setup
       
    def test_predict(self):
        mean_state_past, var_state_past, mean_state_pred1, var_state_pred1, _, _ = utils.test_filter(self)
        mean_state_pred2, var_state_pred2 = sqrt_ktv.predict(
            mean_state_past=mean_state_past,
            var_state_past=jnp.linalg.cholesky(var_state_past),
            mean_state=self.mean_state[1],
            wgt_state=self.wgt_state[0],
            var_state=jnp.linalg.cholesky(self.var_state[1])
        )
        
        comparison_check(self, mean_state_pred1, mean_state_pred2, var_state_pred1, var_state_pred2)

    def test_update(self):
        _, _, mean_state_pred, var_state_pred,  mean_state_filt1, var_state_filt1 = utils.test_filter(self)
        mean_state_filt2, var_state_filt2 = sqrt_ktv.update(
            mean_state_pred=mean_state_pred,
            var_state_pred=jnp.linalg.cholesky(var_state_pred),
            x_meas=self.x_meas[1],
            mean_meas=self.mean_meas[1],
            wgt_meas=self.wgt_meas[1],
            var_meas=jnp.linalg.cholesky(self.var_meas[1])
        )
        
        comparison_check(self, mean_state_filt1, mean_state_filt2, var_state_filt1, var_state_filt2)

    def test_filter(self):
        mean_state_past, var_state_past, mean_state_pred1, \
            var_state_pred1, mean_state_filt1, var_state_filt1 = utils.test_filter(self)
        mean_state_pred2, var_state_pred2, \
            mean_state_filt2, var_state_filt2 = sqrt_ktv.filter(
                mean_state_past=mean_state_past,
                var_state_past=jnp.linalg.cholesky(var_state_past),
                mean_state=self.mean_state[1],
                wgt_state=self.wgt_state[0],
                var_state=jnp.linalg.cholesky(self.var_state[1]),
                x_meas=self.x_meas[1],
                mean_meas=self.mean_meas[1],
                wgt_meas=self.wgt_meas[1],
                var_meas=jnp.linalg.cholesky(self.var_meas[1])
            )

        comparison_check(self, mean_state_pred1, mean_state_pred2, var_state_pred1, var_state_pred2)
        comparison_check(self, mean_state_filt1, mean_state_filt2, var_state_filt1, var_state_filt2)

    def test_filter2(self):
        mean_state_past, var_state_past, mean_state_pred1, var_state_pred1, \
            mean_state_filt1, var_state_filt1 = utils.test_filter2(self)
        
        mean_state_pred2, var_state_pred2, \
            mean_state_filt2, var_state_filt2 = sqrt_ktv.filter(
                mean_state_past=mean_state_past,
                var_state_past=jnp.linalg.cholesky(var_state_past),
                mean_state=self.mean_state[2],
                wgt_state=self.wgt_state[1],
                var_state=jnp.linalg.cholesky(self.var_state[2]),
                x_meas=self.x_meas[2],
                mean_meas=self.mean_meas[2],
                wgt_meas=self.wgt_meas[2],
                var_meas=jnp.linalg.cholesky(self.var_meas[2])
            )

        comparison_check(self, mean_state_pred1, mean_state_pred2, var_state_pred1, var_state_pred2)
        comparison_check(self, mean_state_filt1, mean_state_filt2, var_state_filt1, var_state_filt2)

    def test_smooth_mv(self):
        mean_state_next, var_state_next, mean_state_filt, var_state_filt, mean_state_pred, var_state_pred, \
            mean_state_smooth1, var_state_smooth1, _, _, _, _ = utils.test_smooth(self)
        
        mean_state_smooth2, var_state_smooth2 = sqrt_ktv.smooth_mv(
            mean_state_next=mean_state_next,
            var_state_next=jnp.linalg.cholesky(var_state_next),
            mean_state_filt=mean_state_filt,
            var_state_filt=jnp.linalg.cholesky(var_state_filt),
            mean_state_pred=mean_state_pred,
            var_state_pred=jnp.linalg.cholesky(var_state_pred),
            wgt_state=self.wgt_state[0],
            var_state=jnp.linalg.cholesky(self.var_state[1])
        )

        comparison_check(self, mean_state_smooth1[0], mean_state_smooth2, var_state_smooth1[0, :, 0, :].squeeze(), var_state_smooth2)

    def test_smooth_sim(self):
        _, _, mean_state_filt, var_state_filt, mean_state_pred, var_state_pred, \
            _, _, mean_state_sim1, V, _, _ = utils.test_smooth(self)

        mean_state_sim2, var_state_sim2 = sqrt_ktv.smooth_sim(
            x_state_next=self.x_state_next,
            mean_state_filt=mean_state_filt,
            var_state_filt=jnp.linalg.cholesky(var_state_filt),
            mean_state_pred=mean_state_pred,
            var_state_pred=jnp.linalg.cholesky(var_state_pred),
            wgt_state=self.wgt_state[0],
            var_state=jnp.linalg.cholesky(self.var_state[1])
        )
        
        comparison_check(self, mean_state_sim1, mean_state_sim2, V, var_state_sim2, places=2)

    def test_smooth(self):
        mean_state_next, var_state_next, mean_state_filt, var_state_filt, mean_state_pred, var_state_pred, \
            mean_state_smooth1, var_state_smooth1, mean_state_sim1, V, _, _ = utils.test_smooth(self)

        mean_state_sim2, var_state_sim2, mean_state_smooth2, var_state_smooth2 = sqrt_ktv.smooth(
            x_state_next=self.x_state_next,
            mean_state_next=mean_state_next,
            var_state_next=jnp.linalg.cholesky(var_state_next),
            mean_state_filt=mean_state_filt,
            var_state_filt=jnp.linalg.cholesky(var_state_filt),
            mean_state_pred=mean_state_pred,
            var_state_pred=jnp.linalg.cholesky(var_state_pred),
            wgt_state=self.wgt_state[0],
            var_state=jnp.linalg.cholesky(self.var_state[1])
        )
        
        comparison_check(self, mean_state_sim1, mean_state_sim2, V, var_state_sim2, places=2)
        comparison_check(self, mean_state_smooth1[0], mean_state_smooth2, var_state_smooth1[0, :, 0, :].squeeze(), var_state_smooth2)

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

        A2, b2, V2 = jax.jit(sqrt_ktv.smooth_cond)(
            mean_state_filt=mean_state_filt,
            var_state_filt=jnp.linalg.cholesky(var_state_filt),
            mean_state_pred=mean_state_pred,
            var_state_pred=jnp.linalg.cholesky(var_state_pred),
            wgt_state=self.wgt_state[0],
            var_state=jnp.linalg.cholesky(self.var_state[1])
        )

        self.assertAlmostEqual(utils.rel_err(A, A2), 0.0)
        comparison_check(self, b, b2, V, V2, places=2)

   
if __name__ == '__main__':
    unittest.main()
    
    
