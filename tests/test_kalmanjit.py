import unittest
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
import rodeo.kalmantv as ktv
import utils
# from jax.config import config
# config.update("jax_enable_x64", True)

class TestKalmanTVJit(unittest.TestCase):
    """
    Check whether jit and unjitted gives the same result.
    """
    setUp = utils.kalman_setup
    
    def test_predict(self):
        self.key, *subkeys = random.split(self.key, 3)
        mean_state_past = random.normal(subkeys[0], (self.n_state,))
        var_state_past = random.normal(subkeys[1], (self.n_state, self.n_state))
        var_state_past = var_state_past.dot(var_state_past.T)
        # without jit
        mean_state_pred1, var_state_pred1 = \
             ktv.predict(mean_state_past, var_state_past,
                         self.mean_state[0], self.trans_state[0], self.var_state[0])
        # with jit
        predict_jit = jax.jit(ktv.predict)
        mean_state_pred2, var_state_pred2 = \
            predict_jit(mean_state_past, var_state_past,
                        self.mean_state[0], self.trans_state[0], self.var_state[0])
        # objective function for gradient
        def obj_fun(mean_state_past, var_state_past, 
                    mean_state, trans_state, var_state): 
            return jnp.mean(
                ktv.predict(mean_state_past, var_state_past,
                            mean_state, trans_state, var_state)[0])
        # grad without jit
        grad1 = jax.grad(obj_fun)(
            mean_state_past, var_state_past,
            self.mean_state[0], self.trans_state[0], self.var_state[0])
        # grad with jit
        grad2 = jax.jit(jax.grad(obj_fun))(
            mean_state_past, var_state_past,
            self.mean_state[0], self.trans_state[0], self.var_state[0])
        self.assertAlmostEqual(utils.rel_err(mean_state_pred1, mean_state_pred2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_pred1, var_state_pred2), 0.0)
        self.assertAlmostEqual(utils.rel_err(grad1, grad2), 0.0)

    def test_update(self):
        self.key, *subkeys = random.split(self.key, 3)
        mean_state_pred = random.normal(subkeys[0], (self.n_state,))
        var_state_pred = random.normal(subkeys[1], (self.n_state, self.n_state))
        var_state_pred = var_state_pred.dot(var_state_pred.T)
        # without jit
        mean_state_filt1, var_state_filt1 = \
             ktv.update(mean_state_pred, var_state_pred, self.trans_meas[0],
                        self.x_meas[0], self.mean_meas[0], self.trans_meas[0], self.var_meas[0])
        # with jit
        update_jit = jax.jit(ktv.update)
        mean_state_filt2, var_state_filt2 = \
            update_jit(mean_state_pred, var_state_pred, self.trans_meas[0],
                       self.x_meas[0], self.mean_meas[0], self.trans_meas[0], self.var_meas[0])
        # objective function for gradient
        def obj_fun(mean_state_pred, var_state_pred,
                    x_meas, mean_meas, trans_meas, var_meas):
            return jnp.mean(
                ktv.update(mean_state_pred, var_state_pred, self.trans_meas[0],
                           x_meas, mean_meas, trans_meas, var_meas)[0])
        # grad without jit
        grad1 = jax.grad(obj_fun)(
            mean_state_pred, var_state_pred,
            self.x_meas[0], self.mean_meas[0], self.trans_meas[0], self.var_meas[0])
        # grad with jit
        grad2 = jax.jit(jax.grad(obj_fun))(
            mean_state_pred, var_state_pred,
            self.x_meas[0], self.mean_meas[0], self.trans_meas[0], self.var_meas[0])
        self.assertAlmostEqual(utils.rel_err(mean_state_filt1, mean_state_filt2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_filt1, var_state_filt2), 0.0)
        self.assertAlmostEqual(utils.rel_err(grad1, grad2), 0.0)

    def test_filter(self):
        self.key, *subkeys = random.split(self.key, 3)
        mean_state_past = random.normal(subkeys[0], (self.n_state,))
        var_state_past = random.normal(subkeys[1], (self.n_state, self.n_state))
        var_state_past = var_state_past.dot(var_state_past.T)
        # without jit
        mean_state_pred1, var_state_pred1, mean_state_filt1, var_state_filt1 = \
            ktv.filter(mean_state_past, var_state_past,
                       self.mean_state[0], self.trans_state[0], self.var_state[0],
                       self.x_meas[0], self.mean_meas[0], self.trans_meas[0], self.var_meas[0])
        # with jit
        filter_jit = jax.jit(ktv.filter)
        mean_state_pred2, var_state_pred2, mean_state_filt2, var_state_filt2 = \
            filter_jit(mean_state_past, var_state_past,
                       self.mean_state[0], self.trans_state[0], self.var_state[0],
                       self.x_meas[0], self.mean_meas[0], self.trans_meas[0], self.var_meas[0])
        # objective function for gradient
        def obj_fun(mean_state_past, var_state_past,
                    mean_state, trans_state, var_state,
                    x_meas, mean_meas, trans_meas, var_meas):
            return jnp.mean(
                ktv.filter(mean_state_past, var_state_past,
                           mean_state, trans_state, var_state,
                           x_meas, mean_meas, trans_meas, var_meas)[0])
        # grad without jit
        grad1 = jax.grad(obj_fun)(
            mean_state_past, var_state_past,
            self.mean_state[0], self.trans_state[0], self.var_state[0],
            self.x_meas[0], self.mean_meas[0], self.trans_meas[0], self.var_meas[0])
        # grad with jit
        grad2 = jax.jit(jax.grad(obj_fun))(
            mean_state_past, var_state_past,
            self.mean_state[0], self.trans_state[0], self.var_state[0],
            self.x_meas[0], self.mean_meas[0], self.trans_meas[0], self.var_meas[0])
        self.assertAlmostEqual(utils.rel_err(mean_state_pred1, mean_state_pred2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_pred1, var_state_pred2), 0.0)
        self.assertAlmostEqual(utils.rel_err(mean_state_filt1, mean_state_filt2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_filt1, var_state_filt2), 0.0)
        self.assertAlmostEqual(utils.rel_err(grad1, grad2), 0.0)
    

    def test_smooth_mv(self):
        self.key, *subkeys = random.split(self.key, 7)
        mean_state_next = random.normal(subkeys[0], (self.n_state,))
        var_state_next = random.normal(subkeys[1], (self.n_state, self.n_state))
        var_state_next = var_state_next.dot(var_state_next.T)
        mean_state_filt = random.normal(subkeys[2], (self.n_state,))
        var_state_filt = random.normal(subkeys[3], (self.n_state, self.n_state))
        var_state_filt = var_state_filt.dot(var_state_filt.T)
        mean_state_pred = random.normal(subkeys[4], (self.n_state,))
        var_state_pred = random.normal(subkeys[5], (self.n_state, self.n_state))
        var_state_pred = var_state_pred.dot(var_state_pred.T)
        # without jit
        mean_state_smooth1, var_state_smooth1 = \
            ktv.smooth_mv(mean_state_next, var_state_next,
                          mean_state_filt, var_state_filt,
                          mean_state_pred, var_state_pred,
                          self.trans_state[0])
        # with jit
        mv_jit = jax.jit(ktv.smooth_mv)
        mean_state_smooth2, var_state_smooth2 = \
            mv_jit(mean_state_next, var_state_next,
                   mean_state_filt, var_state_filt,
                   mean_state_pred, var_state_pred,
                   self.trans_state[0])
        # objective function for gradient
        def obj_fun(mean_state_next, var_state_next,
                    mean_state_filt, var_state_filt,
                    mean_state_pred, var_state_pred,
                    trans_state):
            return jnp.mean(
                ktv.smooth_mv(mean_state_next, var_state_next,
                              mean_state_filt, var_state_filt,
                              mean_state_pred, var_state_pred,
                              trans_state)[0])
        # grad without jit
        grad1 = jax.grad(obj_fun)(
            mean_state_next, var_state_next,
            mean_state_filt, var_state_filt,
            mean_state_pred, var_state_pred,
            self.trans_state[0])
        # grad with jit
        grad2 = jax.jit(jax.grad(obj_fun))(
            mean_state_next, var_state_next,
            mean_state_filt, var_state_filt,
            mean_state_pred, var_state_pred,
            self.trans_state[0])
        self.assertAlmostEqual(utils.rel_err(mean_state_smooth1, mean_state_smooth2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_smooth1, var_state_smooth2), 0.0)
        self.assertAlmostEqual(utils.rel_err(grad1, grad2), 0.0)

    def test_smooth_sim(self):
        self.key, *subkeys = random.split(self.key, 4)
        mean_state_past = random.normal(subkeys[0], (self.n_state,))
        var_state_past = random.normal(subkeys[1], (self.n_state, self.n_state))
        var_state_past = var_state_past.dot(var_state_past.T)
        x_state_next = random.normal(subkeys[2], (self.n_state,))
        # without jit
        mean_state_pred1, var_state_pred1, mean_state_filt1, var_state_filt1 = \
            ktv.filter(mean_state_past, var_state_past,
                       self.mean_state[0], self.trans_state[0], self.var_state[0],
                       self.x_meas[0], self.mean_meas[0], self.trans_meas[0], self.var_meas[0])
        mean_state_sim1, var_state_sim1 = \
            ktv.smooth_sim(x_state_next,
                           mean_state_filt1, var_state_filt1,
                           mean_state_pred1, var_state_pred1,
                           self.trans_state[0])
        # with jit
        filter_jit = jax.jit(ktv.filter)
        mean_state_pred2, var_state_pred2, mean_state_filt2, var_state_filt2 = \
            filter_jit(mean_state_past, var_state_past,
                       self.mean_state[0], self.trans_state[0], self.var_state[0],
                       self.x_meas[0], self.mean_meas[0], self.trans_meas[0], self.var_meas[0])
        sim_jit = jax.jit(ktv.smooth_sim)
        mean_state_sim2, var_state_sim2 = \
            sim_jit(x_state_next,
                    mean_state_filt2, var_state_filt2,
                    mean_state_pred2, var_state_pred2,
                    self.trans_state[0])
        # objective function for gradient
        def obj_fun(x_state_next,
                    mean_state_filt, var_state_filt,
                    mean_state_pred, var_state_pred,
                    trans_state):
            return jnp.mean(
                ktv.smooth_sim(x_state_next,
                               mean_state_filt, var_state_filt,
                               mean_state_pred, var_state_pred,
                               trans_state)[0])
        # grad without jit
        grad1 = jax.grad(obj_fun, argnums=1)(
            x_state_next,
            mean_state_filt1, var_state_filt1,
            mean_state_pred1, var_state_pred1,
            self.trans_state[0])
        # grad with jit
        grad2 = jax.jit(jax.grad(obj_fun, argnums=1))(
            x_state_next,
            mean_state_filt2, var_state_filt2,
            mean_state_pred2, var_state_pred2,
            self.trans_state[0])
        self.assertAlmostEqual(utils.rel_err(mean_state_sim1, mean_state_sim2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_sim1, var_state_sim2), 0.0)
        self.assertAlmostEqual(utils.rel_err(grad1, grad2), 0.0)

    def test_smooth(self):
        self.key, *subkeys = random.split(self.key, 6)
        mean_state_past = random.normal(subkeys[0], (self.n_state,))
        var_state_past = random.normal(subkeys[1], (self.n_state, self.n_state))
        var_state_past = var_state_past.dot(var_state_past.T)
        x_state_next = random.normal(subkeys[2], (self.n_state,))
        mean_state_next = random.normal(subkeys[3], (self.n_state,))
        var_state_next = random.normal(subkeys[4], (self.n_state, self.n_state))
        var_state_next = var_state_next.dot(var_state_next.T)
        # without jit
        mean_state_pred1, var_state_pred1, mean_state_filt1, var_state_filt1 = \
            ktv.filter(mean_state_past, var_state_past,
                       self.mean_state[0], self.trans_state[0], self.var_state[0],
                       self.x_meas[0], self.mean_meas[0], self.trans_meas[0], self.var_meas[0])
        mean_state_sim1, var_state_sim1, mean_state_smooth1, var_state_smooth1= \
            ktv.smooth(x_state_next, 
                       mean_state_next, var_state_next,
                       mean_state_filt1, var_state_filt1,
                       mean_state_pred1, var_state_pred1,
                       self.trans_state[0])
        # with jit
        filter_jit = jax.jit(ktv.filter)
        mean_state_pred2, var_state_pred2, mean_state_filt2, var_state_filt2 = \
            filter_jit(mean_state_past, var_state_past,
                       self.mean_state[0], self.trans_state[0], self.var_state[0],
                       self.x_meas[0], self.mean_meas[0], self.trans_meas[0], self.var_meas[0])
        smooth_jit = jax.jit(ktv.smooth)
        mean_state_sim2, var_state_sim2, mean_state_smooth2, var_state_smooth2 = \
            smooth_jit(x_state_next,
                       mean_state_next, var_state_next,
                       mean_state_filt2, var_state_filt2,
                       mean_state_pred2, var_state_pred2,
                       self.trans_state[0])
        # objective function for gradient
        def obj_fun(x_state_next,
                    mean_state_next, var_state_next,
                    mean_state_filt, var_state_filt,
                    mean_state_pred, var_state_pred,
                    trans_state):
            return jnp.mean(
                ktv.smooth(x_state_next,
                           mean_state_next, var_state_next,
                           mean_state_filt, var_state_filt,
                           mean_state_pred, var_state_pred,
                           trans_state)[0])
        # grad without jit
        grad1 = jax.grad(obj_fun)(
            x_state_next,
            mean_state_next, var_state_next,
            mean_state_filt1, var_state_filt1,
            mean_state_pred1, var_state_pred1,
            self.trans_state[0])
        # grad with jit
        grad2 = jax.jit(jax.grad(obj_fun))(
            x_state_next,
            mean_state_next, var_state_next,
            mean_state_filt2, var_state_filt2,
            mean_state_pred2, var_state_pred2,
            self.trans_state[0])
        self.assertAlmostEqual(utils.rel_err(mean_state_smooth1, mean_state_smooth2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_smooth1, var_state_smooth2), 0.0)
        self.assertAlmostEqual(utils.rel_err(mean_state_sim1, mean_state_sim2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_state_sim1, var_state_sim2), 0.0)
        self.assertAlmostEqual(utils.rel_err(grad1, grad2), 0.0)

    def test_forecast(self):
        self.key, *subkeys = random.split(self.key, 3)
        mean_state_pred = random.normal(subkeys[0], (self.n_state,))
        var_state_pred = random.normal(subkeys[1], (self.n_state, self.n_state))
        var_state_pred = var_state_pred.dot(var_state_pred.T)
        # without jit
        mean_fore1, var_fore1 = \
            ktv.forecast(mean_state_pred, var_state_pred,
                         self.mean_meas[0], self.trans_meas[0], self.var_meas[0])
        # with jit
        fore_jit = jax.jit(ktv.forecast)
        mean_fore2, var_fore2 = \
            fore_jit(mean_state_pred, var_state_pred,
                     self.mean_meas[0], self.trans_meas[0], self.var_meas[0])
        # objective function for gradient
        def obj_fun(mean_state_pred, var_state_pred,
                    mean_meas, trans_meas, var_meas):
            return jnp.mean(
                ktv.forecast(mean_state_pred, var_state_pred,
                             mean_meas, trans_meas, var_meas)[0])
        # grad without jit
        grad1 = jax.grad(obj_fun)(
            mean_state_pred, var_state_pred,
            self.mean_meas[0], self.trans_meas[0], self.var_meas[0])
        # grad with jit
        grad2 = jax.jit(jax.grad(obj_fun))(
            mean_state_pred, var_state_pred,
            self.mean_meas[0], self.trans_meas[0], self.var_meas[0])
        self.assertAlmostEqual(utils.rel_err(mean_fore1, mean_fore2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_fore1, var_fore2), 0.0)
        self.assertAlmostEqual(utils.rel_err(grad1, grad2), 0.0)

if __name__ == '__main__':
    unittest.main()
