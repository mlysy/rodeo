import unittest
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as random
from rodeo.ode import *
import utils
# from jax.config import config
# config.update("jax_enable_x64", True)
    
class TestrodeoJit(unittest.TestCase):
    """
    Check whether jit and unjitted gives the same result.
    
    """
    setUp = utils.fitz_setup

    def test_interrogate_rodeo(self):
        # without jit
        wgt_meas1, mean_meas1, var_meas1 = interrogate_rodeo(
            self.key,
            self.fitz_jax,
            ode_weight=self.W_block,
            t=self.t,
            theta=self.theta,
            mean_state_pred=self.x0_block,
            var_state_pred=self.prior_R
        )
        # with jit
        rodeo_jit = jax.jit(interrogate_rodeo, static_argnums=(1,))
        wgt_meas2, mean_meas2, var_meas2 = rodeo_jit(
            self.key,
            self.fitz_jax,
            ode_weight=self.W_block,
            t=self.t,
            theta=self.theta,
            mean_state_pred=self.x0_block,
            var_state_pred=self.prior_R
        )
        # objective function for gradient
        def obj_fun(theta):
            return jnp.mean(
                interrogate_rodeo(
                    self.key, self.fitz_jax,
                    ode_weight=self.W_block, t=self.t, theta=theta,
                    mean_state_pred=self.x0_block,
                    var_state_pred=self.prior_R)[0])
        # grad without jit
        grad1 = jax.grad(obj_fun)(self.theta)
        # grad with jit
        grad2 = jax.jit(jax.grad(obj_fun))(self.theta)
        self.assertAlmostEqual(utils.rel_err(wgt_meas1, wgt_meas2), 0.0)
        self.assertAlmostEqual(utils.rel_err(mean_meas1, mean_meas2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_meas1, var_meas2), 0.0)
        self.assertAlmostEqual(utils.rel_err(grad1, grad2), 0.0)

    def test_interrogate_chkrebtii(self):
        # without jit
        wgt_meas1, mean_meas1, var_meas1 = interrogate_chkrebtii(
            self.key,
            self.fitz_jax,
            ode_weight=self.W_block,
            t=self.t,
            theta=self.theta,
            mean_state_pred=self.x0_block,
            var_state_pred=self.prior_R
        )
        # with jit
        rodeo_jit = jax.jit(interrogate_chkrebtii, static_argnums=(1,))
        wgt_meas2, mean_meas2, var_meas2 = rodeo_jit(
            self.key,
            self.fitz_jax,
            ode_weight=self.W_block,
            t=self.t,
            theta=self.theta,
            mean_state_pred=self.x0_block,
            var_state_pred=self.prior_R
        )
        # objective function for gradient
        def obj_fun(theta):
            return jnp.mean(
                interrogate_chkrebtii(
                    self.key, self.fitz_jax,
                    ode_weight=self.W_block, t=self.t, theta=theta,
                    mean_state_pred=self.x0_block,
                    var_state_pred=self.prior_R)[0])
        # grad without jit
        grad1 = jax.grad(obj_fun)(self.theta)
        # grad with jit
        grad2 = jax.jit(jax.grad(obj_fun))(self.theta)
        self.assertAlmostEqual(utils.rel_err(wgt_meas1, wgt_meas2), 0.0)
        self.assertAlmostEqual(utils.rel_err(mean_meas1, mean_meas2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_meas1, var_meas2), 0.0)
        self.assertAlmostEqual(utils.rel_err(grad1, grad2), 0.0)

    def test_interrogate_schober(self):
        # without jit
        wgt_meas1, mean_meas1, var_meas1 = interrogate_schober(
            self.key,
            self.fitz_jax,
            ode_weight=self.W_block,
            t=self.t,
            theta=self.theta,
            mean_state_pred=self.x0_block,
            var_state_pred=self.prior_R
        )
        # with jit
        rodeo_jit = jax.jit(interrogate_schober, static_argnums=(1,))
        wgt_meas2, mean_meas2, var_meas2 = rodeo_jit(
            self.key,
            self.fitz_jax,
            ode_weight=self.W_block,
            t=self.t,
            theta=self.theta,
            mean_state_pred=self.x0_block,
            var_state_pred=self.prior_R
        )
        # objective function for gradient
        def obj_fun(theta):
            return jnp.mean(
                interrogate_schober(
                    self.key, self.fitz_jax,
                    t=self.t, theta=theta,
                    ode_weight=self.W_block,
                    mean_state_pred=self.x0_block,
                    var_state_pred=self.prior_R)[0])
        # grad without jit
        grad1 = jax.grad(obj_fun)(self.theta)
        # grad with jit
        grad2 = jax.jit(jax.grad(obj_fun))(self.theta)
        self.assertAlmostEqual(utils.rel_err(wgt_meas1, wgt_meas2), 0.0)
        self.assertAlmostEqual(utils.rel_err(mean_meas1, mean_meas2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_meas1, var_meas2), 0.0)
        self.assertAlmostEqual(utils.rel_err(grad1, grad2), 0.0)

    def test_mv(self):
        # without jit
        mu1, var1 = solve_mv(key=self.key, ode_fun=self.fitz_jax, ode_weight=self.W_block,
                             ode_init=self.x0_block, theta=self.theta,
                             t_min=self.t_min, t_max=self.t_max, n_steps=self.n_steps, 
                             prior_weight=self.prior_Q, prior_var=self.prior_R,
                             interrogate=interrogate_rodeo)
        # with jit
        mv_jit = jax.jit(solve_mv, static_argnums=(1, 6, 7))
        mu2, var2 = mv_jit(key=self.key, ode_fun=self.fitz_jax, ode_weight=self.W_block,
                            ode_init=self.x0_block, theta=self.theta,
                            t_min=self.t_min, t_max=self.t_max, n_steps=self.n_steps, 
                            prior_weight=self.prior_Q, prior_var=self.prior_R,
                            interrogate=interrogate_rodeo)
        # objective function for gradient
        def obj_fun(theta):
            return jnp.mean(
                solve_mv(key=self.key, ode_fun=self.fitz_jax, ode_weight=self.W_block,
                        ode_init=self.x0_block, theta=self.theta,
                        t_min=self.t_min, t_max=self.t_max, n_steps=self.n_steps, 
                        prior_weight=self.prior_Q, prior_var=self.prior_R,
                        interrogate=interrogate_rodeo)[0])
        # grad without jit
        grad1 = jax.grad(obj_fun)(self.theta)
        # grad with jit
        grad2 = jax.jit(jax.grad(obj_fun))(self.theta)
        self.assertAlmostEqual(utils.rel_err(mu1, mu2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var1, var2), 0.0)
        self.assertAlmostEqual(utils.rel_err(grad1, grad2), 0.0)
    
    def test_sim(self):
        # without jit
        sim1 = solve_sim(key=self.key, ode_fun=self.fitz_jax, ode_weight=self.W_block,
                        ode_init=self.x0_block, theta=self.theta,
                        t_min=self.t_min, t_max=self.t_max, n_steps=self.n_steps, 
                        prior_weight=self.prior_Q, prior_var=self.prior_R,
                        interrogate=interrogate_rodeo)
        # with jit
        sim_jit = jax.jit(solve_sim, static_argnums=(1, 6, 7))
        sim2 = sim_jit(key=self.key, ode_fun=self.fitz_jax, ode_weight=self.W_block,
                       ode_init=self.x0_block, theta=self.theta,
                       t_min=self.t_min, t_max=self.t_max, n_steps=self.n_steps, 
                       prior_weight=self.prior_Q, prior_var=self.prior_R,
                       interrogate=interrogate_rodeo)
        # objective function for gradient
        def obj_fun(theta):
            return jnp.mean(
                solve_sim(key=self.key, ode_fun=self.fitz_jax, ode_weight=self.W_block,
                        ode_init=self.x0_block, theta=self.theta,
                        t_min=self.t_min, t_max=self.t_max, n_steps=self.n_steps, 
                        prior_weight=self.prior_Q, prior_var=self.prior_R,
                        interrogate=interrogate_rodeo)[0])
        # grad without jit
        grad1 = jax.grad(obj_fun)(self.theta)
        # grad with jit
        grad2 = jax.jit(jax.grad(obj_fun))(self.theta)
        self.assertAlmostEqual(utils.rel_err(sim1, sim2), 0.0)
        self.assertAlmostEqual(utils.rel_err(grad1, grad2), 0.0)
    
    # def test_solve(self):
    #     # without jit
    #     sim1, mu1, var1 = \
    #         solve(key=self.key, ode_fun=self.fitz_jax, ode_weight=self.W_block,
    #               ode_init=self.x0_block, theta=self.theta,
    #               t_min=self.t_min, t_max=self.t_max, n_steps=self.n_steps, 
    #               prior_weight=self.prior_Q, prior_var=self.prior_R,
    #               interrogate=interrogate_rodeo)
    #     # with jit
    #     solve_jit = jax.jit(solve, static_argnums=(1, 6, 7))
    #     sim2, mu2, var2 = \
    #         solve_jit(key=self.key, ode_fun=self.fitz_jax, ode_weight=self.W_block,
    #                   ode_init=self.x0_block, theta=self.theta,
    #                   t_min=self.t_min, t_max=self.t_max, n_steps=self.n_steps, 
    #                   prior_weight=self.prior_Q, prior_var=self.prior_R,
    #                   interrogate=interrogate_rodeo)
    #     # objective function for gradient
    #     def obj_fun(theta):
    #         return jnp.mean(
    #             solve(key=self.key, ode_fun=self.fitz_jax, ode_weight=self.W_block,
    #                   ode_init=self.x0_block, theta=self.theta,
    #                   t_min=self.t_min, t_max=self.t_max, n_steps=self.n_steps, 
    #                   prior_weight=self.prior_Q, prior_var=self.prior_R,
    #                   interrogate=interrogate_rodeo)[0])
    #     # grad without jit
    #     grad1 = jax.grad(obj_fun)(self.theta)
    #     # grad with jit
    #     grad2 = jax.jit(jax.grad(obj_fun))(self.theta)
    #     self.assertAlmostEqual(utils.rel_err(sim1, sim2), 0.0)
    #     self.assertAlmostEqual(utils.rel_err(mu1, mu2), 0.0)
    #     self.assertAlmostEqual(utils.rel_err(var1, var2), 0.0)
    #     self.assertAlmostEqual(utils.rel_err(grad1, grad2), 0.0)

if __name__ == '__main__':
    unittest.main()
