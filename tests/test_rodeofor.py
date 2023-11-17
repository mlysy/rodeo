import unittest
import jax
from rodeo.solve import *
from rodeo.interrogate import *
import ode_block_solve_for as bfor
import utils
# from jax.config import config
# config.update("jax_enable_x64", True)

class TestrodeoFor(unittest.TestCase):
    """
    Test if lax scan version of rodeo gives the same results as for-loop version.

    """
    setUp = utils.fitz_setup 

    def test_interrogate_rodeo(self):
        wgt_meas1, mean_meas1, var_meas1 = interrogate_rodeo(
            key=self.key,
            ode_fun=self.fitz_jax,
            ode_weight=self.W_block,
            t=self.t,
            theta=self.theta,
            mean_state_pred=self.x0_block,
            var_state_pred=self.prior_R
        )
        # for
        wgt_meas2, mean_meas2, var_meas2 = bfor.interrogate_rodeo(
            key=self.key,
            ode_fun=self.fitz_jax,
            ode_weight=self.W_block,
            t=self.t,
            theta=self.theta,
            mean_state_pred=self.x0_block,
            var_state_pred=self.prior_R
        )
        
        self.assertAlmostEqual(utils.rel_err(wgt_meas1, wgt_meas2), 0.0)
        self.assertAlmostEqual(utils.rel_err(mean_meas1, mean_meas2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_meas1, var_meas2), 0.0)
    
    def test_interrogate_chkrebtii(self):
        wgt_meas1, mean_meas1, var_meas1 = interrogate_chkrebtii(
            key=self.key,
            ode_fun=self.fitz_jax,
            ode_weight=self.W_block,
            t=self.t,
            theta=self.theta,
            mean_state_pred=self.x0_block,
            var_state_pred=self.prior_R
        )
        # for
        wgt_meas2, mean_meas2, var_meas2 = bfor.interrogate_chkrebtii(
            key=self.key,
            ode_fun=self.fitz_jax,
            ode_weight=self.W_block,
            t=self.t,
            theta=self.theta,
            mean_state_pred=self.x0_block,
            var_state_pred=self.prior_R
        )
        
        self.assertAlmostEqual(utils.rel_err(wgt_meas1, wgt_meas2), 0.0)
        self.assertAlmostEqual(utils.rel_err(mean_meas1, mean_meas2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_meas1, var_meas2), 0.0)

    def test_interrogate_kramer(self):
        wgt_meas1, mean_meas1, var_meas1 = interrogate_kramer(
            key=self.key,
            ode_fun=self.fitz_jax,
            ode_weight=self.W_block,
            t=self.t,
            theta=self.theta,
            mean_state_pred=self.x0_block,
            var_state_pred=self.prior_R
        )
        # for
        wgt_meas2, mean_meas2, var_meas2 = bfor.interrogate_kramer(
            key=self.key,
            ode_fun=self.fitz_jax,
            ode_weight=self.W_block,
            t=self.t,
            theta=self.theta,
            mean_state_pred=self.x0_block,
            var_state_pred=self.prior_R
        )
        
        self.assertAlmostEqual(utils.rel_err(wgt_meas1, wgt_meas2), 0.0)
        self.assertAlmostEqual(utils.rel_err(mean_meas1, mean_meas2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var_meas1, var_meas2), 0.0)

    def test_solve_sim(self):
        sim1 = solve_sim(key=self.key, ode_fun=self.fitz_jax, ode_weight=self.W_block,
                         ode_init=self.x0_block, theta=self.theta,
                         t_min=self.t_min, t_max=self.t_max, n_steps=self.n_steps, 
                         prior_weight=self.prior_Q, prior_var=self.prior_R,
                         interrogate=interrogate_rodeo)
        # for
        sim2 = bfor.solve_sim(key=self.key, ode_fun=self.fitz_jax, ode_weight=self.W_block,
                              ode_init=self.x0_block, theta=self.theta,
                              t_min=self.t_min, t_max=self.t_max, n_steps=self.n_steps, 
                              prior_weight=self.prior_Q, prior_var=self.prior_R,
                              interrogate=interrogate_rodeo)
        
        self.assertAlmostEqual(utils.rel_err(sim1, sim2), 0.0)
    
    def test_solve_mv(self):
        mu1, var1 = solve_mv(key=self.key, ode_fun=self.fitz_jax, ode_weight=self.W_block,
                             ode_init=self.x0_block, theta=self.theta,
                             t_min=self.t_min, t_max=self.t_max, n_steps=self.n_steps, 
                             prior_weight=self.prior_Q, prior_var=self.prior_R,
                             interrogate=interrogate_rodeo)
        # for
        mu2, var2 = bfor.solve_mv(key=self.key, ode_fun=self.fitz_jax, ode_weight=self.W_block,
                                  ode_init=self.x0_block, theta=self.theta,
                                  t_min=self.t_min, t_max=self.t_max, n_steps=self.n_steps, 
                                  prior_weight=self.prior_Q, prior_var=self.prior_R,
                                  interrogate=interrogate_rodeo)
        self.assertAlmostEqual(utils.rel_err(mu1, mu2), 0.0)
        self.assertAlmostEqual(utils.rel_err(var1, var2), 0.0)
    
    # def test_solve(self):
    #     sim1, mu1, var1 = solve(key=self.key, ode_fun=self.fitz_jax, ode_weight=self.W_block,
    #                             ode_init=self.x0_block, theta=self.theta,
    #                             t_min=self.t_min, t_max=self.t_max, n_steps=self.n_steps, 
    #                             prior_weight=self.prior_Q, prior_var=self.prior_R,
    #                             interrogate=interrogate_kramer)
    #     # for
    #     sim2, mu2, var2 = bfor.solve(key=self.key, ode_fun=self.fitz_jax, ode_weight=self.W_block,
    #                                  ode_init=self.x0_block, theta=self.theta,
    #                                  t_min=self.t_min, t_max=self.t_max, n_steps=self.n_steps, 
    #                                  prior_weight=self.prior_Q, prior_var=self.prior_R,
    #                                  interrogate=interrogate_kramer)
    #     self.assertAlmostEqual(utils.rel_err(mu1, mu2), 0.0)
    #     self.assertAlmostEqual(utils.rel_err(var1, var2), 0.0)
    #     self.assertAlmostEqual(utils.rel_err(sim1, sim2), 0.0, places=7)

if __name__ == '__main__':
    unittest.main()
