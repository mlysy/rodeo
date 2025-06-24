import unittest
from scipy.integrate import odeint


from rodeo import solve_sim, solve_mv
from rodeo.interrogate import interrogate_rodeo
import utils

class TestFitzOdeint(unittest.TestCase):
    """
    Check whether rodeo and odeint gives approximately the same results.
    
    """
    setUp = utils.fitz_setup

    def test_fitz(self):
        det = odeint(self.fitz_odeint, self.x0, self.tseq, args=(self.theta,))
        sim = solve_sim(key=self.key, ode_fun=self.fitz_jax, ode_weight=self.W_block,
                        ode_init=self.x0_block, theta=self.theta,
                        t_min=self.t_min, t_max=self.t_max, n_steps=self.n_steps, 
                        prior_weight=self.prior_Q, prior_var=self.prior_R,
                        interrogate=interrogate_rodeo)
        m = solve_mv(key=self.key, ode_fun=self.fitz_jax, ode_weight=self.W_block,
                     ode_init=self.x0_block, theta=self.theta,
                     t_min=self.t_min, t_max=self.t_max, n_steps=self.n_steps, 
                     prior_weight=self.prior_Q, prior_var=self.prior_R,
                     interrogate=interrogate_rodeo)[0]
        self.assertLessEqual(utils.rel_err(sim[:, :, 0], det), 5.0)
        self.assertLessEqual(utils.rel_err(m[:, :, 0], det), 5.0)

if __name__ == '__main__':
    unittest.main()
