# TODO List

## Major Items

Needed for JSS submission.

- [ ] Change all instances of `rodeo-jax` to `rodeo`.

- [ ] Put a note in the `README` that the previous version of this package is at `github.com/mlysy/rodeo-legacy`.  

- [ ] Use the `src` package layout.  Same as **pfjax**.  Might need to wrangle with `setup.cfg` and `pyproject.toml`.

- [ ] Change `ibm_init.py` to `ibm.py`.  Also, please add some module-level documentation for this (i.e., what is the IBM model, how it's a CAR, it's analytic formulas, etc).

- [ ] More generally, please think carefully about the names of things.  Should it be `ode_solve.py`, or maybe just `ode.py`?  Users could then call things with e.g., `rodeo.ode.solve_mv()`.  Or should it be `rodeo.ode_solve_mv()`?

- [ ] Check that all dependencies are necessary, e.g., `operator`.

## Minor Items

Maybe not necessary for submission, but definitely better coding practice.  Maybe can be done after submission, during the review process...

- [ ] Consider writing documentation using **jupytext** and **myst_nb**.  Please see **projplot** or **pjfax**.  In particular, you're basically writing documentation in Jupyter Notebooks -- complete with code exectution, figures, etc -- instead of all that by hand using ReST.

- [ ] Consider using formal "typing" for function argument/return types.  A good example of this is **blackjax**.  Also, please see  https://github.com/google/jaxtyping.  I think **sphinx-autodoc** knows how to handle this.

- [ ] Consider adding more examples to the documentation.  You have them already in the `examples` folder!  Please see what we did in **pfjax** for separating between user-facing and developer documentation (only the former gets rendered on readthedocs).

- [ ] Put a note in the `README`  of `rodeo-legacy` that the newer version of the package is at `github.com/mlysy/rodeo-legacy`.
