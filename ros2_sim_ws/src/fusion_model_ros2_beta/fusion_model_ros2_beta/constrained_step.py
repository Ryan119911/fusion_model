"""Small constrained LM subproblem; nonlinear feasibility is checked later."""
import numpy as np


def feasible_quadratic_step(system, gradient, margins, constraint_jacobian, *, max_step=None):
    from scipy.optimize import minimize
    system=np.asarray(system,dtype=float)
    gradient=np.asarray(gradient,dtype=float)
    margins=np.asarray(margins,dtype=float)
    jac=np.asarray(constraint_jacobian,dtype=float)
    n=len(gradient)
    if max_step is not None and (not np.isfinite(max_step) or max_step <= 0):
        raise ValueError('invalid constrained LM trust radius')
    if (system.shape!=(n,n) or jac.shape!=(len(margins),n)
            or not all(np.all(np.isfinite(v)) for v in (system,gradient,margins,jac))):
        raise ValueError('invalid constrained LM subproblem')
    # Scale variables using LM's diagonal to improve numerical conditioning.
    scale=1/np.sqrt(np.maximum(np.diag(system),1e-10))
    matrix=system*scale[:,None]*scale[None,:]
    g=gradient*scale
    a=jac*scale[None,:]
    # Exact forward audit allows tiny round-off at the boundary; no expansion
    # of the physical domain is allowed in the final nonlinear check.
    b=np.maximum(margins,0.)
    def objective(v):
        return .5*float(v@matrix@v)+float(g@v),matrix@v+g
    fit=minimize(objective,np.zeros(n),jac=True,method='SLSQP',
        bounds=None if max_step is None else [(-max_step/s,max_step/s) for s in scale],
        constraints=[dict(type='ineq',fun=lambda v:b+a@v,jac=lambda v:a)],
        options=dict(maxiter=300,ftol=1e-9))
    if not fit.success or not np.all(np.isfinite(fit.x)) or np.min(b+a@fit.x)<-1e-7:
        raise ValueError('constrained LM subproblem did not converge: '+str(fit.message))
    return fit.x*scale
