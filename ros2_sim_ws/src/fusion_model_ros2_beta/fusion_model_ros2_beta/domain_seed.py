"""Constrained gamma initialization, followed by full model inversion.

Never clips exported poses or changes the forward renderer. Failure is explicit.
"""
import numpy as np


def fit_gamma_seed(matrix, sweep, heading, original, *, margin=1e-3,
                   return_node_logits=False):
    from scipy.optimize import minimize
    matrix, sweep = np.asarray(matrix, float), np.asarray(sweep, float)
    original = np.asarray(original, float)
    heading = np.unwrap(np.asarray(heading, float))
    x0 = np.linalg.lstsq(matrix, np.arctanh(np.clip(original/np.pi,-.999,.999)), rcond=None)[0]
    candidates = []
    for branch in (-1, 0, 1):
        centre = heading+branch*2*np.pi
        lower,upper = centre-np.pi/6+margin,centre+np.pi/6-margin
        # Skip branches with no possible overlap with absolute angle limits.
        if np.any(lower > np.pi) or np.any(upper < -np.pi):
            continue
        def decode(v):
            t = np.tanh(matrix@v)
            return np.pi*t,np.pi*(1-t*t)[:,None]*matrix
        def objective(v):
            p,j = decode(v)
            delta = p-original
            return .5*float(delta@delta),delta@j
        def constraint(v):
            p,_ = decode(v)
            d = sweep@p
            return np.r_[d-lower,upper-d]
        def jacobian(v):
            _,j=decode(v)
            d=sweep@j
            return np.r_[d,-d]
        # An old gamma near +/-pi saturates tanh and can trap SLSQP.
        # Start also from the requested heading and from the interior. Every
        # candidate is still checked against the exact dense constraints.
        source_heading = np.linalg.lstsq(sweep, centre, rcond=None)[0]
        heading_seed = np.linalg.lstsq(matrix, np.arctanh(
            np.clip(source_heading/np.pi, -.98, .98)), rcond=None)[0]
        for start in (x0, heading_seed, np.zeros_like(x0)):
            fit=minimize(objective,np.clip(start,-7,7),jac=True,method='SLSQP',bounds=[(-7,7)]*len(x0),
                constraints=[dict(type='ineq',fun=constraint,jac=jacobian)],
                options=dict(maxiter=500,ftol=1e-10))
            violation=max(0.,float(-constraint(fit.x).min()))
            if np.isfinite(fit.fun) and violation<1e-7:
                candidates.append((float(fit.fun),decode(fit.x)[0],fit.x.copy()))
                break
    if not candidates:
        from scipy.optimize import linprog
        linear_feasible = False
        for branch in (-1, 0, 1):
            centre = heading + branch*2*np.pi
            lower, upper = centre-np.pi/6+margin, centre+np.pi/6-margin
            feasible = linprog(np.zeros(sweep.shape[1]), A_ub=np.vstack((sweep, -sweep)),
                b_ub=np.r_[upper, -lower], bounds=[(-np.pi, np.pi)]*sweep.shape[1], method='highs')
            linear_feasible |= bool(feasible.success)
        raise ValueError('no feasible CGL gamma initialization found; '
            f'heading_range_rad=[{heading.min():.6f},{heading.max():.6f}], '
            f'linear_relaxation_feasible={linear_feasible}; do not start unconstrained inversion')
    selected = min(candidates,key=lambda pair:pair[0])
    return (selected[1],selected[2]) if return_node_logits else selected[1]


def initialize_gamma(renderer,xy,posture,ids,gamma,matrices,point_indices,
                     *, return_node_logits=False):
    import torch
    with torch.no_grad():
        dense_xy,_,dense_ids=renderer.densify_for_rendering(xy,posture,ids)
        heading=renderer.forward_trajectory_heading(dense_xy,dense_ids).cpu().numpy()
        basis=torch.eye(len(xy),device=xy.device,dtype=xy.dtype)
        sweep=np.column_stack([renderer.densify_scalar_for_rendering(xy,basis[i],ids).cpu().numpy()
                               for i in range(len(xy))])
    dense_ids=dense_ids.cpu().numpy()
    source_ids=ids.cpu().numpy()
    result=np.asarray(gamma,float).copy()
    node_logits=np.zeros((len(matrices),matrices[0].shape[1]),dtype=np.float32)
    for stroke_index,(matrix,indices) in enumerate(zip(matrices,point_indices)):
        rows=dense_ids==source_ids[indices[0]]
        fitted_points,fitted_nodes=fit_gamma_seed(
            matrix.detach().cpu().numpy(),sweep[rows][:,indices],
            heading[rows],result[indices],return_node_logits=True)
        result[indices]=fitted_points
        node_logits[stroke_index]=fitted_nodes
    points=result.astype(np.float32)
    return (points,node_logits) if return_node_logits else points
