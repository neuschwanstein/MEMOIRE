import numpy as np
from numpy import sqrt,log
import numpy.linalg as la
import cvxpy as cvx

from cd.model.utility import RiskNeutralUtility


def conv(n,p):
    n_true = 100000
    m = 100
    lamb = 1/2

    X_true = np.random.randn(n_true,p)
    r_true = X_true.sum(axis=1)
    # r_true = r_true/r_true.std()

    X = np.random.randn(m,n,p)
    r = X.sum(axis=2)
    # r = 1/np.sqrt(p) * r

    q = 1/(2*lamb) * (X*r[:,:,None]).mean(axis=1)
    insample = (r*np.einsum('mnp,mp->mn',X,q)).mean(axis=1)
    outsample = (r_true * np.einsum('tp,mp->mt',X_true,q)).mean(axis=1)

    return insample,outsample


def solve(ts,u=RiskNeutralUtility(),lamb=1):
    if len(ts.shape) == 3:
        m,n,p = ts.shape
    else:
        n,p = ts.shape
        ts = [ts]
        m = 1
    qs = np.empty((m,p))

    if isinstance(u,RiskNeutralUtility):
        qs = 1/lamb * ts.mean(axis=1)

    else:
        q = cvx.Variable(p)
        t = cvx.Parameter(n,p)
        obj = 1/n * cvx.sum_entries(u.cvx_util(t*q)) - lamb/2*cvx.norm(q)**2
        prob = cvx.Problem(cvx.Maximize(obj))

        for i,tval in enumerate(ts):
            t.value = tval
            try:
                prob.solve()
            except cvx.SolverError:
                prob.solve(solver=cvx.SCS)
            try:
                qs[i] = q.value.A1
            except:
                qs[i] = q.value

    return qs


def in_error(qs,u=RiskNeutralUtility(),ts=None,lamb=1):
    if isinstance(u,RiskNeutralUtility) or ts is None:
        return lamb * la.norm(qs,axis=1)**2
    if len(qs.shape) == 1:
        ins = np.mean(u(ts@qs))
    else:
        ins = np.einsum('mnp,mp->mn',ts,qs)
        ins = u(ins).mean(axis=-1)
    return ins


def out_error(qs,u=RiskNeutralUtility(),t_true=None,q_true=None,lamb=1):
    p = qs.shape[-1]
    if isinstance(u,RiskNeutralUtility) and q_true is not None:
        return lamb * qs@q_true
    if (len(qs.shape) == 1):
        outs = np.mean(u(t_true@qs))
    else:
        outs = t_true[:,:p]@qs.T
        outs = u(outs).mean(axis=0)
    return outs


def max_error(u=RiskNeutralUtility(),qs=None,ts=None,t_true=None,q_true=None,lamb=1):
    if qs is None:
        qs = solve(ts,u,lamb)
    inerror = in_error(qs,u=u,ts=ts,lamb=lamb)
    outerror = out_error(qs,u=u,t_true=t_true,lamb=lamb)
    maxerror = inerror - outerror
    idx = np.argmax(maxerror)
    return maxerror[idx],inerror[idx],outerror[idx]


def error_u(inerror,outerror):
    # return (inerror - outerror).max()
    return np.percentile(inerror - outerror,95)


def error_ce(inerror,outerror,u):
    # return (u.inverse(inerror) - u.inverse(outerror)).max()
    return np.percentile(u.inverse(inerror) - u.inverse(outerror),95)


def create_sample(t_true,m,n):
    n_true = len(t_true)
    idx = np.random.choice(n_true,size=(m,n))
    return t_true[idx,:]


def solved(X,r,lamb,u=RiskNeutralUtility()):
    n,p = X.shape
    a = cvx.Variable(n)
    # K = X@X.T
    K = np.inner(X,X)
    obj = 1/n * cvx.sum_entries(u.cvx_util(cvx.mul_elemwise(r,K*a))) - lamb*cvx.quad_form(a,K)
    prob = cvx.Problem(cvx.Maximize(obj))
    prob.solve()
    try:
        return a.value.A1
    except:
        return a.value


def bound_genu(ns,xi,lamb=1,delta=0.05):
    tol = log(1/delta)
    return xi**2/lamb * (1/ns + 4*sqrt(tol/2/ns))


def bound_gence(ns,xi,u,inerror,outerror,lamb=1,delta=0.05):
    idx = np.argmax(u.inverse(inerror) - u.inverse(outerror))
    return u.subinverse(inerror[idx]) * bound_genu(ns,xi,lamb,delta)


def bound_sou(ns,ps,q_true,lamb=1,delta=0.05):
    tol = log(1/delta)
    lamb = lamb/2
    n_qtrue = lamb * la.norm(q_true)**2
    # eu_true = u(t_true@q_true).mean()
    om = 4*ps**2*(32+tol)/lamb/ns
    return 2*om + ps*sqrt(om) + n_qtrue


def bound_soce(ns,ps,q_true,u,soerror,lamb=1,delta=0.05):
    tol = log(1/delta)
    n_qtrue = lamb/2 * la.norm(q_true)**2
    # eu_true = u(t_true@q_true).mean()
    om = 4*ps*(32+tol)/ns
    # return u.inverse(eu_true) - u.subinverse(soerror)*(4*om + sqrt(2*ps*om) + n_qtrue)
    return -u.subinverse(soerror)*(4*om + sqrt(2*ps*om) + n_qtrue)
