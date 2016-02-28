import numpy as np

import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.atoms.elementwise.exp import exp


class LipschitzExp(Elementwise):
    def __init__(self,x,β=1,x0=0):
        self.β = self.cast_to_const(β)
        self.x0 = self.cast_to_const(x0)
        super().__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self,values):
        β = self.β.value
        x0 = self.x0.value
        x = values[0]
        exp = np.exp
        return (x >= x0)* (1/β * (1 - np.exp(-β*x))) + \
               (x < x0) * (x*np.exp(-β*x0) + 1/β*(1-(1+β*x0)*np.exp(-β*x0)))

    def sign_from_args(self):
        return u.Sign.UNKNOWN

    def func_curvature(self):
        return u.Curvature.CONCAVE

    def monotonicity(self):
        return [u.monotonicity.DECREASING]

    def get_data(self):
        return [self.β,self.x0]

    def validate_arguments(self):
        '''Check that β>0.'''
        if not (self.β.is_positive() and
                self.β.is_constant() and self.β.is_scalar() and
                self.x0.is_constant() and self.x0.is_scalar()):
            raise ValueError('β must be a non-negative scalar constant, and x0 must a constant scalar.')

    @staticmethod
    def graph_implementation(arg_objs,size,data=None):
        x = arg_objs[0]
        β,x0 = data[0],data[1]
        β_val,x0_val = β.value,x0.value

        if isinstance(β,Parameter):
            β = lu.create_param(β,(1,1))
        else:
            β = lu.create_const(β.value,(1,1))
        if isinstance(x0,Parameter):
            x0 = lu.create_param(x0,(1,1))
        else:
            x0 = lu.create_const(x0.value,(1,1))

        ξ,ψ = lu.create_var(size),lu.create_var(size)
        one = lu.create_const(1,(1,1))
        one_over_β = lu.create_const(1/β_val,(1,1))
        k = np.exp(-β_val*x0_val)
        k = lu.create_const(k,(1,1))

        # 1/β * (1 - exp(-β*(ξ+x0)))
        ξ_plus_x0 = lu.sum_expr([ξ,x0])
        minus_β_times_ξ_plus_x0  = lu.neg_expr(lu.mul_expr(β,ξ_plus_x0,size))
        exp_ξ,constr_exp = exp.graph_implementation([minus_β_times_ξ_plus_x0],size)
        minus_exp_minus_etc = lu.neg_expr(exp_ξ)
        left_branch = lu.mul_expr(one_over_β, lu.sum_expr([one,minus_exp_minus_etc]),size)

        # ψ*exp(-β*r0)
        right_branch = lu.mul_expr(k,ψ,size)

        obj = lu.sum_expr([left_branch,right_branch])

        #x-x0 == ξ + ψ, ξ >= 0, ψ <= 0
        zero = lu.create_const(0,(1,1))
        constraints = constr_exp
        constraints.append(lu.create_eq(x,lu.sum_expr([x0,ξ,ψ])))
        constraints.append(lu.create_geq(ξ,zero))
        constraints.append(lu.create_leq(ψ,zero))

        return (obj, constraints)
