import numpy as np

import cvxpy.lin_ops.lin_utils as lu
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.atoms.elementwise.exp import exp


class LipschitzExp(Elementwise):
    def __init__(self,x,beta=1):
        self.beta = self.cast_to_const(beta)
        super(LipschitzExp, self).__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self,values):
        b = self.beta.value
        x = values[0]
        return np.where(x<=0,x,b*(1-np.exp(-x/b)))

    def sign_from_args(self):
        # return u.Sign.UNKNOWN
        return (False,False)

    def is_atom_convex(self):
        return False

    def is_atom_concave(self):
        return True

    def is_incr(self,idx):
        return True

    def is_decr(self,idx):
        return False

    def get_data(self):
        return [self.beta]

    def validate_arguments(self):
        '''Check that beta>0.'''
        if not (self.beta.is_positive() and
                self.beta.is_constant() and
                self.beta.is_scalar()):
            raise ValueError('beta must be a non-negative scalar constant')

    def _grad(self,values):
        rows = self.args[0].size[0]*self.args[0].size[1]
        cols = self.size[0]*self.size[1]
        x = values[0]
        grad_vals = np.where(x<0,1,np.exp(-x/self.beta.value))
        return [LipschitzExp.elemwise_grad_to_diag(grad_vals,rows,cols)]

    @staticmethod
    def graph_implementation(arg_objs,size,data=None):
        beta = data[0]
        x = arg_objs[0]

        v = lu.create_var(size)
        w = lu.create_var(size)

        if isinstance(beta, Parameter):
            beta = lu.create_param(beta,(1,1))
        else:                   # Beta is constant
            beta = lu.create_const(beta.value,(1,1))

        exppart,cexp = exp.graph_implementation(
            [lu.neg_expr(lu.div_expr(w,beta))],
            size)

        # negexp = lu.sum_expr(
        #     [lu.create_const(1,(1,1)),
        #      lu.neg_expr(exppart)])
        negexp = lu.neg_expr(exppart)

        negexp = lu.mul_elemwise(beta,negexp)

        obj = lu.sum_expr([v,negexp])
        # print(obj)

        c1 = lu.create_leq(v)
        c2 = lu.create_geq(w)
        c3 = lu.create_eq(x,lu.sum_expr([v,w]))
        constraints = cexp
        constraints.append(c1)
        constraints.append(c2)
        constraints.append(c3)

        return (obj,constraints)
