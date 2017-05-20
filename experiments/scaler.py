import numpy as np
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms


class SqrtScale(mscale.ScaleBase):

    name = 'sqrtscale'

    def __init__(self, axis, **kwargs):
        mscale.ScaleBase.__init__(self)

    def get_transform(self):
        return self.SqrtTransform()

    def set_default_locators_and_formatters(self, axis):
        pass

    class SqrtTransform(mtransforms.Transform):

        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self):
            mtransforms.Transform.__init__(self)

        def transform_non_affine(self,a):
            # return 'Hello'
            # return np.where(np.abs(a)<=1e-6,a,(1/a)**2)
            # return (1/a)**2
            # return np.sqrt(a)
            # return a**2
            # masked = np.ma.masked_where(np.abs(a)<1e-5,a)
            # if masked.mask.any():
            #     # return np.ma.power(a,-2)
            #     return np.ma.sqrt(a)
            # else:
            #     # return np.power(a,-2)
            #     return np.sqrt(a)
            return np.sqrt(a)

        def inverted(self):
            return SqrtScale.InvertedSqrtScale()
            # return self

    class InvertedSqrtScale(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init_(self):
            mtransforms.Transform.__init__(self)

        def transform_non_affine(self,a):
            # return np.where(a<=1e-3,a,(1/a)**2)
            # return np.power(a**2)
            # return 1/np.sqrt(a)
            return a**2

        def inverted(self):
            return SqrtScale.SqrtTransform()


mscale.register_scale(SqrtScale)
