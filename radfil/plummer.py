import numpy as np
from astropy.modeling.core import Fittable1DModel
from astropy.modeling.parameters import Parameter


class BasePlummer1D(Fittable1DModel):
    """Base class for 1D Plummer-like models, centering at 0. Do not use this directly.

    See Also
    --------
    Plummer1D
    """

    amplitude = Parameter(default=1.)
    powerIndex = Parameter(default=2.)
    flatteningRadius = Parameter(default=1.)

    def bounding_box(self, factor=10.):
        """
        Tuple defining the default ``bounding_box`` limits,
        ``(x_low, x_high)``

        Parameters
        ----------
        factor : float
            The multiple of `stddev` used to define the limits.
            The default is 10.
            Notice that the relative amplitude is dependent on powerIndex as well.

        Examples
        --------
        >>> from astropy.modeling.models import Gaussian1D
        >>> model = Plummer1D(flatteningRadius=2)
        >>> model.bounding_box
        (-20., 20.)

        This range can be set directly (see: `Model.bounding_box
        <astropy.modeling.Model.bounding_box>`) or by using a different factor,
        like:

        >>> model.bounding_box = model.bounding_box(factor=2)
        >>> model.bounding_box
        (-4.0, 4.0)
        """

        # The function itself always centers at 0.
        dx = factor * self.flatteningRadius.value

        return (-dx, dx)



class Plummer1D(BasePlummer1D):
    """
    One dimensional Plummer-like model.
    (Arzoumanian et al. 2011, Zucker et al. 2017)

    Parameters
    ----------
    amplitude : float
        Amplitude of the Plummer-like function at 0 (the center).
    powerIndex : float
        The power-law index, p, in the function.
    flatteningRadius : float
        R_flat in the function.

    Notes
    -----

    Model formula:

        .. math:: f(x) = N_0 / (1 + (r/R_flat)**2)**((p-1)/2)

    Examples (for regular astropy.modeling.Gaussian1D, but the structure is the same.)
    --------
    >>> from astropy.modeling import models
    >>> def tie_center(model):
    ...         mean = 50 * model.stddev
    ...         return mean
    >>> tied_parameters = {'mean': tie_center}

    Specify that 'mean' is a tied parameter in one of two ways:

    >>> g1 = models.Gaussian1D(amplitude=10, mean=5, stddev=.3,
    ...                             tied=tied_parameters)

    or

    >>> g1 = models.Gaussian1D(amplitude=10, mean=5, stddev=.3)
    >>> g1.mean.tied
    False
    >>> g1.mean.tied = tie_center
    >>> g1.mean.tied
    <function tie_center at 0x...>

    Fixed parameters:

    >>> g1 = models.Gaussian1D(amplitude=10, mean=5, stddev=.3,
    ...                        fixed={'stddev': True})
    >>> g1.stddev.fixed
    True

    or

    >>> g1 = models.Gaussian1D(amplitude=10, mean=5, stddev=.3)
    >>> g1.stddev.fixed
    False
    >>> g1.stddev.fixed = True
    >>> g1.stddev.fixed
    True

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from astropy.modeling.models import Gaussian1D

        plt.figure()
        s1 = Gaussian1D()
        r = np.arange(-5, 5, .01)

        for factor in range(1, 4):
            s1.amplitude = factor
            plt.plot(r, s1(r), color=str(0.25 * factor), lw=2)

        plt.axis([-5, 5, -1, 4])
        plt.show()

    """

    @staticmethod
    def evaluate(x, amplitude, powerIndex, flatteningRadius):
        """
        Plummer1D model function.
        """
        return amplitude / (1.+(x/flatteningRadius)**2.)**((powerIndex-1.)/2.)


    @staticmethod
    def fit_deriv(x, amplitude, powerIndex, flatteningRadius):
        """
        Plummer1D model function derivatives.
        """

        d_amplitude = 1. / (1.+(x/flatteningRadius)**2.)**((powerIndex-1.)/2.)
        d_powerIndex = (-amplitude/2.)*np.log(1.+(x/flatteningRadius)**2.)*\
                       (1.+(x/flatteningRadius)**2.)**((1.-powerIndex)/2.)
        d_flatteningRadius = amplitude*((1.-powerIndex)/2.)*(-2.*x**2./flatteningRadius**3.)*\
                             (1.+(x/flatteningRadius)**2.)**(-(powerIndex+1.)/2.)

        return [d_amplitude, d_powerIndex, d_flatteningRadius]
