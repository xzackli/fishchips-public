"""Classes for handling computed cosmologies."""

from classy import Class  # CLASS python wrapper


class Observables:
    """Stores information about cosmological observables and parameters."""

    def __init__(self, parameters, fiducial, left, right):
        """
        Create the Observables class with parameters to be constrained.

        Parameters
        ----------
        parameters (list of strings) : parameters to be constrained, for
            example A_s or omega_m.
        fiducial (list of floats) : the fiducial values of the parameters
            to be constrained, this would correspond to the centers of the
            ellipses you desire.
        left (list of floats) : the left side values of the parameters
            to be evaluated for the numerical derivative.
        right (list of floats) : the right side values of the parameters to
            be evaluated for the numerical derivative.

        """
        self.parameters = parameters
        self.fiducial = fiducial
        self.left = left
        self.right = right
        self.cosmos = {}

    def compute_cosmo(self, key, classy_dict):
        """
        Generate an entry in the dictionary Observables.cosmos with CLASS.

        Parameters
        ----------
        key (string) : key to store the computed cosmology in the dictionary
        classy_dict (dictionary) : contains the inputs for the CLASS python
            wrapper such as 'output' and 'l_max_scalars'.

        """
        cosmo = Class()
        cosmo.set(classy_dict)
        cosmo.compute()
        self.cosmos[key] = cosmo
