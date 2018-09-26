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

    def get_cosmo(self, classy_dict):
        """
        Compute a cosmology with CLASS.

        This is purely for convenience.
        Parameters
        ----------
        classy_dict (dictionary) : contains the inputs for the CLASS python
            wrapper such as 'output' and 'l_max_scalars'.

        """
        cosmo = Class()
        cosmo.set(classy_dict)
        cosmo.compute()
        return cosmo
        
    def compute_cosmo(self, key, classy_dict):
        """
        Generate an entry in the dictionary Observables.cosmos with CLASS.

        Parameters
        ----------
        key (string) : key to store the computed cosmology in the dictionary
        classy_dict (dictionary) : contains the inputs for the CLASS python
            wrapper such as 'output' and 'l_max_scalars'.

        """
        self.cosmos[key] = self.get_cosmo(classy_dict)
        
    def grid_cosmo(self, parameter, parameter_grid, classy_dict, verbose=False):
        """
        Compute a grid of cosmologies, varying one parameter over a grid.
        
        Parameters
        ----------
        parameter (string) : name of parameter in the CLASS dict
        parameter_grid (list of float) : grid over which the parameter will
            be varied
        classy_dict (dictionary) : base dictionary to be copied for each 
            cosmology evaluation
            
        Returns
        -------
            list of CLASS objects : contains a list of computed cosmologies,
                as a result of varying the parameter over the grid
        """
        cosmo_list = []
        for grid_value in parameter_grid:
            cosmo = Class()
            temp_dict = classy_dict.copy()
            temp_dict[parameter] = grid_value
            if verbose:
                print(temp_dict)
            cosmo.set(temp_dict)
            cosmo.compute()
            cosmo_list.append(cosmo)
        return cosmo_list
        
    def clean_cosmo(self):
        # requires python 3 here
        for key in self.cosmos:
            self.cosmos[key].struct_cleanup()
            self.cosmos[key].empty()
