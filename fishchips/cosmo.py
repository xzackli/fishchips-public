"""Classes for handling computed cosmologies."""

from classy import Class  # CLASS python wrapper
import numpy as np

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
        self.parameter_index = dict(zip(parameters, list(range(len(parameters)))))
        
        # dictionary and steps used for keeping track of convergence
        self.check_deriv_cosmos = {}
        self.check_deriv_steps = {}

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

    # utility functions for checking one-sided convergence
    def check_deriv_compute(self, par, steps, classy_dict, verbose=False):
        
        self.check_deriv_cosmos[par] = self.grid_cosmo(par, 
                                                       steps + self.fiducial[self.parameter_index[par]], 
                                                       classy_dict, verbose=verbose)
        self.check_deriv_steps[par] = steps
        
    
    def check_deriv_plot(self, par, ell, l_max, spec='tt', lensed=True):
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1,2,figsize=(16,7))
        par_str = (r'\mathrm{' + par.replace('_', r'\_') + r'}')
        
        
        axes[0].set_ylabel(r"$(\Delta C_{\ell}^{" + spec + "} / \Delta" + par_str + ") / C_{\ell}^{" + spec + "}$")
        axes[0].set_xlabel(r"$\ell$")
        
        fid_cl = self.cosmos["fiducial"].lensed_cl(l_max)[spec]
        fid_cl[fid_cl == 0.0] = 1.0

        
        for step, c in zip(self.check_deriv_steps[par], self.check_deriv_cosmos[par]):
            normed_deriv = (c.lensed_cl(l_max)[spec] - fid_cl) / step / fid_cl
            normed_deriv[:2] = np.nan
            axes[0].plot( normed_deriv, label=r"$\Delta " + str(par_str) + f"= {step}$")
            
        axes[0].legend(fancybox=True, frameon = True)    
        axes[0].axvline(ell, ls="dashed")
        
        axes[1].set_title( '$\ell = $' + str(ell))
        
        fid_val = self.fiducial[self.parameter_index[par]]
        cl_at_ell = np.array([ c.lensed_cl(l_max)[spec][ell]
                     for c in self.check_deriv_cosmos[par]])
        derivs = (fid_cl[ell] - cl_at_ell) / self.check_deriv_steps[par]
        
        axes[1].plot( self.check_deriv_steps[par], 
                 derivs, "k-" )
        for x, y in zip(self.check_deriv_steps[par], derivs):
            l = plt.scatter([x], [y])
            l.set_zorder(50)
        axes[1].set_ylabel(r"$\Delta C_{\ell = " + str(ell) + r"}^{" + spec + "} / \Delta" + par_str + "$")
        axes[1].set_xlabel("$\Delta" + par_str + "$")
        
        
        min_y = min( 0.0, np.min(derivs) * 1.1 )
        max_y = max( 0.0, np.max(derivs) * 1.1 )
        axes[1].set_ylim(min_y, max_y)
        axes[1].set_xlim(0.0, np.max(self.check_deriv_steps[par]) * 1.1)
        plt.tight_layout()