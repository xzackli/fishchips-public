# -*- coding: utf-8 -*-
"""Experiment objects for direct use and inheritance.

This module provides a number of common cosmological experiment configurations.
To include a novel kind of experiment or data, inherit the Experiment class.
"""

import numpy as np


class Experiment:
    """
    Abstract class for generic Fisher matrix computation.

    All experiments inherit this. It defines the basic structure needed for
    computation of Fisher matrices, namely just the `compute_fisher` function
    call. More complicated experiments will require a constructor which
    establishes noise properties, for example.
    """

    def get_fisher(self, cosmos):
        """Initialize the experiment with noise parameters.

        Args:
            cosmos (list of string): names of parameters
            means (list of float): mean values of parameters
        """
        raise NotImplementedError("You need to implement the computation of "
                                  "the Fisher matrix!")


class CMB_Primary(Experiment):
    """
    Class for computing Fisher matrices from the CMB primary (TT/TE/EE).

    This experiment class requires some instrument parameters, and computes
    white noise for each multipole. The computation of the Fisher matrix
    follows equation 4 of arxiv:1402.4108.
    """

    def __init__(self, theta_fwhm=[10., 7., 5.],
                 sigma_T=[68.1, 42.6, 65.4],
                 sigma_P=[109.4, 81.3, 133.6],
                 f_sky=0.65, l_min=2, l_max=2500,
                 verbose=False):
        """
        Initialize the experiment with noise parameters.

        Uses the Planck bluebook parameters by default.

        Parameters
        ----------
            theta_fwhm (list of float): beam resolution in arcmin
            sigma_T (list of float): temperature resolution in muK
            sigma_P (list of float): polarization resolution in muK
            f_sky (float): sky fraction covered
            l_min (int): minimum ell for CMB power spectrum
            l_max (int): maximum ell for CMB power spectrum
            verbose (boolean): flag for printing out debugging output

        """
        self.verbose = verbose

        # convert from arcmin to radians
        self.theta_fwhm = theta_fwhm * np.array([np.pi/60./180.])
        self.sigma_T = sigma_T * np.array([np.pi/60./180.])
        self.sigma_P = sigma_P * np.array([np.pi/60./180.])
        self.num_channels = len(theta_fwhm)
        self.f_sky = f_sky

        self.l_min = l_min
        self.l_max = l_max

        # compute noise in muK**2, adapted from Monte Python
        self.noise_T = np.zeros(self.l_max+1, 'float64')
        self.noise_P = np.zeros(self.l_max+1, 'float64')
        self.noise_TE = np.zeros(self.l_max+1, 'float64')

        for l in range(self.l_min, self.l_max+1):
            self.noise_T[l] = 0
            self.noise_P[l] = 0
            for channel in range(self.num_channels):
                self.noise_T[l] += self.sigma_T[channel]**-2 *\
                    np.exp(
                        -l*(l+1)*self.theta_fwhm[channel]**2/8./np.log(2.))
                self.noise_P[l] += self.sigma_P[channel]**-2 *\
                    np.exp(
                        -l*(l+1)*self.theta_fwhm[channel]**2/8./np.log(2.))
            self.noise_T[l] = 1/self.noise_T[l]
            self.noise_P[l] = 1/self.noise_P[l]

    def get_fisher(self, cosmos):
        """
        Return a Fisher matrix.

        Parameters
        ----------
            pars (list of string): names of parameters in Fisher matrix
            means (list of float): mean values of parameters in Fisher matrix

        Returns
        -------
            Numpy array of floats with dimensions (len(params), len(params))

        """
        npar = len(cosmos['parameters'])
        self.fisher = np.zeros((npar, npar))
        self.fisher_ell = np.zeros(self.l_max)

        # for i, j in itertools.combinations_with_replacement(range(npar), r=2):
        #     # following eq 4 of https://arxiv.org/pdf/1402.4108.pdf
        #     fisher_ij = 0.0
        #     # probably a more efficient way to do this exists
        #     for l in range(self.l_min, self.l_max):
        #
        #         Cl = np.array([[self.fid['tt'][l] + self.noise_T[l], self.fid['te'][l]  + self.noise_TE[l]],
        #                        [self.fid['te'][l] + self.noise_TE[l], self.fid['ee'][l] + self.noise_P[l]]])
        #         invCl = np.linalg.inv(Cl)
        #
        #         dCl_i = np.array( [[self.df[pars[i]+'_tt'][l], self.df[pars[i]+'_te'][l]],
        #                            [self.df[pars[i]+'_te'][l], self.df[pars[i]+'_ee'][l]] ] )
        #         dCl_j = np.array( [[self.df[pars[j]+'_tt'][l], self.df[pars[j]+'_te'][l]],
        #                            [self.df[pars[j]+'_te'][l], self.df[pars[j]+'_ee'][l]] ] )
        #
        #         inner_term = np.dot( np.dot( invCl, dCl_i ), np.dot( invCl, dCl_j ) )
        #         fisher_contrib = (2*l+1)/2. * self.f_sky * np.trace(inner_term)
        #
        #         if i == j and pars[i] == self.info_var:
        #             # print(pars[i], pars[j])
        #             self.fisher_ell[l] += fisher_contrib
        #         fisher_ij += fisher_contrib
        #
        #     # fisher is diagonal
        #     self.fisher[i, j] = fisher_ij
        #     self.fisher[j, i] = fisher_ij
        #
        # return self.fisher
