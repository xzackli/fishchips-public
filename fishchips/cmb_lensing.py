"""Experiment classes for lensing experiments."""

from fishchips.experiments import Experiment
import numpy as np
import itertools

class CMB_Lensing_Only(Experiment):
    """Stores information on noise, priors, and computed Fisher matrices.
    Just TT/TE/EE with lensing, LCDM noise from orphics
    """
    
        # DEFAULTS ARE FOR PLANCK EXPERIMENT
    def __init__(self, 
             lens_beam = 7.0,lens_noiseT = 33.,lens_noiseP = 56.,
             lens_tellmin = 2,lens_tellmax = 3000,lens_pellmin = 2,
             lens_pellmax = 3000,lens_kmin = 80,lens_kmax = 3000, lens_f_sky=0.65,
                bin_width=80, estimators=('TT','TE','EE','EB','TB'), 
                NlTT=None, NlEE=None, NlBB=None):

        # get lensing noise
        # Initialize cosmology and Clkk. Later parts need dimensionless spectra.
        self.l_min = lens_tellmin
        # CLASS can only go up to 5500
        self.l_max = min(max(lens_tellmax, lens_pellmax)+1000,5500) 
        self.k_min = lens_kmin
        self.k_max = lens_kmax
        self.f_sky = lens_f_sky
        
        # import orphics only here! 
        from orphics import lensing, io, stats, cosmology, maps
        # generate cosmology with orphics
        lmax = self.l_max
        cc = cosmology.Cosmology(lmax=lmax,pickling=True,dimensionless=False)
        theory = cc.theory
        ells = np.arange(2,lmax,1)
        clkk = theory.gCl('kk',ells)
        Tcmb = 2.726
        
        # compute noise curves
        sT = lens_noiseT * (np.pi/60./180.)
        sP = lens_noiseP * (np.pi/60./180.)
        theta_FWHM = lens_beam * (np.pi/60./180.)
        muK = Tcmb*1.0e6
        # unitless white noise
        exp_term = np.exp(ells*(ells+1)*(theta_FWHM**2)/(8*np.log(2)))
        if NlTT is None:
            NlTT = sT**2 * exp_term #/ muK**2
        else:
            NlTT = NlTT[ells]
        if NlEE is None:
            NlEE = sP**2 * exp_term #/ muK**2
        else:
            NlEE = NlEE[ells]
        if NlBB is None:
            NlBB = sP**2 * exp_term #/ muK**2
        else:
            NlBB = NlBB[ells]
            
            
        NlTT[ells > lens_tellmax] = 1e100
        NlEE[ells > lens_pellmax] = 1e100
        NlBB[ells > lens_pellmax] = 1e100
            
        self.NlTT = NlTT
        self.NlEE = NlEE
        self.NlBB = NlBB
        
        # Define bin edges for noise curve
        bin_edges = np.arange(2,lmax,bin_width)
        
        # compute orphics lensing noise
        ls,nlkks,theory_,qest = lensing.lensing_noise(
            ells=ells,
            ntt=NlTT,
            nee=NlEE,
            nbb=NlBB,
            ellmin_t=lens_tellmin, ellmin_e=lens_pellmin,ellmin_b=lens_pellmin,
            ellmax_t=lens_tellmax, ellmax_e=lens_pellmax,ellmax_b=lens_pellmax,
            bin_edges=bin_edges,
            estimators=estimators,
            ellmin_k = 2,
            ellmax_k = lens_kmax + 500, # calculate out a bit
            theory=theory,
            dimensionless=False)

        self.orphics_kk = clkk
        self.orphics_ls = ls
        self.orphics_nls = nlkks['mv']
        self.noise_k = np.interp(np.arange(self.l_max+1), ls, nlkks['mv'])

        self.noise_k[np.arange(self.l_max+1) <= lens_kmin] = 1e100
        self.noise_k[np.arange(self.l_max+1) >= lens_kmax] = 1e100


    def compute_fisher_from_spectra(self, fid, df, pars):
        """
        Compute the Fisher matrix given fiducial and derivative dicts.

        This function is for generality, to enable easier interfacing with
        codes like CAMB. The input parameters must be in the units of the
        noise, muK^2.

        Parameters
        ----------
        fid (dictionary) : keys are '{parameter_XY}' with XY in {tt, te, ee}.
            These keys point to the actual power spectra.

        df (dictionary) :  keys are '{parameter_XY}' with XY in {tt, te, ee}.
            These keys point to numerically estimated derivatives generated
            from precomputed cosmologies.

        pars (list of strings) : the parameters being constrained in the
            Fisher analysis.

        """
        npar = len(pars)
        self.fisher = np.zeros((npar, npar))
        print()
        for i,j in itertools.combinations_with_replacement( range(npar),r=2):
            # following eq 4 of https://arxiv.org/pdf/1402.4108.pdf
            fisher_ij = 0.0
            for l in range(self.k_min, self.k_max):

                Clkk_plus_Nlkk_sq = (fid['kk'][l] + self.noise_k[l])**2
                dCl_kk = df[pars[i]+'_kk'][l]
                
                fisher_contrib = (2*l+1)/2. * self.f_sky * \
                    (df[pars[i]+'_kk'][l] * 
                     df[pars[j]+'_kk'][l])/Clkk_plus_Nlkk_sq
                fisher_ij += fisher_contrib
                
            # fisher is diagonal
            self.fisher[i,j] = fisher_ij
            self.fisher[j,i] = fisher_ij
        
        return self.fisher

    def get_fisher(self, obs, lensed_Cl=True):
        """
        Return a Fisher matrix using a dictionary full of CLASS objects.

        This function wraps the functionality of `compute_fisher_from_spectra`,
        for use with a dictionary filled with CLASS objects.

        Parameters
        ----------
            obs (Observations instance) : contains many evaluated CLASS cosmologies, at
                both the derivatives and the fiducial in the cosmos object.

        Returns
        -------
            Numpy array of floats with dimensions (len(params), len(params))

        """
        # first compute the fiducial
        fid_cosmo = obs.cosmos['fiducial']
        Tcmb = fid_cosmo.T_cmb()
        fid_cl = fid_cosmo.lensed_cl(self.l_max)
        
        fid = {'kk': 0.25 * ((fid_cl['ell']+1)
               *(fid_cl['ell']))**2 * fid_cl['pp']}


        # the primary task of this function is to compute the derivatives from `cosmos`,
        # the dictionary of computed CLASS cosmologies
        dx_array = np.array(obs.right) - np.array(obs.left)

        df = {}
        # loop over parameters, and compute derivatives
        for par, dx in zip(obs.parameters, dx_array):
            cl_left = obs.cosmos[par + '_left'].lensed_cl(self.l_max)
            cl_right = obs.cosmos[par + '_right'].lensed_cl(self.l_max)
            
            kk_left = (0.25 * (cl_left['ell']+2)*(cl_left['ell']+1)
                      *(cl_left['ell'])*(cl_left['ell']-1) * cl_left['pp'])
            kk_right = (0.25 * (cl_right['ell']+2)*(cl_right['ell']+1)
                      *(cl_right['ell'])*(cl_right['ell']-1) * cl_right['pp'])

            df[par + '_kk'] = (kk_right - kk_left) / dx
        self.df = df
        self.fid = fid
        
        return self.compute_fisher_from_spectra(fid,
                                                df,
                                                obs.parameters)

    
    