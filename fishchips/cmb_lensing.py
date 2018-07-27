"""Experiment classes for lensing experiments."""

from orphics import lensing, io, stats, cosmology, maps
from fishchips.experiments import Experiment
import numpy as np

class CMB_Lensing_Only(Experiment):
    """Stores information on noise, priors, and computed Fisher matrices.
    Just TT/TE/EE with lensing, LCDM noise from orphics
    """
    
        # DEFAULTS ARE FOR PLANCK EXPERIMENT
    def __init__(self, 
             lens_beam = 7.0,lens_noiseT = 33.,lens_noiseP = 56.,
             lens_tellmin = 2,lens_tellmax = 3000,lens_pellmin = 2,
             lens_pellmax = 3000,lens_kmin = 80,lens_kmax = 2000, lens_f_sky=0.65 ):

        # get lensing noise
        # Initialize cosmology and Clkk. Later parts need dimensionless spectra.
        self.l_min = lens_tellmin
        self.l_max = lens_tellmax
        self.k_min = lens_kmin
        self.k_max = lens_kmax
        self.f_sky = lens_f_sky
        cc = cosmology.Cosmology(lmax=self.l_max,pickling=True,dimensionless=True)
        theory = cc.theory
        ells = np.arange(2,self.l_max,1)
        clkk = theory.gCl('kk',ells)

        # Make a map template for calculating the noise curve on
        shape,wcs = maps.rect_geometry(width_deg = 5.,px_res_arcmin=1.5)
        # Define bin edges for noise curve
        bin_edges = np.arange(80,lens_kmax,20)
        nlgen = lensing.NlGenerator(shape,wcs,theory,bin_edges,lensedEqualsUnlensed=True)
        # Experiment parameters, here for Planck
        polCombs = ['TT','TE','EE','EB','TB']

        _,_,_,_ = nlgen.updateNoise(
            beamX=lens_beam,noiseTX=lens_noiseT,noisePX=lens_noiseP,
            tellminX=lens_tellmin,tellmaxX=lens_tellmax,
            pellminX=lens_pellmin,pellmaxX=lens_pellmax)

        ls,nls,bells,nlbb,efficiency = nlgen.getNlIterative(polCombs,lens_kmin,lens_kmax,
                                                            lens_tellmax,lens_pellmin,lens_pellmax,
                                                            verbose=True,plot=False)

        self.orphics_kk = clkk
        self.orphics_ls = ls
        self.orphics_nls = nls
        self.noise_k = np.interp(np.arange(self.l_max+1), ls, nls)

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
        
        for i,j in itertools.combinations_with_replacement( range(nparams),r=2):
            # following eq 4 of https://arxiv.org/pdf/1402.4108.pdf
            fisher_ij = 0.0
            for l in range(self.k_min, self.k_max):

                Clkk_plus_Nlkk_sq = (fid['kk'][l] + self.noise_k[l])**2
                dCl_kk = df[pars[i]+'_kk'][l]
                
                fisher_contrib = (2*l+1)/2. * self.f_sky * \
                    (df[pars[i]+'_kk'][l] * 
                     df[pars[j]+'_kk'][l])/Clkk_plus_Nlkk_sq

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
        fid_cosmo = obs.cosmos['CLASS_fiducial']
        Tcmb = fid_cosmo.T_cmb()
        if lensed_Cl:
            fid_cl = fid_cosmo.lensed_cl(self.l_max)
        else:
            fid_cl = fid_cosmo.raw_cl(self.l_max)
        fid = {'kk': 0.25 * ((fid_cl['ell']+2)*(fid_cl['ell']+1)
               *(fid_cl['ell'])*(fid_cl['ell']-1) * fid_cl['pp'])}

        # the primary task of this function is to compute the derivatives from `cosmos`,
        # the dictionary of computed CLASS cosmologies
        dx_array = np.array(obs.right) - np.array(obs.left)

        df = {}
        # loop over parameters, and compute derivatives
        for par, dx in zip(obs.parameters, dx_array):
            cl_left = obs.cosmos[par + '_CLASS_left'].lensed_cl(self.l_max)
            cl_right = obs.cosmos[par + '_CLASS_right'].lensed_cl(self.l_max)
            
            kk_left = (0.25 * (cl_left['ell']+2)*(cl_left['ell']+1)
                      *(cl_left['ell'])*(cl_left['ell']-1) * cl_left['pp'])
            kk_right = (0.25 * (cl_right['ell']+2)*(cl_right['ell']+1)
                      *(cl_right['ell'])*(cl_right['ell']-1) * cl_right['pp'])

            df[par + '_kk'] = (kk_right - kk_left) / dx

        return self.compute_fisher_from_spectra(fid,
                                                df,
                                                obs.parameters)

    
    