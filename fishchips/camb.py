# -*- coding: utf-8 -*-
"""Experiment objects for use with CAMB instead of CLASS.
"""

import numpy as np
import itertools
from subprocess import Popen, PIPE
import shutil
from astropy.io import ascii

from fishchips.experiments import CMB_Primary

class CAMBy:
    def __init__(self, lensed, unlensed, TCMB=2.7255):
        """
        Takes in astropy tables
        """
        self.TCMB = TCMB
        
        # NOTE: CAMB RETURNS ONLY UP TO ELL 2, SO 
        # PAD FIRST TWO
        self.lensed_cl_dict = dict(
            zip(lensed.colnames,[ 
                np.hstack([[0.,0.],lensed[p]]) for p in lensed.colnames]))
        self.raw_cl_dict = dict(
            zip(unlensed.colnames,[
                np.hstack([[0.,0.],unlensed[p]]) for p in unlensed.colnames]))
        
        # FIX THE ELL ARRAY
        self.lensed_cl_dict['ell'] = np.arange(len(self.lensed_cl_dict['ell']))
        self.raw_cl_dict['ell'] = np.arange(len(self.raw_cl_dict['ell']))
        
        
        # convert format to mimic Classy
        
        for xx in ['tt', 'ee', 'bb', 'te']:
            ell = self.lensed_cl_dict['ell']
            self.lensed_cl_dict[xx][1:] *= 2 * np.pi / (ell * (ell+1))[1:]
        for xx in ["tt", 'ee', 'te']:
            ell = self.raw_cl_dict['ell']
            self.raw_cl_dict[xx][1:] *= 2 * np.pi / (ell * (ell+1))[1:]
        
        #--> deflection d:
        #Cl^dd = l(l+1) C_l^phiphi
        #--> convergence kappa and shear gamma: the share the same harmonic power spectrum:
        #Cl^gamma-gamma = 1/4 * [(l+2)!/(l-2)!] C_l^phi-phi
        ells = np.arange(len(self.raw_cl_dict['pp']))
        self.lensed_cl_dict['pp'] = self.raw_cl_dict['pp']
        self.lensed_cl_dict['pp'][1:] /= ells[1:]**4
        self.lensed_cl_dict['kk'] = self.lensed_cl_dict['pp'] * (ells * (ells+1.)/2.)**2.
        
    def clip_l_max(self, input_dict, l_max):
        return dict(
            zip(input_dict.keys(), 
                [input_dict[p][:l_max+1] for p in input_dict]))
        
    def T_cmb(self):
        return self.TCMB
    
    # CONVERT TO DICTIONARY TOMOORROW
    def lensed_cl(self, l_max):
        return self.clip_l_max(self.lensed_cl_dict, l_max)
    
    def raw_cl(self, l_max):
        return self.clip_l_max(self.raw_cl_dict, l_max)

    
        
        
class CAMB_Observables:
    """
    Operate CAMB through system calls in the manner that CLASS behaves.
    """
    
    def __init__(self, parameters, fiducial, left, right, 
                 CAMB_directory, output_root,
                 CAMB_template="params.ini", CAMB_executable="camb"):
        """
        Create the CAMB_Observables class with parameters to be constrained.

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
        
        self.CAMB_template = CAMB_template
        self.CAMB_directory = CAMB_directory
        self.CAMB_executable = CAMB_executable
        
        self.parameters = parameters
        self.fiducial = fiducial
        self.left = left
        self.right = right
        self.cosmos = {}
        self.output_root = output_root
    
    def write_ini_file(self, CAMB_dict, 
                       template_file="params.ini", destination_file="temp.ini"):
        """
        Load in the template ini file and then write a new one.
        
        Takes in a dictionary with {parameter: value} pairs. Then
        iteratively looks in the template file for the parameter,
        and replaces the line with "parameter = value".
        """
        #print(f'wrting {self.CAMB_directory}/temp.ini')
        with open(self.CAMB_directory + "/" + template_file) as old, open( f'{self.CAMB_directory}/fishchips.ini', 'w') as new:
            for line in old:
                unchanged = True
                if (len(line) > 0):
                    if line[0] != '#':
                        for param in CAMB_dict:
                            if (param + " " in line) and ("=" in line):
                                new.write(f"{param} = {CAMB_dict[param]}\n")
                                unchanged = False
                                break
                if unchanged:
                    new.write(line)
    
    def compute_cosmo(self, key, CAMB_dict, debug=False):
        """
        Run CAMB with a certain set of parameters.
        """
        
        # first write a temp.ini
        self.write_ini_file(CAMB_dict, template_file=self.CAMB_template)
        
        if debug:
            print("Calling CAMB.")
        # now run CAMB
            print([f'./{self.CAMB_executable}',
                     f'fishchips.ini'], self.CAMB_directory)
        process = Popen([f'./{self.CAMB_executable}',
                     f'fishchips.ini'], 
                        stdout=PIPE, 
                        stderr=PIPE, 
                        cwd=self.CAMB_directory)
        stdout, stderr = process.communicate()
        if debug:
            print(str(stdout).replace('\\n', '\n'))
            print(str(stderr).replace('\\n', '\n'))
        
        # now read the CAMB output
        lensed = ascii.read(f"{self.CAMB_directory}{self.output_root}_lensedCls.dat",
                  names= ["ell", "tt", "ee", "bb", "te"] )
        unlensed = ascii.read(f"{self.CAMB_directory}{self.output_root}_scalCls.dat",
          names = ["ell", "tt", 'ee', 'te', 'pp', 'tp'])

        self.cosmos[key] = CAMBy(lensed, unlensed)
        
class CAMB_CMB_Primary(CMB_Primary):
    
    def get_fisher(self, camb_obs, lensed_Cl=True):
        """
        Return a Fisher matrix using a dictionary full of CAMB objects.

        This function wraps the functionality of `compute_fisher_from_spectra`,
        for use with a dictionary filled with CAMB objects.

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
        if lensed_Cl:
            fid_cl = fid_cosmo.lensed_cl(self.l_max)
        else:
            fid_cl = fid_cosmo.raw_cl(self.l_max)
        fid = {'tt': (Tcmb*1.0e6)**2 * fid_cl['tt'],
               'te': (Tcmb*1.0e6)**2 * fid_cl['te'],
               'ee': (Tcmb*1.0e6)**2 * fid_cl['ee']}

        # the primary task of this function is to compute the derivatives from `cosmos`,
        # the dictionary of computed CLASS cosmologies
        dx_array = np.array(obs.right) - np.array(obs.left)

        df = {}
        # loop over parameters, and compute derivatives
        for par, dx in zip(obs.parameters, dx_array):
            if lensed_Cl:
                cl_left = obs.cosmos[par + '_left'].lensed_cl(self.l_max)
                cl_right = obs.cosmos[par + '_right'].lensed_cl(self.l_max)
            else:
                cl_left = obs.cosmos[par + '_left'].raw_cl(self.l_max)
                cl_right = obs.cosmos[par + '_right'].raw_cl(self.l_max)

            for spec_xy in ['tt', 'te', 'ee']:
                df[par + '_' + spec_xy] = (Tcmb*1.0e6)**2 *\
                    (cl_right[spec_xy] - cl_left[spec_xy]) / dx

        return self.compute_fisher_from_spectra(fid,
                                                df,
                                                obs.parameters)
