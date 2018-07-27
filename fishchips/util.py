"""Plotting utilities and miscellaneous functions."""

import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

ALPHA1 = 1.52
ALPHA2 = 2.48
ALPHA3 = 3.44
"""float: These three constants are multiplied with 1D sigma to get 2D 68%/95%/99% contours."""

PLOT_MULT = 4.
"""float: Ratio between plot window and sigma."""

PRECISE_CLASS_DICT = {
                 'neglect_CMB_sources_below_visibility': 0.001,
                 'perturb_sampling_stepsize': 0.01,
                 'reionization_optical_depth_tol': 1e-07,
                 'tol_background_integration': 0.001,
                 'tol_perturb_integration': 1e-7,
                 'tol_thermo_integration': 1e-7,
                 'transfer_neglect_delta_k_S_e': 0.11,
                 'transfer_neglect_delta_k_S_t0': 0.15,
                 'transfer_neglect_delta_k_S_t1': 0.04,
                 'transfer_neglect_delta_k_S_t2': 0.15,
                 'k_max_tau0_over_l_max': 6,
                 'accurate_lensing': 1}
"""dict: Built-in high precision settings. This will take a long time to compute!"""


def get_ellipse(par1, par2, params, cov, scale1=1, scale2=1):
    """
    Extract ellipse parameters from covariance matrix.

    Parameters
    ----------
        par1 (string): name of parameter 1
        par2 (string): name of parameter 2
        params (list of strings): contains names of parameters to constrain
        cov (numpy array): covariance matrix

    Return
    ------
        tuple, ellipse a, b, angle in degrees, sigma_x, sigma_y, sigma_xy

    """
    # equations 1-4 Coe 2009. returns in degrees
    # first look up indices of parameters
    pind = dict(zip(params, list(range(len(params)))))
    i1 = pind[par1]
    i2 = pind[par2]
    sigma_x2 = cov[i1, i1] * scale1*scale1
    sigma_y2 = cov[i2, i2] * scale2*scale2
    sigma_xy = cov[i1, i2] * scale1*scale2

    if ((sigma_y2/sigma_x2) < 1e-10) or ((sigma_x2/sigma_y2) < 1e-10):
        a2 = max(sigma_x2, sigma_y2) + sigma_xy**2 / max(sigma_x2, sigma_y2)
        b2 = min(sigma_x2, sigma_y2) - sigma_xy**2 / max(sigma_x2, sigma_y2)
    else:
        a2 = (sigma_x2+sigma_y2)/2. + np.sqrt((sigma_x2 - sigma_y2)**2/4. +
                                              sigma_xy**2)
        b2 = (sigma_x2+sigma_y2)/2. - np.sqrt((sigma_x2 - sigma_y2)**2/4. +
                                              sigma_xy**2)
    angle = np.arctan(2.*sigma_xy/(sigma_x2-sigma_y2)) / 2.
    if (sigma_x2 < sigma_y2):
        a2, b2 = b2, a2

    return np.sqrt(a2), np.sqrt(b2), angle * 180.0 / np.pi, \
        np.sqrt(sigma_x2), np.sqrt(sigma_y2), sigma_xy


def plot_ellipse(ax, par1, par2, obs, cov, color='black',
                 resize_lims=True, positive_definite=[], one_sigma_only=False,
                 scale1=1, scale2=1, ls1='--', ls2='-'):
    """
    Plot 1 and 2-sigma ellipses, from Coe 2009.

    Parameters
    ----------
        ax (matpotlib axis): axis upon which the ellipses will be drawn
        par1 (string): parameter 1 name
        par2 (string): parameter 2 name
        obs (Observables object): contains names of parameters to constrain, etc
        cov (numpy array): covariance matrix
        color (string): color to plot ellipse with
        resize_lims (boolean): flag for changing the axis limits
        positive_definite (list of string): convenience input,
            parameter names passed in this list will be cut off at 0 in plots.
        scale1 and scale2 are for plotting scale

    Returns
    -------
        list of float : sigma_x, sigma_y, sigma_xy for judging the size of the
            plotting window

    """
    params = obs.parameters
    pind = dict(zip(params, list(range(len(params)))))
    i1 = pind[par1]
    i2 = pind[par2]
    a, b, theta, sigma_x, sigma_y, sigma_xy = get_ellipse(
        par1, par2, params, cov, scale1, scale2)

    fid1 = obs.fiducial[i1] * scale1
    fid2 = obs.fiducial[i2] * scale2

    if not one_sigma_only:
        e1 = Ellipse(
            xy=(fid1, fid2),
            width=a * 2 * ALPHA2, height=b * 2 * ALPHA2,
            angle=theta, edgecolor=color, lw=2, facecolor='none', ls=ls2)
        ax.add_artist(e1)
        e1.set_clip_box(ax.bbox)

    # 1-sigma ellipse
    e2 = Ellipse(
        xy=(fid1, fid2),
        width=a*2*ALPHA1, height=b*2*ALPHA1,
        angle=theta, edgecolor=color, lw=2, facecolor='none', ls=ls1)
    ax.add_artist(e2)
    e2.set_alpha(1.0)
    e2.set_clip_box(ax.bbox)

    if resize_lims:
        if par1 in positive_definite:
            ax.set_xlim(max(0.0, -PLOT_MULT*sigma_x),
                        fid1+PLOT_MULT*sigma_x)
        else:
            ax.set_xlim(fid1 - PLOT_MULT * sigma_x,
                        fid1 + PLOT_MULT * sigma_x)
        if par2 in positive_definite:
            ax.set_ylim(max(0.0, fid2 - PLOT_MULT * sigma_y),
                        fid2 + PLOT_MULT * sigma_y)
        else:
            ax.set_ylim(fid2 - PLOT_MULT * sigma_y,
                        fid2 + PLOT_MULT * sigma_y)

    return sigma_x, sigma_y, sigma_xy


def plot_triangle(obs, cov, f=None, ax=None, color='black', positive_definite=[],
                  labels=None, scales=None):
    """
    Makes a standard triangle plot.

    Args:
        f (optional,matplotlib figure): pass this if you already have a figure
        ax (optional,matplotlib axis): existing axis
    """
    nparams = len(obs.parameters)
    if scales is None:
        scales = np.ones(nparams)

    if ax is None or f is None:
        print('generating new axis')
        f, ax = plt.subplots(nparams, nparams, figsize=(12, 12))

    if labels is None:
        labels = [(r'$\mathrm{' + p.replace('_', r'\_') + r'}$')
                  for p in obs.parameters]
    print(labels)
    # stitch together axes to row=nparams-1 and col=0
    # and turn off non-edge
    for ii in range(nparams):
        for jj in range(nparams):
            if ii == jj:
                ax[jj, ii].get_yaxis().set_visible(False)
                if ii < nparams-1:
                    ax[jj, ii].get_xaxis().set_ticks([])

            if ax[jj, ii] is not None:
                if ii < jj:
                    if jj < nparams-1:
                        ax[jj, ii].set_xticklabels([])
                    if ii > 0:
                        ax[jj, ii].set_yticklabels([])

                    if jj > 0:
                        # stitch axis to the one above it
                        if ax[0, ii] is not None:
                            ax[jj, ii].get_shared_x_axes().join(
                                ax[jj, ii], ax[0, ii])
                    elif ii < nparams-1:
                        if ax[jj, nparams-1] is not None:
                            ax[jj, ii].get_shared_y_axes().join(
                                ax[jj, ii], ax[jj, nparams-1])

    # call plot_ellipse
    for ii in range(nparams):
        for jj in range(nparams):
            if ax[jj, ii] is not None:
                if ii < jj:
                    plot_ellipse(ax[jj, ii], obs.parameters[ii],
                                 obs.parameters[jj], obs, cov, color=color,
                                 positive_definite=positive_definite,
                                 scale1=scales[ii], scale2=scales[jj])
                    if jj == nparams-1:
                        ax[jj, ii].set_xlabel(labels[ii], labelpad=30)
                        for tick in ax[jj, ii].get_xticklabels():
                            tick.set_rotation(45)
                    if ii == 0:
                        ax[jj, ii].set_ylabel(labels[jj])
                elif ii == jj:
                    # plot a gaussian if we're on the diagonal
                    sig = np.sqrt(cov[ii, ii])
                    if obs.parameters[ii] in positive_definite:
                        grid = np.linspace(
                            obs.fiducial[ii],
                            obs.fiducial[ii] + PLOT_MULT * sig, 100)
                    else:
                        grid = np.linspace(
                            obs.fiducial[ii] - PLOT_MULT*sig,
                            obs.fiducial[ii] + PLOT_MULT*sig, 100)
                    posmult = 1.0
                    if obs.parameters[ii] in positive_definite:
                        posmult = 2.0
                    ax[jj, ii].plot(grid,
                                    posmult * np.exp(
                                       -(grid-obs.fiducial[ii])**2 /
                                       (2 * sig**2)) / (sig * np.sqrt(2*np.pi)),
                                    '-', color=color)
                    if ii == nparams-1:
                        ax[jj, ii].set_xlabel(labels[ii], labelpad=30)
                else:
                    ax[jj, ii].axis('off')

    return f, ax


# From Tram's cosmology notebooks on neutrinos with CLASS.
def get_masses(dmsq_atm, sum_masses, hierarchy):
    """
    Compute separate masses given a sum of neutrino mass.
    
    Parameters
    ----------
        dmsq_atm (float) : squared mass difference between
            the large gap between species
        sum_masses (float): sum of the masses of the neutrino
            species
        hierarchy (string): specify 'NH' for normal or 'IH' 
            for inverted. actually just needs an 'n' in the 
            string for normal, else inverted.
            
    Returns
    -------
        float, float corresponding to 2,1 degeneracy masses

    """
    if 'n' in hierarchy.lower():
        #Normal hierarchy:
        r = np.roots([3,-4*sum_masses,sum_masses**2-dmsq_atm])
        m_minus = min(r)
    else:
        r = np.roots([3,2*sum_masses,4*dmsq_atm-sum_masses**2])
        m_minus = max(r)
    return m_minus, np.sqrt(dmsq_atm+m_minus**2)

def neutrino_dict(input_dict, dmsq_atm=2e-3, hierarchy='NH'):
    """
    Process a dictionary and replace sum_mnu.
    
    To get CLASS to get separate neutrino species, you need to change
    the dictionary a bit. We make a copy and replace sum_mnu with 
    m_ncdm and deg_ncdm.
    
    Parameters
    ----------
        input_dict (dict) : a dictionary for calling CLASS, 
            contains the key sum_mnu
        dmsq_atm (float) : squared mass gap from atmospheric measurements
        hierarchy (string) : specify normal or inverted hierarchy, use
            'NH' or 'IH'.
    
    Returns
    -------
        dict : processed dict with m_ncdm and dec_ncdm set to the 
            right things

    """
    neut_dict = input_dict.copy()
    if 'sum_mnu' in input_dict:
        sum_masses = input_dict['sum_mnu']
        del neut_dict['sum_mnu']
        [mm, mp] = get_masses(dmsq_atm, sum_masses, hierarchy)

        neut_dict['m_ncdm'] = str(mm)+','+str(mp)
        neut_dict['deg_ncdm'] = '2,1'
    
    neut_dict['N_ncdm'] = 2
    return neut_dict

