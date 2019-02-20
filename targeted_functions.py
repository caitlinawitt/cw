from __future__ import division
import glob
import os
import numpy as np
import cPickle as pickle
from scipy.stats import skewnorm
import copy_reg

import warnings
warnings.filterwarnings("error")

from enterprise.signals import parameter
from enterprise.pulsar import Pulsar
from enterprise.signals import selections
from enterprise.signals import signal_base
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
import enterprise.constants as const
from enterprise.signals import utils

from empirical_distributions import EmpiricalDistribution1D
from empirical_distributions import EmpiricalDistribution2D
#from empirical_distributions import EmpiricalDistribution3D

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

###Justin's original version of gw_antenna_pattern

def create_gw_antenna_pattern(theta, phi, gwtheta, gwphi):
    """
    Function to create pulsar antenna pattern functions as defined
    in Ellis, Siemens, and Creighton (2012).
    :param theta: Polar angle of pulsar location.
    :param phi: Azimuthal angle of pulsar location.
    :param gwtheta: GW polar angle in radians
    :param gwphi: GW azimuthal angle in radians
    
    :return: (fplus, fcross, cosMu), where fplus and fcross
             are the plus and cross antenna pattern functions
             and cosMu is the cosine of the angle between the 
             pulsar and the GW source.
    """

    # use definition from Sesana et al 2010 and Ellis et al 2012
    m = np.array([-np.sin(gwphi), np.cos(gwphi), 0.0])
    n = np.array([-np.cos(gwtheta)*np.cos(gwphi), 
                  -np.cos(gwtheta)*np.sin(gwphi),
                  np.sin(gwtheta)])
    omhat = np.array([-np.sin(gwtheta)*np.cos(gwphi), 
                      -np.sin(gwtheta)*np.sin(gwphi),
                      -np.cos(gwtheta)])

    phat = np.array([np.sin(theta)*np.cos(phi), 
                     np.sin(theta)*np.sin(phi), 
                     np.cos(theta)])

    fplus = 0.5 * (np.dot(m, phat)**2 - np.dot(n, phat)**2) / (1+np.dot(omhat, phat))
    fcross = (np.dot(m, phat)*np.dot(n, phat)) / (1 + np.dot(omhat, phat))
    cosMu = -np.dot(omhat, phat)

    return fplus, fcross, cosMu


@signal_base.function
def cw_delay(toas, theta, phi, pdist, p_dist=1, p_phase=None, 
             cos_gwtheta=0, gwphi=0, log10_mc=9, log10_dL=2, log10_fgw=-8, 
             phase0=0, psi=0, cos_inc=0, log10_h=None, 
             inc_psr_term=True, model='phase_approx', tref=57387*86400):
    """
    Function to create GW incuced residuals from a SMBMB as 
    defined in Ellis et. al 2012,2013.
    :param toas: Pular toas in seconds
    :param theta: Polar angle of pulsar location.
    :param phi: Azimuthal angle of pulsar location.
    :param cos_gwtheta: Cosine of Polar angle of GW source in celestial coords [radians]
    :param gwphi: Azimuthal angle of GW source in celestial coords [radians]
    :param log10_mc: log10 of Chirp mass of SMBMB [solar masses]
    :param log10_dL: log10 of Luminosity distance to SMBMB [Mpc]
    :param log10_fgw: log10 of Frequency of GW (twice the orbital frequency) [Hz]
    :param phase0: Initial Phase of GW source [radians]
    :param psi: Polarization of GW source [radians]
    :param cos_inc: cosine of Inclination of GW source [radians]
    :param p_dist: Pulsar distance to use other than those in psr [kpc]
    :param p_phase: Use pulsar phase to determine distance [radian]
    :param psrTerm: Option to include pulsar term [boolean]
    :param model: Which model to use to describe the frequency evolution
    :param tref: Reference time for phase and frequency [s]
    
    :return: Vector of induced residuals
    """
    
    # convert pulsar distance
    p_dist = (pdist[0] + pdist[1]*p_dist)*const.kpc/const.c
    
    # convert units
    mc = 10**log10_mc * const.Tsun
    gwtheta = np.arccos(cos_gwtheta)
    fgw = 10**log10_fgw
    
    # is log10_h is given, use it
    if log10_h is not None:
        dist = 2 * mc**(5/3) * (np.pi*fgw)**(2/3) / 10**log10_h
    else:
        dist = 10**log10_dL * const.Mpc / const.c

    # get antenna pattern funcs and cosMu
    fplus, fcross, cosMu = create_gw_antenna_pattern(theta, phi, gwtheta, gwphi)
    
    # get pulsar time
    toas -= tref
    if p_dist > 0:
        tp = toas-p_dist*(1.-cosMu)
    else:
        tp = toas
    
    # orbital frequency
    w0 = np.pi * fgw
    phase0 /= 2 # orbital phase
    omegadot = 96/5 * mc**(5/3) * w0**(11/3)
    
    #print('w0='+str(w0))
    #print('mc='+str(mc))
    #print('pdist='+str(p_dist))
    #print('cosMu='+str(cosMu))
    #print('in exponent='+str(1. + 256/5 * mc**(5/3) * w0**(8/3) * p_dist*(1.-cosMu)))

    if model == 'evolve':

        # calculate time dependent frequency at earth and pulsar
        omega = w0 * (1. - 256/5 * mc**(5/3) * w0**(8/3) * toas)**(-3/8)
        omega_p = w0 * (1. - 256/5 * mc**(5/3) * w0**(8/3) * tp)**(-3/8)
        
        if p_dist > 0:
            omega_p0 = w0 * (1. + 256/5 * mc**(5/3) * w0**(8/3) * p_dist*(1.-cosMu))**(-3/8)
        else:
            omega_p0 = w0

        # calculate time dependent phase
        phase = phase0 + 1/32*mc**(-5/3) * (w0**(-5/3) - omega**(-5/3))
        
        if p_phase is None:
            phase_p = phase0 + 1/32*mc**(-5/3) * (w0**(-5/3) - omega_p**(-5/3))
        else:
            phase_p = phase0 + p_phase + 1/32*mc**(-5/3) * (omega_p0**(-5/3) - omega_p**(-5/3))
        
    elif model == 'phase_approx':
        
        omega = np.pi*fgw
        phase = phase0 + omega*toas

        omega_p = w0 * (1. + 256/5 * mc**(5/3) * w0**(8/3) * p_dist*(1.-cosMu))**(-3/8)

        if p_phase is None:
            phase_p = phase0 + 1/32/mc**(5/3) * (w0**(-5/3) - omega_p**(-5/3)) + omega_p*toas
        else:
            phase_p = phase0 + p_phase + omega_p*toas
            
    else:
        omega = np.pi*fgw
        phase = phase0 + omega*toas

        omega_p = omega
        phase_p = phase0 + omega*tp
    
    # define time dependent coefficients
    At = np.sin(2*phase)*(1+cos_inc*cos_inc)
    Bt = 2*np.cos(2*phase)*cos_inc
    At_p = np.sin(2*phase_p)*(1+cos_inc*cos_inc)
    Bt_p = 2*np.cos(2*phase_p)*cos_inc

    # now define time dependent amplitudes
    alpha = mc**(5./3.)/(dist*omega**(1./3.))
    alpha_p = mc**(5./3.)/(dist*omega_p**(1./3.))
    
    # define rplus and rcross
    rplus = alpha*(At*np.cos(2*psi)+Bt*np.sin(2*psi))
    rcross = alpha*(-At*np.sin(2*psi)+Bt*np.cos(2*psi))
    rplus_p = alpha_p*(At_p*np.cos(2*psi)+Bt_p*np.sin(2*psi))
    rcross_p = alpha_p*(-At_p*np.sin(2*psi)+Bt_p*np.cos(2*psi))

    # residuals
    if inc_psr_term:
        res = fplus*(rplus_p-rplus)+fcross*(rcross_p-rcross)
    else:
        res = -fplus*rplus - fcross*rcross

    return res

def CWSignal(cw_wf, inc_psr_term=True):

    BaseClass = deterministic_signals.Deterministic(cw_wf, name='cgw')
    
    class CWSignal(BaseClass):
        
        def __init__(self, psr):
            super(CWSignal, self).__init__(psr)
            self._wf[''].add_kwarg(inc_psr_term=inc_psr_term)
            #if inc_psr_term:
            #    pdist = parameter.Normal(psr.pdist[0], psr.pdist[1])('_'.join([psr.name, 'cgw', 'pdist']))
            #    pphase = parameter.Uniform(0, 2*np.pi)('_'.join([psr.name, 'cgw', 'pphase']))
            #    self._params['p_dist'] = pdist
            #    self._params['p_phase'] = pphase
            #    self._wf['']._params['p_dist'] = pdist 
            #    self._wf['']._params['p_phase'] = pphase
    
    return CWSignal


def get_noise_from_pal2(noisefile):
    psrname = noisefile.split('/')[-1].split('_noise.txt')[0]
    fin = open(noisefile, 'r')
    lines = fin.readlines()
    params = {}
    for line in lines:
        ln = line.split()
        if 'efac' in line:
            par = 'efac'
            flag = ln[0].split('efac-')[-1]
        elif 'equad' in line:
            par = 'log10_equad'
            flag = ln[0].split('equad-')[-1]
        elif 'jitter_q' in line:
            par = 'log10_ecorr'
            flag = ln[0].split('jitter_q-')[-1]
        elif 'RN-Amplitude' in line:
            par = 'log10_A'
            flag = ''
        elif 'RN-spectral-index' in line:
            par = 'gamma'
            flag = ''
        else:
            break
        if flag:
            name = [psrname, flag, par]
        else:
            name = [psrname, par]
        pname = '_'.join(name)
        params.update({pname: float(ln[1])})
    return params

def get_global_parameters(pta):
    pars = []
    for sc in pta._signalcollections:
        pars.extend(sc.param_names)
    
    gpars = np.unique(filter(lambda x: pars.count(x)>1, pars))
    ipars = np.array([p for p in pars if p not in gpars])
        
    return gpars, ipars

## jump proposals from SJV

class JumpProposal(object):
    
    def __init__(self, pta, snames=None, fgw=3e-8, psr_dist=None, 
                 rnposteriors=None, juporbposteriors=None):
        """Set up some custom jump proposals"""
        self.params = pta.params
        self.pnames = pta.param_names
        self.npar = len(pta.params)
        self.ndim = sum(p.size or 1 for p in pta.params)
        self.psrnames = pta.pulsars
        
        # parameter map
        self.pmap = {}
        ct = 0
        for p in pta.params:
            size = p.size or 1
            self.pmap[str(p)] = slice(ct, ct+size)
            ct += size
        
        # parameter indices map
        self.pimap = {}
        for ct, p in enumerate(pta.param_names):
            self.pimap[p] = ct
        
        # collecting signal parameters across pta
        if snames is None:
            self.snames = dict.fromkeys(np.unique([[qq.signal_name for qq in pp._signals]
                                                   for pp in pta._signalcollections]))
            for key in self.snames: self.snames[key] = []

            for sc in pta._signalcollections:
                for signal in sc._signals:
                    self.snames[signal.signal_name].extend(signal.params)
            for key in self.snames: self.snames[key] = np.unique(self.snames[key]).tolist()
        else:
            self.snames = snames
            
        self.fgw = fgw
        self.psr_dist = psr_dist

        # initialize empirical distributions for the red noise parameters from a previous MCMC
        if rnposteriors is not None and os.path.isfile(rnposteriors):
            with open(rnposteriors) as f:
                self.rnDistr = pickle.load(f)
                self.rnDistr_psrnames = [ r.param_names[0].split('_')[0] for r in self.rnDistr ]
        else:
            self.rnDistr = None
            self.rnDistr_psrnames = None

        # initialize empirical distributions for the Jupiter orbital elements from a previous MCMC
        if juporbposteriors is not None and os.path.isfile(juporbposteriors):
            with open(juporbposteriors) as f:
                self.juporbDistr = pickle.load(f)
        else:
            self.juporbDistr = None
            
    def draw_from_prior(self, x, iter, beta):
        """Prior draw.
        
        The function signature is specific to PTMCMCSampler.
        """
        
        q = x.copy()
        lqxy = 0
        
        # randomly choose parameter
        idx = np.random.randint(0, self.npar)
        
        # if vector parameter jump in random component
        param = self.params[idx]
        if param.size:
            idx2 = np.random.randint(0, param.size)
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        # scalar parameter
        else:
            q[idx] = param.sample()
        
        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(q[self.pmap[str(param)]])
                
        return q, float(lqxy)

    def draw_from_ephem_prior(self, x, iter, beta):
        q = x.copy()
        lqxy = 0
        
        signal_name = 'phys_ephem'
        
        # draw parameter from signal model
        param = np.random.choice(self.snames[signal_name])
        if param.size:
            idx2 = np.random.randint(0, param.size)
            
            q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

            lqxy = param.get_logpdf(x[self.pmap[str(param)]]) \
                    - param.get_logpdf(q[self.pmap[str(param)]])
    
        # scalar parameter
        else:
            q[self.pmap[str(param)]] = param.sample()
            
            lqxy = param.get_logpdf(x[self.pmap[str(param)]]) \
                    - param.get_logpdf(q[self.pmap[str(param)]])
        
        return q, float(lqxy)

    def draw_from_jup_orb_priors(self, x, iter, beta):
        q = x.copy()
        lqxy = 0
        
        idx = 0
        for i,p in enumerate(self.params):

            if p.name == 'jup_orb_elements':
                idx = i
        
        # draw parameter from signal model
        param = self.params[idx]
        idx2 = np.random.randint(0, param.size)
            
        q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) \
                    - param.get_logpdf(q[self.pmap[str(param)]])
        
        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(q[self.pmap[str(param)]])

        return q, float(lqxy)
    
    def draw_from_jup_orb_1d_posteriors(self, x, iter, beta):
        q = x.copy()
        lqxy = 0
        
        # randomly choose one of the 1D posteriors of the Jupiter orbital elements
        j = np.random.randint(0, len(self.juporbDistr))
        print(type(self.juporbDistr[j]))
        
        idx = self.pimap[str(self.juporbDistr[j].param_name)]
        
        q[idx] = self.juporbDistr[j].draw()
            
        lqxy = self.juporbDistr[j].logprob(x[idx]) - self.juporbDistr[j].logprob(q[idx])

        return q, float(lqxy)
    
    def draw_from_ephem_posteriors(self, x, iter, beta):
        q = x.copy()
        lqxy = 0
        
        idx = 0
        for i,p in enumerate(self.params):
            if 'jup_orb' in p.name:
                idx = i
        myparam = self.params[idx]
        
        # randomly choose one of the ephemeris posteriors
        j = np.random.randint(0, len(self.juporbDistr))
        indices = [ self.pimap[str(p)] for p in self.juporbDistr[j].param_names ]
        oldvals = [ x[idx2] for idx2 in indices ]
        newvals = self.juporbDistr[j].draw()
        
        for idx2,n in zip(indices,newvals):
            q[idx2] = n
            
        lqxy = self.juporbDistr[j].logprob(oldvals) - self.juporbDistr[j].logprob(newvals)

        return q, float(lqxy)
    
    def draw_skyposition(self, x, iter, beta):
    
        q = x.copy()
        lqxy = 0
        
        # jump in both cos_gwtheta and gwphi, drawing both values from the prior
        cw_params = ['cos_gwtheta', 'gwphi']
        for cp in cw_params:
            
            idx = 0
            for i,p in enumerate(self.params):

                if p.name == cp:
                    idx = i
        
            # draw parameter from signal model
            param = self.params[idx]
            q[self.pmap[str(param)]] = param.sample()
        
        return q, float(lqxy)

    def draw_from_cw_log_uniform_distribution(self, x, iter, beta):
    
        q = x.copy()
        lqxy = 0
        
        idx = self.pnames.index('log10_mc')
        q[idx] = np.random.uniform(7, 10)
        
        return q, float(lqxy)
    def draw_from_cw_log_uniform_distribution_freq(self, x, iter, beta):
    
        q = x.copy()
        lqxy = 0
        
        idx = self.pnames.index('log10_fgw')
        q[idx] = np.random.uniform(-9, np.log10(3*10**(-7)))
        
        return q, float(lqxy)
    
    def draw_from_cw_log_uniform_distribution_strain(self, x, iter, beta):
    
        q = x.copy()
        lqxy = 0
        
        idx = self.pnames.index('log10_h')
        q[idx] = np.random.uniform(-18, -11)
        
        return q, float(lqxy)

    def draw_from_cw_prior(self, x, iter, beta):
    
        q = x.copy()
        lqxy = 0
        
        # randomly choose one of the cw parameters and get index
        cw_params = [ p for p in self.pnames if p in ['cos_gwtheta', 'cos_inc', 'gwphi', 'log10_mc', 
                                                      'phase0', 'psi', 'log10_h', 'log10_fgw']]
        myparam = np.random.choice(cw_params)
        idx = 0
        for i,p in enumerate(self.params):

            if p.name == myparam:
                idx = i
        
        # draw parameter from signal model
        param = self.params[idx]
        q[self.pmap[str(param)]] = param.sample()
        
        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(q[self.pmap[str(param)]])

        return q, float(lqxy)
    
    def draw_from_mass_freq_prior(self, x, iter, beta):
    
        q = x.copy()
        lqxy = 0
        
        # randomly choose one of the cw parameters and get index
        cw_params = [ p for p in self.pnames if p in ['log10_mc', 'log10_fgw']]
        myparam = np.random.choice(cw_params)
        idx = 0
        for i,p in enumerate(self.params):

            if p.name == myparam:
                idx = i
        
        # draw parameter from signal model
        param = self.params[idx]
        q[self.pmap[str(param)]] = param.sample()
        
        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(q[self.pmap[str(param)]])

        return q, float(lqxy)

    
    def draw_from_rnposteriors(self, x, iter, beta):
        
        q = x.copy()
        lqxy = 0
        
        # randomly pick a pulsar
        mypsr = np.random.choice(self.psrnames)
        
        if self.rnDistr is not None and mypsr in self.rnDistr_psrnames:
                
            i = self.rnDistr_psrnames.index(mypsr)
            
            oldsample = [x[self.pnames.index(mypsr + '_gamma')], 
                         x[self.pnames.index(mypsr + '_log10_A')]]
                
            newsample = self.rnDistr[i].draw() 
            
            q[self.pnames.index(mypsr + '_gamma')] = newsample[0]
            q[self.pnames.index(mypsr + '_log10_A')] = newsample[1]
            
            # forward-backward jump probability
            lqxy = self.rnDistr[i].logprob(oldsample) - self.rnDistr[i].logprob(newsample)
                
        else:

            # if there is no empirical distribution for this pulsar's red noise parameters, 
            # choose one of the red noise parameters and draw a sample from the prior
            myparam = np.random.choice([mypsr + '_gamma', mypsr + '_log10_A'])
            idx = 0
            for i,p in enumerate(self.params):

                if p.name == myparam:
                    idx = i

            # draw parameter from signal model
            param = self.params[idx]
            q[self.pmap[str(param)]] = param.sample()
        
            # forward-backward jump probability
            lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(q[self.pmap[str(param)]])
        
        return q, float(lqxy)
    
    def draw_from_rnpriors(self, x, iter, beta):
        
        q = x.copy()
        lqxy = 0
        
        # randomly pick a pulsar
        mypsr = np.random.choice(self.psrnames)
        
        myparam = np.random.choice([mypsr + '_gamma', mypsr + '_log10_A'])
        idx = 0
        for i,p in enumerate(self.params):

            if p.name == myparam:
                idx = i

        # draw parameter from signal model
        param = self.params[idx]
        q[self.pmap[str(param)]] = param.sample()
        
        return q, float(lqxy)
    
    def draw_from_pdist_prior(self, x, iter, beta):
        
        q = x.copy()
        lqxy = 0
        
        # randomly pick a pulsar
        myparam = np.random.choice([p for p in self.pnames if 'p_dist' in p])
        
        idx = 0
        for i,p in enumerate(self.params):

            if p.name == myparam:
                idx = i

        # draw parameter from signal model
        param = self.params[idx]
        q[self.pmap[str(param)]] = param.sample()
        
        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(q[self.pmap[str(param)]])
        
        return q, float(lqxy)

    def draw_from_pphase_prior(self, x, iter, beta):
        
        q = x.copy()
        lqxy = 0
        
        # randomly pick a pulsar
        myparam = np.random.choice([p for p in self.pnames if 'p_phase' in p])
        
        idx = 0
        for i,p in enumerate(self.params):

            if p.name == myparam:
                idx = i

        # draw parameter from signal model
        param = self.params[idx]
        q[self.pmap[str(param)]] = param.sample()
        
        # forward-backward jump probability
        lqxy = param.get_logpdf(x[self.pmap[str(param)]]) - param.get_logpdf(q[self.pmap[str(param)]])
        
        return q, float(lqxy)
    
    def draw_gwtheta_comb(self, x, iter, beta):
        
        q = x.copy()
        lqxy = 0
        
        # the variance of the Gaussian we are drawing from is very small
        # to account for the comb-like structure of the posterior
        sigma = const.c/self.fgw/const.kpc
        
        # now draw an integer to go to a nearby spike
        N = int(0.1/sigma)
        n = np.random.randint(-N,N)
        newval = np.arccos(x[self.pnames.index('cos_gwtheta')]) \
                    + (sigma/2)*np.random.randn() + n*sigma
        
        q[self.pnames.index('cos_gwtheta')] = np.cos(newval)
                
        return q, float(lqxy)

    def draw_gwphi_comb(self, x, iter, beta):
        
        # this jump takes into account the comb-like structure of the likelihood 
        # as a function of gwphi, with sharp spikes superimposed on a smoothly-varying function
        # the width of these spikes is related to the GW wavelength
        # this jump does two things:
        #  1. jumps an integer number of GW wavelengths away from the current point
        #  2. draws a step size from a Gaussian with variance equal to half the GW wavelength, 
        #     and takes a small step from its position in a new spike
        
        q = x.copy()
        lqxy = 0
        
        # compute the GW wavelength
        sigma = const.c/self.fgw/const.kpc
        
        # now draw an integer to go to a nearby spike
        # we need to move over a very large number of spikes to move appreciably in gwphi
        # the maximum number of spikes away you can jump 
        # corresponds to moving 0.1 times the prior range
        idx = 0
        for i,p in enumerate(self.params):
            if p.name == 'gwphi':
                idx = i
        N = int(0.1*(self.params[idx]._pmax - self.params[idx]._pmin)/sigma)
        n = np.random.randint(-N,N)
        
        q[self.pnames.index('gwphi')] = x[self.pnames.index('gwphi')] + (sigma/2)*np.random.randn() + n*sigma

        return q, float(lqxy)

    def draw_strain_skewstep(self, x, iter, beta):
        
        q = x.copy()
        lqxy = 0
        
        a = 2
        s = 1
        
        diff = skewnorm.rvs(a, scale=s)
        q[self.pnames.index('log10_h')] = x[self.pnames.index('log10_h')] - diff
        lqxy = skewnorm.logpdf(-diff, a, scale=s) - skewnorm.logpdf(diff, a, scale=s)
        
        return q, float(lqxy)

    def draw_strain_inc(self, x, iter, beta):
        
        q = x.copy()
        lqxy = 0
        
        # half of the time, jump so that you conserve h*(1 + cos_inc^2)
        # the rest of the time, jump so that you conserve h*cos_inc
        
        which_jump = np.random.random()
        

        

        if 'log10_h' in self.pnames:
            
            if which_jump > 0.5:
            
                q[self.pnames.index('cos_inc')] = np.random.uniform(-1,1)
                q[self.pnames.index('log10_h')] = x[self.pnames.index('log10_h')] \
                                                    + np.log10(1+x[self.pnames.index('cos_inc')]**2) \
                                                    - np.log10(1+q[self.pnames.index('cos_inc')]**2)
                        
            else:
                
                # if jumping to conserve h*cos_inc, make sure the sign of cos_inc does not change
                if x[self.pnames.index('cos_inc')] > 0:
                    q[self.pnames.index('cos_inc')] = np.random.uniform(0,1)
                else:
                    q[self.pnames.index('cos_inc')] = np.random.uniform(-1,0)
        
                q[self.pnames.index('log10_h')] = x[self.pnames.index('log10_h')] \
                                                    + np.log10(x[self.pnames.index('cos_inc')]/q[self.pnames.index('cos_inc')])
        elif 'log10_fgw' in self.pnames:
            
            if which_jump > 0.5:
            
                q[self.pnames.index('cos_inc')] = np.random.uniform(-1,1)
                ratio =  np.log10(1+x[self.pnames.index('cos_inc')]**2) - np.log10(1+q[self.pnames.index('cos_inc')]**2)

                                               
            else:
                
                # if jumping to conserve h*cos_inc, make sure the sign of cos_inc does not change
                if x[self.pnames.index('cos_inc')] > 0:
                    q[self.pnames.index('cos_inc')] = np.random.uniform(0,1)
                else:
                    q[self.pnames.index('cos_inc')] = np.random.uniform(-1,0)
        
                ratio = np.log10(x[self.pnames.index('cos_inc')]/q[self.pnames.index('cos_inc')])

            cw_params = [ p for p in self.pnames if p in ['log10_mc', 'log10_fgw']]
            myparam = np.random.choice(cw_params)
            
            idx = 0
            for i,p in enumerate(self.params):
                 if p.name == myparam:
                    idx = i
            param = self.params[idx]
            
            if myparam == 'log10_mc':
                q[self.pnames.index('log10_mc')] = q[self.pmap[str(param)]] = param.sample()
                q[self.pnames.index('log10_fgw')] = 3/2*(-5/3*q[self.pnames.index('log10_mc')] \
                                                        +2/3*x[self.pnames.index('log10_fgw')] \
                                                        +5/3*x[self.pnames.index('log10_mc')] \
                                                        + ratio)

            else:
                q[self.pnames.index('log10_fgw')] = q[self.pmap[str(param)]] = param.sample()
                q[self.pnames.index('log10_mc')] = 3/5*(-2/3*q[self.pnames.index('log10_fgw')] \
                                                        +2/3*x[self.pnames.index('log10_fgw')] \
                                                        +5/3*x[self.pnames.index('log10_mc')] \
                                                        + ratio)
                    
            
        else:
        
            if which_jump > 0.5:
            
                q[self.pnames.index('cos_inc')] = np.random.uniform(-1,1)
                q[self.pnames.index('log10_mc')] = x[self.pnames.index('log10_mc')] \
                                                    + 3/5*np.log10(1+x[self.pnames.index('cos_inc')]**2) \
                                                    - 3/5*np.log10(1+q[self.pnames.index('cos_inc')]**2)
                        
            else:
                
                # if jumping to conserve h*cos_inc, make sure the sign of cos_inc does not change
                if x[self.pnames.index('cos_inc')] > 0:
                    q[self.pnames.index('cos_inc')] = np.random.uniform(0,1)
                else:
                    q[self.pnames.index('cos_inc')] = np.random.uniform(-1,0)
        
                q[self.pnames.index('log10_mc')] = x[self.pnames.index('log10_mc')] \
                                                    + 3/5*np.log10(x[self.pnames.index('cos_inc')]/q[self.pnames.index('cos_inc')])
                    
        return q, float(lqxy)
    
    def draw_strain_psi(self, x, iter, beta):
        
        q = x.copy()
        lqxy = 0
        
        # draw a new value of psi, then jump in log10_h so that either h*cos(2*psi) or h*sin(2*psi) are conserved
        which_jump = np.random.random()
        
        if 'log10_h' in self.pnames:
            if which_jump > 0.5:
                # jump so that h*cos(2*psi) is conserved            
                # make sure that the sign of cos(2*psi) does not change
                if x[self.pnames.index('psi')] > 0.25*np.pi and x[self.pnames.index('psi')] < 0.75*np.pi:
                    q[self.pnames.index('psi')] = np.random.uniform(0.25*np.pi,0.75*np.pi)
                else:
                    newval = np.random.uniform(-0.25*np.pi,0.25*np.pi)
                    if newval < 0:
                        newval += np.pi
                    q[self.pnames.index('psi')] = newval
                    
                ratio = np.cos(2*x[self.pnames.index('psi')])/np.cos(2*q[self.pnames.index('psi')])
                q[self.pnames.index('log10_h')] = x[self.pnames.index('log10_h')] + np.log10(ratio)       
                
            else:
                # jump so that h*sin(2*psi) is conserved            
                # make sure that the sign of sin(2*psi) does not change
                if x[self.pnames.index('psi')] < np.pi/2:
                    q[self.pnames.index('psi')] = np.random.uniform(0,np.pi/2)
                else:
                    q[self.pnames.index('psi')] = np.random.uniform(np.pi/2,np.pi)
                    
                ratio = np.sin(2*x[self.pnames.index('psi')])/np.sin(2*q[self.pnames.index('psi')])
                q[self.pnames.index('log10_h')] = x[self.pnames.index('log10_h')] + np.log10(ratio)
        elif 'log10_fgw' in self.pnames:
            if which_jump > 0.5:
                # jump so that h*cos(2*psi) is conserved            
                # make sure that the sign of cos(2*psi) does not change
                if x[self.pnames.index('psi')] > 0.25*np.pi and x[self.pnames.index('psi')] < 0.75*np.pi:
                    q[self.pnames.index('psi')] = np.random.uniform(0.25*np.pi,0.75*np.pi)
                else:
                    newval = np.random.uniform(-0.25*np.pi,0.25*np.pi)
                    if newval < 0:
                        newval += np.pi
                    q[self.pnames.index('psi')] = newval
                    
                ratio = np.cos(2*x[self.pnames.index('psi')])/np.cos(2*q[self.pnames.index('psi')])

                
            else:
                # jump so that h*sin(2*psi) is conserved            
                # make sure that the sign of sin(2*psi) does not change
                if x[self.pnames.index('psi')] < np.pi/2:
                    q[self.pnames.index('psi')] = np.random.uniform(0,np.pi/2)
                else:
                    q[self.pnames.index('psi')] = np.random.uniform(np.pi/2,np.pi)
                    
                ratio = np.sin(2*x[self.pnames.index('psi')])/np.sin(2*q[self.pnames.index('psi')])
                
            # draw one and calculate the other!!!
            cw_params = [ p for p in self.pnames if p in ['log10_mc', 'log10_fgw']]
            myparam = np.random.choice(cw_params)
            
            idx = 0
            for i,p in enumerate(self.params):
                 if p.name == myparam:
                    idx = i
            param = self.params[idx]
            
            if myparam == 'log10_mc':
                q[self.pnames.index('log10_mc')] = q[self.pmap[str(param)]] = param.sample()
                q[self.pnames.index('log10_fgw')] = 3/2*(-5/3*q[self.pnames.index('log10_mc')] \
                                                        +2/3*x[self.pnames.index('log10_fgw')] \
                                                        +5/3*x[self.pnames.index('log10_mc')] \
                                                        + np.log10(ratio))

            else:
                q[self.pnames.index('log10_fgw')] = q[self.pmap[str(param)]] = param.sample()
                q[self.pnames.index('log10_mc')] = 3/5*(-2/3*q[self.pnames.index('log10_fgw')] \
                                                        +2/3*x[self.pnames.index('log10_fgw')] \
                                                        +5/3*x[self.pnames.index('log10_mc')] \
                                                        + np.log10(ratio))
        else:
            if which_jump > 0.5:
                # jump so that h*cos(2*psi) is conserved            
                # make sure that the sign of cos(2*psi) does not change
                if x[self.pnames.index('psi')] > 0.25*np.pi and x[self.pnames.index('psi')] < 0.75*np.pi:
                    q[self.pnames.index('psi')] = np.random.uniform(0.25*np.pi,0.75*np.pi)
                else:
                    newval = np.random.uniform(-0.25*np.pi,0.25*np.pi)
                    if newval < 0:
                        newval += np.pi
                    q[self.pnames.index('psi')] = newval
                    
                ratio = np.cos(2*x[self.pnames.index('psi')])/np.cos(2*q[self.pnames.index('psi')])
                q[self.pnames.index('log10_mc')] = x[self.pnames.index('log10_mc')] + 3/5*np.log10(ratio)       
                
            else:
                # jump so that h*sin(2*psi) is conserved            
                # make sure that the sign of sin(2*psi) does not change
                if x[self.pnames.index('psi')] < np.pi/2:
                    q[self.pnames.index('psi')] = np.random.uniform(0,np.pi/2)
                else:
                    q[self.pnames.index('psi')] = np.random.uniform(np.pi/2,np.pi)
                    
                ratio = np.sin(2*x[self.pnames.index('psi')])/np.sin(2*q[self.pnames.index('psi')])
                q[self.pnames.index('log10_mc')] = x[self.pnames.index('log10_mc')] + 3/5*np.log10(ratio)
                
        return q, float(lqxy)
    
    def phase_psi_reverse_jump(self, x, iter, beta):
        
        q = x.copy()
        lqxy = 0

        param = np.random.choice([str(p) for p in self.pnames if 'phase' in p])
        
        if param == 'phase0':
            q[self.pnames.index('phase0')] = np.mod(x[self.pnames.index('phase0')] + np.pi, 2*np.pi)
            q[self.pnames.index('psi')] = np.mod(x[self.pnames.index('psi')] + np.pi/2, np.pi)
        else:
            q[self.pnames.index(param)] = np.mod(x[self.pnames.index(param)] + np.pi, 2*np.pi)
                
        return q, float(lqxy)
    
    def fix_cyclic_pars(self, prepar, postpar, iter, beta):
        
        q = postpar.copy()
        
        for param in self.params:
            if 'phase' in param.name:
                q[self.pmap[str(param)]] = np.mod(postpar[self.pmap[str(param)]], 2*np.pi)
            elif param.name == 'psi':
                q[self.pmap[str(param)]] = np.mod(postpar[self.pmap[str(param)]], np.pi)
            elif param.name == 'gwphi':
                if param._pmin == 0 and param._pmax == 2*np.pi:
                    q[self.pmap[str(param)]] = np.mod(postpar[self.pmap[str(param)]], 2*np.pi)
                
        return q, 0

    def fix_psr_dist(self, prepar, postpar, iter, beta):
        
        q = postpar.copy()
        
        for param in self.params:
            if 'p_dist' in param.name:
                
                psr_name = param.name.split('_')[0]
                
                while self.psr_dist[psr_name][0] + self.psr_dist[psr_name][1]*q[self.pmap[str(param)]] < 0:
                    q[self.pmap[str(param)]] = param.sample()
                
        return q, 0
    
    ###this if for freq sampling###
    

    

    
    
def get_parameter_groups(pta, rnpsrs=None):
    """Utility function to get parameter groupings for sampling."""
    ndim = len(pta.param_names)
    groups  = [range(0, ndim)]
    params = pta.param_names

    snames = np.unique([[qq.signal_name for qq in pp._signals] 
                        for pp in pta._signalcollections])
    
    # sort parameters by signal collections
    ephempars = []
    rnpars = []
    cwpars = []
    wnpars = []

    for sc in pta._signalcollections:
        for signal in sc._signals:
            if signal.signal_name == 'red noise':
                rnpars.extend(signal.param_names)
            elif signal.signal_name == 'phys_ephem':
                ephempars.extend(signal.param_names)
            elif signal.signal_name == 'cgw':
                cwpars.extend(signal.param_names)
            elif signal.signal_name == 'efac':
                wnpars.extend(signal.param_names)
            elif signal.signal_name == 'equad':
                wnpars.extend(signal.param_names)
                
    
    if 'red noise' in snames:
    
        # create parameter groups for the red noise parameters
        rnpsrs = [ p.split('_')[0] for p in params if '_log10_A' in p and 'gwb' not in p]

        for psr in rnpsrs:
            groups.extend([[params.index(psr + '_gamma'), params.index(psr + '_log10_A')]])
            
        groups.extend([[params.index(p) for p in rnpars]])
    #addition for sampling wn
    #this groups efac and equad together for each pulsar
    if 'efac' and 'equad' in snames:
    
        # create parameter groups for the red noise parameters
        wnpsrs = [ p.split('_')[0] for p in params if '_efac' in p]

        for psr in wnpsrs:
            groups.extend([[params.index(psr + '_efac'), params.index(psr + '_log10_equad')]])
            
        groups.extend([[params.index(p) for p in wnpars]])
        
    if 'efac' and 'equad' and 'red noise' in snames:
    
        # create parameter groups for the red noise parameters
        psrs = [ p.split('_')[0] for p in params if '_efac' in p and '_log10_A' in p and 'gwb' not in p]

        for psr in psrs:
            groups.extend([[params.index(psr + '_efac'), params.index(psr + '_log10_equad'),
                            params.index(psr + '_gamma'), params.index(psr + '_log10_A')]])
            
                    
    # set up groups for the BayesEphem parameters
    if 'phys_ephem' in snames:
        
        ephempars = np.unique(ephempars)
        juporb = [p for p in ephempars if 'jup_orb' in p]
        groups.extend([[params.index(p) for p in ephempars if p not in juporb]])
        groups.extend([[params.index(jp) for jp in juporb]])
        for i1 in range(len(juporb)):
            for i2 in range(i1+1, len(juporb)):
                groups.extend([[params.index(p) for p in [juporb[i1], juporb[i2]]]])
        
    if 'cgw' in snames:
    
        # divide the cgw parameters into two groups: 
        # the common parameters and the pulsar phase and distance parameters
        cw_common = np.unique(list(filter(lambda x: cwpars.count(x)>1, cwpars)))
        groups.extend([[params.index(cwc) for cwc in cw_common]])

        cw_pulsar = np.array([p for p in cwpars if p not in cw_common])
        if len(cw_pulsar) > 0:
            
            pdist_params = [ p for p in cw_pulsar if 'p_dist' in p ]
            pphase_params = [ p for p in cw_pulsar if 'p_phase' in p ]
            
            for pd,pp in zip(pdist_params,pphase_params):
                #groups.extend([[params.index(pd), params.index('cos_gwtheta'), params.index('gwphi')]])
                groups.extend([[params.index(pd), params.index('log10_mc')]])
                groups.extend([[params.index(pp), params.index('phase0'), params.index('log10_mc')]])
                groups.extend([[params.index(pp), params.index('phase0'), params.index('log10_mc'), 
                                params.index('cos_inc'), params.index('psi')]])
                groups.extend([[params.index(pd), params.index(pp), 
                                params.index('log10_mc')]])
                if 'log10_fgw' in cw_common:
                    groups.extend([[params.index(pd), params.index('log10_fgw')]])
                    groups.extend([[params.index(pp), params.index('phase0'), params.index('log10_fgw')]])
                    groups.extend([[params.index(pp), params.index('phase0'), params.index('log10_fgw'), 
                                    params.index('cos_inc'), params.index('psi')]])
                    groups.extend([[params.index(pd), params.index(pp), 
                                    params.index('log10_fgw')]])
                    
                    groups.extend([[params.index(pd), params.index(pp), 
                                    params.index('log10_fgw'), params.index('log10_mc')]])
                    groups.extend([[params.index(pp), params.index('phase0'), params.index('log10_mc'), params.index('log10_fgw')]])
                    groups.extend([[params.index(pp), params.index('phase0'), params.index('log10_mc'), 
                                    params.index('cos_inc'), params.index('psi'), params.index('log10_fgw')]])
            
        # now try other combinations of the common cgw parameters
        ###add f_gw as a combo for frequency sampling - save for now
        combos = [['cos_gwtheta', 'gwphi'], 
                  ['cos_gwtheta', 'gwphi', 'log10_mc'], 
                  ['cos_gwtheta', 'gwphi', 'psi'], 
                  ['log10_mc', 'cos_gwtheta', 'gwphi'], 
                  ['log10_mc', 'cos_inc', 'phase0', 'psi'],
                  ['log10_h', 'cos_gwtheta', 'gwphi'], 
                  ['log10_h', 'cos_inc', 'phase0', 'psi'], 
                  ['log10_h', 'phase0'], 
                  ['log10_mc', 'phase0'], 
                  ['cos_inc', 'phase0'], 
                  ['phase0', 'psi'],
                  
                  ['log10_mc', 'cos_inc', 'phase0', 'psi', 'log10_fgw'], 
                  ['log10_mc', 'phase0', 'log10_fgw'], 
                  ['cos_inc', 'phase0', 'log10_fgw'], 
                  ['phase0', 'psi','log10_fgw'],
                  ['log10_mc', 'log10_fgw'],
                  ['cos_gwtheta', 'gwphi','log10_fgw'],
                  ['log10_fgw', 'cos_inc', 'phase0', 'psi'],
                  ['log10_fgw', 'phase0'],
                  
                  ['log10_mc', 'cos_inc', 'phase0', 'psi', 'log10_fgw','gwb_log10_A' ], 
                  ['log10_mc', 'phase0', 'log10_fgw','gwb_log10_A' ], 
                  ['cos_inc', 'phase0', 'log10_fgw','gwb_log10_A' ], 
                  ['phase0', 'psi','log10_fgw','gwb_log10_A' ],
                  ['log10_mc', 'log10_fgw','gwb_log10_A' ],
                  ['cos_gwtheta', 'gwphi','log10_fgw','gwb_log10_A' ],
                  ['log10_fgw', 'cos_inc', 'phase0', 'psi','gwb_log10_A' ],
                  ['log10_fgw', 'phase0','gwb_log10_A' ]]
                
        
        for combo in combos:
            if all(c in cw_common for c in combo):
                groups.extend([[params.index(c) for c in combo]])

    if 'cgw' in snames and 'phys_ephem' in snames:
        # add a group that contains the Jupiter orbital elements and the common GW parameters
        juporb = list([p for p in ephempars if 'jup_orb' in p])

        cw_common = list(np.unique(list(filter(lambda x: cwpars.count(x)>1, cwpars))))

        
        myparams = juporb + cw_common
        
        groups.extend([[params.index(p) for p in myparams]])
                
    return groups
