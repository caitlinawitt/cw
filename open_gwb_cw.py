
# coding: utf-8

# In[1]:
from __future__ import division
import numpy as np
import glob, json
from astropy import units as u
from astropy.coordinates import SkyCoord
import scipy as sp
import os
import cPickle as pickle

import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals

import corner
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

from io import StringIO

import libstempo as T
import libstempo.plot as LP

import targeted_functions as fns 





# In[2]:


datadir = '/scratch/caw0057/mdc_2/mdc2/group1/dataset_3b/'
fitdir = '/scratch/caw0057/mdc_2/mdc2/group1/3b_fitted/'
noisedir = '/scratch/caw0057/mdc_2/mdc2/group1/'

noisefile = noisedir + 'group1_psr_noise.json'
pars = sorted(glob.glob(fitdir+'*.par'))
tims = sorted(glob.glob(fitdir+'*.tim'))

psrs = []
for p, t in zip(pars, tims):
    psr = Pulsar(p, t)
    psrs.append(psr)


with open(noisefile, 'r') as fp:
        noise = json.load(fp)
setpars = {}
for i in noise:
    for j in noise[i]:
        if 'efac' in j:
            par = i+'_efac'
            val = noise[i][j]
        elif 'equad' in j:
            par = i+'_log10_equad'
            val = noise[i][j]
        elif 'log10_A' in j:
            par = i + '_log10_A'
            val = noise[i][j]
        elif 'spec_ind' in j:
            par = i + '_gamma'
            val = noise[i][j]
        setpars[par] = val

tmin = [p.toas.min() for p in psrs]
tmax = [p.toas.max() for p in psrs]
Tspan = np.max(tmax) - np.min(tmin)


##### parameters and priors #####

# Uniform prior on EFAC
efac = parameter.Constant()
equad = parameter.Constant()

# red noise parameters
# Uniform in log10 Amplitude and in spectral index
log10_A = parameter.Constant()
gamma = parameter.Constant()


# GWB parameters (initialize with names to set as common signal)
gwb_log10_A = parameter.Uniform(-18,-12)('gwb_log10_A')
gwb_gamma = parameter.Constant(13./3)('gwb_gamma')

##### Set up signals #####

# white noise
ef = white_signals.MeasurementNoise(efac=efac)
eq = white_signals.EquadNoise(log10_equad=equad)

# red noise (powerlaw with 30 frequencies)
pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
rn = gp_signals.FourierBasisGP(spectrum=pl, components=30)

# timing model
tm = gp_signals.TimingModel()


# GWB with hellings and downs correlations
cpl = utils.powerlaw(log10_A=gwb_log10_A, gamma=gwb_gamma)
orf = utils.hd_orf()
gwb = gp_signals.FourierBasisCommonGP(cpl, orf, components=30, Tspan=Tspan, name='gw')

# cw params
cos_gwtheta = parameter.Uniform(-1,1)('cos_gwtheta')
gwphi = parameter.Uniform(0,2*np.pi)('gwphi')
#log10_mc = parameter.LinearExp(7,10)('log10_mc')
log10_mc = parameter.Uniform(7,10)('log10_mc')
#log10_fgw = BoundedNormal(log_f, log_f_err, -9, np.log10(3*10**(-7)))('log10_fgw')
log10_fgw = parameter.Uniform(-9, np.log10(3*10**(-7)))('log10_fgw')

phase0 = parameter.Uniform(0, 2*np.pi)('phase0')
psi = parameter.Uniform(0, np.pi)('psi')
cos_inc = parameter.Uniform(-1, 1)('cos_inc')

##sarah's change
p_phase = parameter.Uniform(0, 2*np.pi)
p_dist = parameter.Normal(0, 1)

#log10_h = parameter.LinearExp(-18, -11)('log10_h')
log10_h = parameter.Uniform(-18, -11)('log10_h')
#log10_dL = parameter.Constant(np.log10(85.8))('log10_dL')



cw_wf = fns.cw_delay(cos_gwtheta=cos_gwtheta, gwphi=gwphi, log10_mc=log10_mc, 
                 log10_h=log10_h, log10_fgw=log10_fgw, phase0=phase0,
                 psi=psi, cos_inc=cos_inc, p_dist = p_dist, p_phase = p_phase, model = 'evolve', tref = np.max(tmax))
cw = fns.CWSignal(cw_wf, inc_psr_term=True)

# full model is sum of components
model = ef + eq + rn + tm + gwb + cw


# In[36]:


    
psrs = []
for p, t in zip(pars, tims):
    psr = Pulsar(p, t)
    psrs.append(psr)

# initialize PTA
pta = signal_base.PTA([model(psr) for psr in psrs])

pta.set_default_params(setpars)


xs = {par.name: par.sample() for par in pta.params}


# dimension of parameter space
ndim= len(pta.param_names)

# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.01**2)

# set up jump groups by red noise groups
groups = fns.get_parameter_groups(pta)
groups.extend([[pta.param_names.index('gwb_log10_A')]])


outdir = '/scratch/caw0057/mdc_2/chains/combo4'

# intialize sampler
sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, groups=groups,
                 outDir=outdir, resume = False)

with open(outdir+'/parameters.json', 'w') as fp:
    json.dump(pta.param_names, fp)

##from sarah
inc_psr_term = True
##why this??
if inc_psr_term:
    psr_dist = {}
    for psr in psrs:
        psr_dist[psr.name] = psr.pdist
else:
    psr_dist = None


init = None
initdir = '/users/caw0057/empirical_dist/'
    # add prior draws to proposal cycle
#if any('_log10_A' in p for p in pta.param_names):
    #rnposteriors = initdir + 'rn_distr.pkl'
#else:
rnposteriors = None
        
eph = False
if eph:
    juporbposteriors = initdir + 'unix_jup_orb_distr_1D.pkl'
else:
    juporbposteriors = None

    
jp = fns.JumpProposal(pta, psr_dist=psr_dist, rnposteriors=rnposteriors, juporbposteriors=juporbposteriors)

#if any('_log10_A' in p for p in pta.param_names):
#    sampler.addProposalToCycle(jp.draw_from_rnposteriors, 30)
#    sampler.addProposalToCycle(jp.draw_from_rnpriors, 30)
    
sampler.addProposalToCycle(jp.draw_from_cw_prior, 20) #pick a cw param & jump
sampler.addProposalToCycle(jp.draw_from_cw_log_uniform_distribution, 5) #draw from uniform Mc
sampler.addProposalToCycle(jp.draw_from_cw_log_uniform_distribution_strain, 5) #draw from uniform strain

sampler.addProposalToCycle(jp.draw_from_cw_log_uniform_distribution_freq, 5) #draw from uniform freq
#sampler.addProposalToCycle(jp.draw_skyposition, 10)
#    sampler.addProposalToCycle(jp.draw_strain_skewstep, 10)
#    sampler.addProposalToCycle(jp.draw_gwphi_comb, 5)
#    sampler.addProposalToCycle(jp.draw_gwtheta_comb, 5)
##these aren't needed for constant sky position
if 'phase0' in pta.param_names and 'psi' in pta.param_names:
    sampler.addProposalToCycle(jp.phase_psi_reverse_jump, 1)
if 'psi' in pta.param_names:
    sampler.addProposalToCycle(jp.draw_strain_psi, 2)
if 'cos_inc' in pta.param_names:
    sampler.addProposalToCycle(jp.draw_strain_inc, 2)
        
if inc_psr_term:
    sampler.addProposalToCycle(jp.draw_from_pdist_prior, 30)
    sampler.addProposalToCycle(jp.draw_from_pphase_prior, 30)
    
if eph:
    sampler.addProposalToCycle(jp.draw_from_ephem_prior, 10)
#        sampler.addProposalToCycle(jp.draw_from_jup_orb_priors, 20)
    sampler.addProposalToCycle(jp.draw_from_jup_orb_1d_posteriors, 20)
    #sampler.addProposalToCycle(jp.draw_from_ephem_posteriors, 20)
    
sampler.addAuxilaryJump(jp.fix_cyclic_pars)


#jp = JumpProposal(pta)
sampler.addProposalToCycle(jp.draw_from_prior, 5) #pick a param & jump




# convert dictionary to array
x0 = np.hstack(p.sample() for p in pta.params)

N = int(1e6)
sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)


# In[54]:




