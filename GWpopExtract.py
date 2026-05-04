import bilby as bb
import gwpopulation as gwpop
import jax
import jax.scipy.special as scs
from bilby.core.prior import PriorDict, Uniform
from bilby.hyper.model import Model
from gwpopulation.experimental.jax import JittedLikelihood
import numpy as np
import glob
import pandas as pd
import re

gwpop.set_backend("jax")

xp = gwpop.utils.xp

def read_pe(filename):
    keys = ['mass_1', 'mass_2', 'chirp_mass', 'cos_tilt_1', 'cos_tilt_2', 'a_1', 'a_2', 'redshift']
    names = ['mass_1', 'mass_2', 'chirp_mass', 'cos_tilt_1', 'cos_tilt_2', 'a_1', 'a_2', 'redshift']

    result = pd.DataFrame()

    for i in range(len(keys)):
        result[names[i]] = pd.read_hdf(filename, key=keys[i])

    return result

def power_law(xx, alpha, high, low):
    norm = xp.where(
        xp.array(alpha) == -1,
        1 / xp.log(high / low),
        (1 + alpha) / xp.array(high ** (1 + alpha) - low ** (1 + alpha)),
    )
    prob = xp.power(xx, alpha)
    prob *= norm
    prob *= (xx <= high) & (xx >= low)
    return prob

def gaussian(xx, mu, sigma, high, low):
    def logsubexp(log_p, log_q):
        return log_p + xp.log(1 - xp.exp(log_q - log_p))

    zz = xp.array(xx - mu) / sigma
    aa = xp.array(low - mu) / sigma
    bb = xp.array(high - mu) / sigma
    log_pdf = -(zz**2) / 2.0 - np.log(2.0 * np.pi) / 2.0 - xp.log(sigma)

    # cf https://github.com/scipy/scipy/blob/v1.15.1/scipy/stats/_continuous_distns.py#L10189
    log_norm = xp.select(
        [bb <= 0, aa > 0, bb > 0],
        [
            logsubexp(scs.log_ndtr(bb), scs.log_ndtr(aa)),
            logsubexp(scs.log_ndtr(-aa), scs.log_ndtr(-bb)),
            xp.log1p(-scs.ndtr(aa) - scs.ndtr(-bb)),
        ],
        xp.nan,
    )
    log_pdf -= log_norm
    return xp.nan_to_num(xp.exp(log_pdf)) * (xx >= low) * (xx <= high)

def power_law_peak_0(dataset,
        alpha,
        mmin,
        mmax):
    mass = dataset["chirp_mass"] #this can be swapped between "chirp_mass" and "mass_1" depending on what is being analyzed
    power = power_law(mass, alpha, mmax, mmin)

    return power

def power_law_peak_1(
        dataset,
        alpha,
        mmin,
        mmax,
        mu,
        sigma,
        lam):#
    mass = dataset["mass_1"] #this can be swapped between "chirp_mass" and "mass_1" depending on what is being analyzed
    power = power_law(mass, alpha, mmax, mmin)
    gauss = gaussian(mass, mu, sigma, mmax, mmin)

    return (1-lam)*power + lam * gauss

def power_law_peak_2(
        dataset,
        alpha,
        mmin,
        mmax,
        mu1,
        sigma1,
        mu2,
        sigma2,
        lam,
        lamlow):#
    mass = dataset["mass_1"] #this can be swapped between "chirp_mass" and "mass_1" depending on what is being analyzed
    power = power_law(mass, alpha, mmax, mmin)
    gauss1 = gaussian(mass, mu1, sigma1, mmax, mmin)
    gauss2 = gaussian(mass, mu2, sigma2, mmax, mmin)

    return (1-lam)*power + lam * (1-lamlow) * gauss1 + lam * lamlow * gauss2

def power_law_peak_3(
        dataset,
        alpha,
        mmin,
        mmax,
        mu1,
        sigma1,
        mu2,
        sigma2,
        mu3,
        sigma3,
        lam,
        lamlow,
        lamlower):#
    mass = dataset["mass_1"] #this can be swapped between "chirp_mass" and "mass_1" depending on what is being analyzed
    power = power_law(mass, alpha, mmax, mmin)
    gauss1 = gaussian(mass, mu1, sigma1, mmax, mmin)
    gauss2 = gaussian(mass, mu2, sigma2, mmax, mmin)
    gauss3 = gaussian(mass, mu3, sigma3, mmax, mmin)

    return (1-lam)*power + lam * (1-lamlow) * (1-lamlower) * gauss1 + lam * lamlow * gauss2 + lam * lamlower * gauss3

def power_law_peak_4(
        dataset,
        alpha,
        mmin,
        mmax,
        mu1,
        sigma1,
        mu2,
        sigma2,
        mu3,
        sigma3,
        mu4,
        sigma4,
        lam,
        lamlow,
        lamlower,
        lamlowest):#
    mass = dataset["chirp_mass"] #this can be swapped between "chirp_mass" and "mass_1" depending on what is being analyzed
    power = power_law(mass, alpha, mmax, mmin)
    gauss1 = gaussian(mass, mu1, sigma1, mmax, mmin)
    gauss2 = gaussian(mass, mu2, sigma2, mmax, mmin)
    gauss3 = gaussian(mass, mu3, sigma3, mmax, mmin)
    gauss4 = gaussian(mass, mu4, sigma4, mmax, mmin)

    return (1-lam)*power + lam * (1-lamlow) * (1-lamlower) * (1-lamlowest) * gauss1 + lam * lamlow * gauss2 + lam * lamlower * gauss3 + lam * lamlowest * gauss4


def broken_power_law(
        dataset,
        alpha,
        alpha2,
        mmin,
        mmax,
        mswap):
    mass = dataset['mass_1'] #this can be swapped between "chirp_mass" and "mass_1" depending on what is being analyzed
    m_break = mswap
    correction = power_law(m_break, alpha=alpha2, low=m_break, high=mmax) / power_law(
        m_break, alpha=alpha, low=mmin, high=m_break
    )
    low_part = power_law(mass, alpha=alpha, low=mmin, high=m_break)
    high_part = power_law(mass, alpha=alpha2, low=m_break, high=mmax)
    prob = low_part * (mass < m_break) * correction + high_part * (mass >= m_break)
    return prob / (1 + correction)

####### load posteriors ######
resultPath = "./GWTC3/*.hdf5"
pe_files = np.sort(glob.glob(resultPath))

posteriors = []

for file in pe_files:
    name = re.findall("GW......_......", file)[0]
    result = read_pe(file)
    posteriors.append(result)

posteriors = posteriors

import dill

with open("gwtc-3-injections.pkl", "rb") as ff:
    injections = dill.load(ff)

injections["mass_2"] = injections["mass_1"]/injections["mass_ratio"]
injections["chirp_mass"] = ((injections["mass_1"]*injections["mass_2"])**(3/5))/((injections["mass_1"]+injections["mass_2"])**(1/5))

model = Model(
    model_functions=[
        power_law_peak_4,
        gwpop.models.spin.iid_spin,
        gwpop.models.redshift.PowerLawRedshift(cosmo_model="Planck15"),
    ],
    cache=False,
)

priors = PriorDict()

# mass
#power law
priors["alpha"] = Uniform(minimum=-5, maximum=-1, latex_label="$\\alpha$")
priors["mmin"] = Uniform(minimum=4, maximum=10, latex_label="$m_{\\min}$")
priors["mmax"] = Uniform(minimum=100, maximum=170, latex_label="$m_{\\max}$")

#priors["alpha2"] = Uniform(minimum=-10, maximum=-1, latex_label="$\\alpha$")
#priors["mswap"] = Uniform(minimum=20, maximum=100, latex_label="$\\sigma$")

#gaussian
priors["mu1"] = Uniform(minimum=5, maximum=11, latex_label="$\\mu$")
priors["sigma1"] = Uniform(minimum=0.5, maximum=4, latex_label="$\\sigma$")
priors["lam"] = Uniform(minimum=0.3, maximum=0.6, latex_label="$\\lambda$")

priors["mu2"] = Uniform(minimum=11, maximum=20, latex_label="$\\mu$")
priors["sigma2"] = Uniform(minimum=0.5, maximum=5, latex_label="$\\sigma$")
priors["lamlow"] = Uniform(minimum=0.1, maximum=0.6, latex_label="$\\lambda$")

priors["mu3"] = Uniform(minimum=20, maximum=40, latex_label="$\\mu$")
priors["sigma3"] = Uniform(minimum=0.5, maximum=5, latex_label="$\\sigma$")
priors["lamlower"] = Uniform(minimum=0.01, maximum=0.2, latex_label="$\\lambda$")

priors["mu4"] = Uniform(minimum=40, maximum=70, latex_label="$\\mu$")
priors["sigma4"] = Uniform(minimum=0.5, maximum=5, latex_label="$\\sigma$")
priors["lamlowest"] = Uniform(minimum=0.001, maximum=0.1, latex_label="$\\lambda$")


# spin
priors["amax"] = 1
priors["alpha_chi"] = Uniform(minimum=1, maximum=6, latex_label="$\\alpha_{\\chi}$")
priors["beta_chi"] = Uniform(minimum=1, maximum=6, latex_label="$\\beta_{\\chi}$")
priors["xi_spin"] = Uniform(minimum=0, maximum=1, latex_label="$\\xi$")
priors["sigma_spin"] = Uniform(minimum=0.3, maximum=4, latex_label="$\\sigma$")

priors["lamb"] = Uniform(minimum=-1, maximum=10, latex_label="$\\lambda_{z}$")

vt = gwpop.vt.ResamplingVT(model=model, data=injections, n_events=len(posteriors))

likelihood = gwpop.hyperpe.HyperparameterLikelihood(
    posteriors=posteriors,
    hyper_prior=model,
    selection_function=vt,
)

like = -np.inf

# loops likelihood evalution as will occasionally fail
while like < -10000000:
    parameters = priors.sample()
    likelihood.log_likelihood_ratio(parameters)
    #print(likelihood.log_likelihood_ratio(parameters))
    jit_likelihood = JittedLikelihood(likelihood)
    like = jit_likelihood.log_likelihood_ratio(parameters)
    print(jit_likelihood.log_likelihood_ratio(parameters))

label = 'powerLaw+peak'
outdir = 'chirpmassGWTC3'

result = bb.run_sampler(
    likelihood=jit_likelihood,
    priors=priors,
    sampler="dynesty",
    nlive = 500,
    label=label,
    sample="acceptance-walk",
    naccept=15,
    outdir=outdir,
    save="hdf5",
)

#calculate the rate
rates = list()
for ii in range(len(result.posterior)):
    parameters = dict(result.posterior.iloc[ii])
    rates.append(float(likelihood.generate_rate_posterior_sample(parameters)))
result.posterior["rate"] = rates

result.save_to_file(filename=label+'_result.hdf5', outdir=outdir)



