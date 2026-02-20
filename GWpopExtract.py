import bilby as bb
import gwpopulation as gwpop
import jax
import matplotlib.pyplot as plt
import pandas as pd
from bilby.core.prior import PriorDict, Uniform
from bilby.hyper.model import Model
from gwpopulation.experimental.jax import JittedLikelihood
import numpy as np
import glob
import pandas as pd

gwpop.set_backend("jax")

xp = gwpop.utils.xp

def read_pe(filename):
    keys = ['mass_1', 'mass_ratio', 'a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2', 'redshift'] #, 'prior'

    result = pd.DataFrame()

    for key in keys:
        result[key] = pd.read_hdf(filename, key=key)

    return result

####### load posteriors ######
resultPath = "./pe/*.hdf5"
pe_files = np.sort(glob.glob(resultPath))

posteriors = []

for file in pe_files:
    result = read_pe(file)
    posteriors.append(result)

import dill

with open("gwtc-3-injections.pkl", "rb") as ff:
    injections = dill.load(ff)

print(injections.keys())
model = Model(
    model_functions=[
        gwpop.models.mass.two_component_primary_mass_ratio,
        gwpop.models.spin.iid_spin,
        gwpop.models.redshift.PowerLawRedshift(cosmo_model="Planck15"),
    ],
    cache=False,
)

vt = gwpop.vt.ResamplingVT(model=model, data=injections, n_events=len(posteriors))

likelihood = gwpop.hyperpe.HyperparameterLikelihood(
    posteriors=posteriors,
    hyper_prior=model,
    selection_function=vt,
)


priors = PriorDict()

# mass
priors["alpha"] = Uniform(minimum=-2, maximum=4, latex_label="$\\alpha$")
priors["beta"] = Uniform(minimum=-4, maximum=12, latex_label="$\\beta$")
priors["mmin"] = Uniform(minimum=2, maximum=2.5, latex_label="$m_{\\min}$")
priors["mmax"] = Uniform(minimum=80, maximum=100, latex_label="$m_{\\max}$")
priors["lam"] = Uniform(minimum=0, maximum=1, latex_label="$\\lambda_{m}$")
priors["mpp"] = Uniform(minimum=10, maximum=50, latex_label="$\\mu_{m}$")
priors["sigpp"] = Uniform(minimum=1, maximum=10, latex_label="$\\sigma_{m}$")
priors["gaussian_mass_maximum"] = 100
# spin
priors["amax"] = 1
priors["alpha_chi"] = Uniform(minimum=1, maximum=6, latex_label="$\\alpha_{\\chi}$")
priors["beta_chi"] = Uniform(minimum=1, maximum=6, latex_label="$\\beta_{\\chi}$")
priors["xi_spin"] = Uniform(minimum=0, maximum=1, latex_label="$\\xi$")
priors["sigma_spin"] = Uniform(minimum=0.3, maximum=4, latex_label="$\\sigma$")

priors["lamb"] = Uniform(minimum=-1, maximum=10, latex_label="$\\lambda_{z}$")




parameters = priors.sample()
likelihood.log_likelihood_ratio(parameters)
print(likelihood.log_likelihood_ratio(parameters))
jit_likelihood = JittedLikelihood(likelihood)
print(jit_likelihood.log_likelihood_ratio(parameters))
print(jit_likelihood.log_likelihood_ratio(parameters))



result = bb.run_sampler(
    likelihood=jit_likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=100,
    label="cosmo",
    sample="acceptance-walk",
    naccept=5,
    save="hdf5",
)
