import numpy as np
import gwpopulation as gwpop
from gwpopulation.utils import to_numpy
from bilby.core.prior import PriorDict, Uniform
from bilby.hyper.model import Model
import bilby
import matplotlib.pyplot as plt
import jax.scipy.special as scs

gwpop.set_backend("jax")

xp = gwpop.utils.xp

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
    mass = dataset["mass_1"]
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
    mass = dataset["mass_1"]
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
    mass = dataset["mass_1"]
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
    mass = dataset["mass_1"]
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
    mass = dataset["mass_1"]
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
    mass = dataset['mass_1']
    m_break = mswap
    correction = power_law(m_break, alpha=alpha2, low=m_break, high=mmax) / power_law(
        m_break, alpha=alpha, low=mmin, high=m_break
    )
    low_part = power_law(mass, alpha=alpha, low=mmin, high=m_break)
    high_part = power_law(mass, alpha=alpha2, low=m_break, high=mmax)
    prob = low_part * (mass < m_break) * correction + high_part * (mass >= m_break)
    return prob / (1 + correction)

fig, axs = plt.subplots(1, 1, figsize=(12, 4))


file_name = 'mass1BPL/powerLaw'
result = bilby.result.read_in_result(filename=file_name + '_result.hdf5')

mass_1 = np.linspace(2, 150, 1000)
mass_ratio = np.linspace(0.001, 1, 500)
mass_1_grid, mass_ratio_grid = np.meshgrid(mass_1, mass_ratio)

data = dict(
                mass_1=mass_1_grid,
                mass_ratio=mass_ratio_grid,
                mass_2=mass_1_grid * mass_ratio_grid,
            )


model = Model(
    model_functions=[
        broken_power_law,
        #gwpop.models.spin.iid_spin,
        #gwpop.models.redshift.PowerLawRedshift(cosmo_model="Planck15"),
    ],
    cache=False,
)


lines = dict(mass_1=list(), mass_ratio=list())
ppd = np.zeros_like(data["mass_1"])

samples = result.posterior

for ii in range(len(samples)):
    parameters = dict(samples.iloc[ii])
    model.parameters.update(parameters)
    prob = model.prob(data)
    prob *= parameters["rate"]
    ppd += prob

    mass_1_prob = np.trapezoid(prob, mass_ratio, axis=0)
    mass_ratio_prob = np.trapezoid(prob, mass_1, axis=-1)

    lines["mass_1"].append(mass_1_prob)
    lines["mass_ratio"].append(mass_ratio_prob)

for key in lines:
    lines[key] = np.vstack([to_numpy(line) for line in lines[key]])

ppd /= len(samples)
ppd = to_numpy(ppd)

mass_1 = to_numpy(mass_1)
mass_ratio = to_numpy(mass_ratio)

mass_1_ppd = np.trapezoid(ppd, mass_ratio, axis=0)
mass_ratio_ppd = np.trapezoid(ppd, mass_1, axis=-1)

axs.semilogy(mass_1, mass_1_ppd, label='test', color='black')
axs.fill_between(
    mass_1,
    np.percentile(lines["mass_1"], 5, axis=0),
    np.percentile(lines["mass_1"], 95, axis=0),
    alpha=0.5,
    color='black'
)
axs.set_xlim(2, 100)
axs.set_ylim(5*(10**-5), 5*(10**1))
#axs.set_xscale('log')
axs.set_xlabel("$m_{1}$ [$M_{\\odot}$]")
#axs.legend(bbox_to_anchor=(0.5, 1.15), loc="upper center")
ylabel = "$\\frac{d\\mathcal{R}}{dm_{1}}$ [Gpc$^{-3}$yr$^{-1}M_{\\odot}^{-1}$]"
axs.set_ylabel(ylabel)

#vlines
#axs.axvline(x = 12.96, color = 'r')
#axs.text(13.5, 0.0005, "13", fontsize=12)

#axs.axvline(x = 21.27, color = 'r')
#axs.text(21.7, 0.0005, "21.3", fontsize=12)

#axs.axvline(x = 27.07, color = 'r')
#axs.text(27.7, 0.0005, "27.7", fontsize=12)

#axs.axvline(x = 41.49, color = 'r')
#axs.text(42.2, 0.0005, "41.5", fontsize=12)

fig.show()
fig.savefig(file_name + '_spec.png')





