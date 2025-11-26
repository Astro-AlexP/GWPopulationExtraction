from gwosc.datasets import find_datasets
import bilby
from gwpy.timeseries import TimeSeries
from gwosc import datasets
import numpy as np
import os.path

# catalogs = ['GWTC', 'GWTC-1-confident', 'GWTC-1-marginal', 'GWTC-2', 'GWTC-2.1-auxiliary'
# , 'GWTC-2.1-confident', 'GWTC-2.1-marginal', 'GWTC-3-confident', 'GWTC-3-marginal', 'GWTC-4.0', 'IAS-O3a', 'Initial_LIGO_Virgo'
# , 'O1_O2-Preliminary', 'O3_Discovery_Papers', 'O3_IMBH_marginal', 'O4_Discovery_Papers']

catalog = 'GWTC-1-confident'

events = find_datasets(type='event', catalog=catalog)

screened_events = []

for event in events:
    if event[:-3] != "GW170817":
        screened_events.append(event)

events = screened_events

for event in events:
    label = event[:-3]
    outdir = "Data/outdir_" + label

    if not os.path.exists(outdir + "/" + label + "_result.json"):

        logger = bilby.core.utils.logger
        bilby.core.utils.setup_logger(outdir=outdir, label=label)

        logger.info("Starting {}".format(label))

        trigger_time = datasets.event_gps(label)
        detectors = datasets.event_detectors(label)
        maximum_frequency = 512
        minimum_frequency = 20
        roll_off = 0.4  # Roll off duration of tukey window in seconds, default is 0.4s
        duration = 4  # Analysis segment duration
        post_trigger_duration = 2
        # Time between trigger time and end of segment
        end_time = trigger_time + post_trigger_duration
        start_time = end_time - duration

        psd_duration = 32 * duration
        psd_start_time = start_time - psd_duration
        psd_end_time = start_time

        # We now use gwpy to obtain analysis and psd data and create the ifo_list
        ifo_list = bilby.gw.detector.InterferometerList([])
        for det in detectors:
            try:
                logger.info("Downloading analysis data for ifo {}".format(det))
                ifo = bilby.gw.detector.get_empty_interferometer(det)
                data = TimeSeries.fetch_open_data(det, start_time, end_time)
                ifo.strain_data.set_from_gwpy_timeseries(data)

                failed = True
                attempts = 0
                while failed:
                    logger.info("Downloading psd data for ifo {}".format(det))
                    psd_data = TimeSeries.fetch_open_data(det, psd_start_time+attempts, psd_end_time)
                    psd_alpha = 2 * roll_off / duration
                    psd = psd_data.psd(
                        fftlength=duration, overlap=0, window=("tukey", psd_alpha), method="median"
                    )
                    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
                        frequency_array=psd.frequencies.value, psd_array=psd.value
                    )
                    ifo.maximum_frequency = maximum_frequency
                    ifo.minimum_frequency = minimum_frequency

                    if not np.isnan(np.min(psd.value)):
                        failed = False
                        ifo_list.append(ifo)

                    else:
                        logger.info("psd data contains NaN, reducing psd length")
                        attempts += 1

            except:
                logger.warning("Unable to get required data for ifo {}".format(det))

        logger.info("Saving data plots to {}".format(outdir))
        bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
        ifo_list.plot_data(outdir=outdir, label=label)

        # We now define the prior.
        # We have defined our prior distribution in a local file, GW150914.prior
        # The prior is printed to the terminal at run-time.
        # You can overwrite this using the syntax below in the file,
        # or choose a fixed value by just providing a float value as the prior.
        priors = bilby.gw.prior.BBHPriorDict(filename="GW.prior")

        # Add the geocent time prior
        priors["geocent_time"] = bilby.core.prior.Uniform(
            trigger_time - 0.1, trigger_time + 0.1, name="geocent_time"
        )

        # In this step we define a `waveform_generator`. This is the object which
        # creates the frequency-domain strain. In this instance, we are using the
        # `lal_binary_black_hole model` source model. We also pass other parameters:
        # the waveform approximant and reference frequency and a parameter conversion
        # which allows us to sample in chirp mass and ratio rather than component mass
        waveform_generator = bilby.gw.WaveformGenerator(
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments={
                "waveform_approximant": "IMRPhenomPv2",
                "reference_frequency": 50,
            },
        )

        # In this step, we define the likelihood. Here we use the standard likelihood
        # function, passing it the data and the waveform generator.
        # Note, phase_marginalization is formally invalid with a precessing waveform such as IMRPhenomPv2
        likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            ifo_list,
            waveform_generator,
            priors=priors,
            time_marginalization=False,
            phase_marginalization=False,
            distance_marginalization=True,
        )

        # Finally, we run the sampler. This function takes the likelihood and prior
        # along with some options for how to do the sampling and how to save the data
        result = bilby.run_sampler(
            likelihood,
            priors,
            sampler="dynesty",
            outdir=outdir,
            label=label,
            nlive=2000,
            naccept=60,
            sample="acceptance-walk",
            npool=48,
            check_point_delta_t=600,
            check_point_plot=True,
            conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
            result_class=bilby.gw.result.CBCResult,
        )

    else:
        print("Data for " + label + " already exists, moving to next GW")
