from pycbc.io import FieldArray
import numpy as np
import tempfile
from pycbc.inject import RingdownHDFInjectionSet
from pycbc.psd import aLIGOZeroDetHighPower, from_txt, from_string
from pycbc.types.frequencyseries import FrequencySeries
from pycbc.noise import noise_from_psd


class RingdownInjections:
    """
    Create a set of damped sinusoid injections that can be added to data.

    These ring-down signals will just assume one mode, the fundamental
    l=2, m=2, n=0 mode.

    tcs: float, array_like
        A value, or a set of values, giving the peak time(s) of any ring-down
        signal injections (GPS seconds).
    freqs: float, array_like
        A value, or set of values, giving the fundamental mode frequency(ies)
        of the ring-down injections (Hz).
    amps: float, array_like
        A value, or set of values, giving the fundamental mode amplitude(s) of
        the ring-down injections (strain). 
    taus: float, array_like
        A value, or set of values, giving the decay time of the fundamental
        mode of the ring-down injections (s).
    ras: float, array_like
        A value, or set of values, giving the right ascension(s) of the
        ring-down source(s) (rads).
    decs: float, array_like
        A value, or set of values, giving the declinations(s) of the
        ring-down source(s) (rads).
    psis: float, array_like
        A value, or set of values, giving the polarisation angle(s) of the
        ring-down source(s) (rads). Defaults to zero.
    phis: float, array_like
        A value, or set of values, giving the initial phase(s) of the ring-down
        signal(s) (rads). Defaults to zero.
    inclinations: float, array_like
        A value, or set of values, giving the inclination angle(s) of the
        ring-down source(s) (rads). Defaults to pi/2 rads (90 degs).
    detector: str
        A string giving the detector name. If this is given simulated time
        series data will be created (based on the subsequent arguments) and the
        signals will be added into it. Examples are "H1", "L1" or "V1" for the
        LIGO Hanford detector, LIGO Livingston detector, and Virgo detector,
        respectively.
    starttime: float
        The start time of the simulated data (GPS seconds). Make sure the
        injection `tcs` values are consistent with this. Default is 1000000000.
    duration: float
        The duration of the simulated data (s).  Make sure the injection `tcs`
        values are consistent with this. Default is 1 second.
    deltat: float
        The sample time step of the simulated data (s). Default is 1/8192 s,
        i.e., a Nyquist frequency of 4096 Hz.
    psd: FrequencySeries, callable, str, float
        A value setting the PSD from which to generate the noise. PyCBC PSD
        functions or FrequencySeries objects can be passed, or strings giving
        files containing PSDs or valid LALSimulation aliases for analytical
        PSDs can be used, or a single float can be used to have a uniform PSD
        as a function of frequency.
    asd: str, float
        If you have a file containing an ASD rather than a PSD, or want to pass
        a single ASD value, use this rather than the psd option.
    flow: float
        The low frequency cut-off of the PSD (Hz). Defaults to 20 Hz.
    """

    def __init__(
        self,
        tcs,
        freqs,
        amps,
        taus,
        ras,
        decs,
        psis=0.0,
        phis=0.0,
        inclinations=np.pi / 2.,
        detector=None,
        starttime=1000000000,
        duration=1.0,
        deltat=1 / 8192,
        psd=aLIGOZeroDetHighPower,
        asd=None,
        flow=20.0,
    ):
        # get the number of injections (use injection times, tcs)
        if isinstance(tcs, float):
            tcs = np.asarray([tcs])
        self.ninj = len(tcs)

        # set up a FieldArray to contain required ring-down parameters
        self.__injections = FieldArray(
            self.ninj,
            dtype=[
                ("approximant", "S20"),
                ("f_220", "<f8"),
                ("lmns", "S3"),
                ("tau_220", "<f8"),
                ("amp220", "<f8"),
                ("phi220", "<f8"),
                ("polarization", "<f8"),
                ("inclination", "<f8"),
                ("ra", "<f8"),
                ("dec", "<f8"),
                ("tc", "<f8"),
            ]
        )

        pnames = ["f_220", "tau_220", "amp220", "phi220", "ra", "dec", "inclination", "polarization", "tc"]

        for param, pvalue in zip(
            pnames,
            [freqs, taus, amps, phis, ras, decs, inclinations, psis, tcs],
        ):
            if isinstance(pvalue, float):
                self.__injections[param] = np.full(self.ninj, pvalue)
            elif isinstance(pvalue, (list, np.ndarray)):
                if len(pvalue) != self.ninj:
                    raise ValueError("{} must have the same number of entries as 'tc'".format(param))

                self.__injections[param] = pvalue
            else:
                raise TypeError("Input must be a float or list")

        # the PyCBC "TdQNMfromFreqTau" approximant creates time-domain ring-down signals
        self.__injections["approximant"] = np.full(self.ninj, "TdQNMfromFreqTau")
        self.__injections["lmns"] = np.full(self.ninj, "221")  # use 1 22 mode (the 220 mode)

        # create the injections
        self.create_injections()

        # create the simulated data if requested
        if detector is not None:
            self.create_data(detector, starttime, duration, deltat, psd, asd=asd, flow=flow)

            # inject signal(s) into the data
            self.inject()

    def create_injections(self, filename=None):
        """
        Create the injection via writing and reading to a HDF5 file. If a
        filename is given then that will be used to write the file. If it is
        not given then a temporary file will be used that will be deleted once
        the injections are loaded.

        Parameters
        ----------
        filename: str
            The name of a file, with the hdf5 suffix, to output the injections to.
        """

        if filename is None:
            # create temporary file for injection
            tmpfile = tempfile.TemporaryFile(suffix="hdf5")
        else:
            tmpfile = filename

        # output the injection
        RingdownHDFInjectionSet.write(tmpfile, self.__injections)

        # re-read in the injections
        self.injection_set = RingdownHDFInjectionSet(tmpfile)

    def inject(self, data=None, detector=None):
        """
        Inject the ring-down signals into data.
        
        Parameters
        ----------
        data: TimeSeries
            The time series data into which to inject the signals. This will
            be changed by the function.
        detector: str
            The name of the detector.

        """

        if data is not None:
            self.data = data

        if detector is not None:
            self.detector = detector

        # a copy of the data before injection
        self.__noise_only = self.data.copy()

        self.injection_set.apply(self.data, detector_name=self.detector)

        # get a version of the data that just contains the injection
        self.injection_data = self.data.copy() - self.__noise_only

    @property
    def psd(self):
        return self.__psd

    @psd.setter
    def psd(self, psd):
        if callable(psd):
            self.__psd = psd(self.nsamples, self.deltaf, self.flow)
        elif isinstance(psd, FrequencySeries):
            # make sure values below flow are zero
            kmin = int(self.flow / self.deltaf)
            psd.data[:kmin] = 0
            self.__psd = psd
        elif isinstance(psd, str):
            # try reading PSD from file
            try:
                self.__psd = from_txt(
                    psd,
                    self.nsamples,
                    self.deltaf,
                    self.flow,
                    is_asd_file=self.is_asd_file,
                )
            except Exception as e1:
                # try getting PSD from string name
                try:
                    self.__psd = from_string(psd, self.nsamples, self.deltaf, self.flow)
                except Exception as e2:
                    raise IOError("Could not create PSD: {}\n{}".format(e1, e2))
        elif isinstance(psd, float):
            # convert single float into PSD FrequencySeries
            val = psd ** 2 if self.is_asd_file else psd
            psds = np.full(self.nsamples, val)
            
            kmin = int(self.flow / self.deltaf)
            psds[:kmin] = 0
            self.__psd = FrequencySeries(psds, delta_f=self.deltaf)
        else:
            raise TypeError("Could not create PSD from supplied input")

    def create_data(self, detector, starttime, duration, deltat, psd, asd=None, flow=20.0):
        """
        Create a fake time series of data for a given detector based on a given
        power spectral density.

        Parameters
        ----------
        detector: str
            A string giving the detector name.
        starttime: float
            A start time for the simulated data in GPS seconds.
        duration: float
            The duration of the data in seconds.
        deltat: float
            The time step for the time series.
        psd: FrequencySeries, callable, str, float
            A value setting the PSD from which to generate the noise. PyCBC PSD
            functions or FrequencySeries objects can be passed, or strings giving
            files containing PSDs or valid LALSimulation aliases for analytical
            PSDs can be used, or a single float can be used to have a uniform PSD
            as a function of frequency.
        asd: str, float
            If you have a file containing an ASD rather than a PSD, or want to pass
            a single ASD value, use this rather than the psd option.
        flow: float
            The low frequency cut-off of the PSD (Hz). Defaults to 20 Hz.
        """

        self.detector = detector

        # set data information for use when creating PSD
        self.flow = flow
        self.duration = duration
        self.deltat = deltat
        self.nsamples = int(self.duration // self.deltat)

        self.samplerate = 1.0 / self.deltat
        self.nyquist = self.samplerate / 2.0
        self.deltaf = self.samplerate / self.nsamples

        if asd is None:
            self.is_asd_file = False
            self.psd = psd
        else:
            # using file containing ASD rather than PSD
            self.is_asd_file = True
            self.psd = asd

        # create the noise time series
        self.data = noise_from_psd(self.nsamples, self.deltat, self.psd)
        self.data.start_time = starttime
