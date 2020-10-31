import numpy as np
from pycbc.types import TimeSeries
import pycbc.waveform
from pycbc.psd import aLIGOZeroDetHighPower, from_txt, from_string
from pycbc.detector import Detector
from pycbc.types.frequencyseries import FrequencySeries
from pycbc.noise import noise_from_psd


class RingdownTemplateBank:
    """
    A class to generate a template bank for ring-down signals. This uses the
    metric as defined in Creighton, PRD, 60, 022001 (1999) arXiv:gr-qc/9901084.

    Parameters
    ----------
    flow: float
        Minimum frequency to use (Hz)
    fhigh: float
        Maximum frequency to use (Hz)
    qrange: array_like
        Array containing a minimum and maximum Q values
    taurange: array_like
        Array containing a minimum and maximum decay time value (can use
        instead of Q) (s)
    mm: float
        Metric mismatch to use.
    """

    def __init__(self, flow, fhigh, qrange=None, taurange=None, mm=0.03):
        self.flow = flow
        self.fhigh = fhigh

        # set the q range
        self.set_qrange(qrange, taurange)

        # set the mismatch
        self.mm = mm

        # create the template bank
        _ = self.generate_bank()

        self.__idx = 0

    def __getitem__(self, idx):
        return self.bank_freqs[idx], self.bank_qs[idx]

    def __iter__(self):
        for freq, q in zip(self.bank_freqs, self.bank_qs):
            yield freq, q

    @property
    def flow(self):
        return self.__flow

    @flow.setter
    def flow(self, flow):
        if not isinstance(flow, (float, int)):
            raise TypeError("flow must be a number")
        elif flow < 0.0:
            raise ValueError("flow must be positive")
        self.__flow = flow
        self.philow = np.log(flow)

    @property
    def fhigh(self):
        return self.__fhigh

    @fhigh.setter
    def fhigh(self, fhigh):
        if not isinstance(fhigh, (float, int)):
            raise TypeError("fhigh must be a number")
        elif fhigh <= self.flow:
            raise ValueError("fhigh must greater than flow")

        self.__fhigh = fhigh
        self.phihigh = np.log(fhigh)

    def set_qrange(self, qrange=None, taurange=None):
        """
        Set the Q range for the template bank.
        """

        if qrange is None and taurange is None:
            raise ValueError("Must specify either Q range or rau range")

        if qrange is not None:
            try:
                qmin = np.min(qrange)
                qmax = np.max(qrange)
            except Exception as e:
                raise TypeError(
                    "Could not get minimum and maximum values from qrange: {}".format(e)
                )

            if qmin >= qmax:
                raise ValueError("Q min is greater than Q max")

            self.qrange = [qmin, qmax]
        else:
            try:
                taumin = np.min(taurange)
                taumax = np.max(taurange)
            except Exception as e:
                raise TypeError(
                    "Could not get minimum and maximum values from taurange: {}".format(
                        e
                    )
                )

            if taumin >= taumax:
                raise ValueError("tau min is greater than tau max")

            Qs = sorted(
                np.einsum(
                    "i,j->ij", [taumin, taumax], [self.flow, self.fhigh]
                ).flatten()
                * np.pi
            )
            self.qrange = [Qs[0], Qs[-1]]

    @property
    def mm(self):
        return self.__mm

    @mm.setter
    def mm(self, mm):
        if not isinstance(mm, float):
            raise TypeError("Mismatch must be a float")
        elif mm <= 0.0 or mm >= 1.0:
            raise ValueError("Mismatch must be between 0 and 1")

        self.__mm = mm

    def generate_bank(self, usetau=False):
        """
        Generate the template bank.

        Parameters
        ----------
        usetau: bool
            Output values of tau rather than Q (default is False).

        Returns
        -------
        bank: tuple
            A tuple containing two lists: frequencies, Q/tau values
        """

        freqs = []
        qs = []
        taus = []

        qcur = self.qrange[0]

        qscol = []
        while qcur < self.qrange[1]:
            qscol.append(qcur)
            qcur += np.sqrt(2 * self.mm / self.gqq(qcur))

        for q in qscol:
            dphi = np.sqrt(2 * self.mm / self.gphiphi(q))
            phis = np.arange(self.philow, self.phihigh, dphi)
            curfreqs = np.exp(phis)
            freqs.extend(curfreqs.tolist())
            if not usetau:
                qs.extend(np.full(len(phis), q).tolist())
            else:
                taus.extend((q / (np.pi * curfreqs)).tolist())

        self.bank_freqs = freqs
        self.bank_taus = taus
        self.bank_qs = qs

        if not usetau:
            return freqs, qs
        else:
            return freqs, taus

    @staticmethod
    def gqq(q):
        """
        Generate the g_QQ metric component (see Eq. 4.40, 4.41 of arXiv:0908.2085).

        Parameters
        ----------
        q: float, array_like
            The value(s) of Q for the metric.

        Returns
        -------
        gqq: float
            The metric value(s).
        """

        if isinstance(q, (list, tuple, np.ndarray)):
            qs = np.asarray(q)
        else:
            qs = q

        return (1.0 / 8.0) * (3 + 16 * qs ** 4) / ((qs * (1 + 4 * qs ** 2)) ** 2)

    @staticmethod
    def gphiphi(q):
        """
        Generate the g_phiphi metric component (see Eq. 4.40, 4.41 of arXiv:0908.2085).

        Parameters
        ----------
        q: float, array_like
            The value(s) of Q for the metric.

        Returns
        -------
        gphiphi: float
            The metric value(s).
        """

        if isinstance(q, (list, tuple, np.ndarray)):
            qs = np.asarray(q)
        else:
            qs = q

        return (1.0 / 8.0) * (3 + 8 * qs ** 2)

    def __len__(self):
        return len(self.bank_freqs)

    def generate_waveforms(
        self, iota=np.pi / 2.0, amp=1e-23, flow=20.0, deltat=1.0 / 4096, duration=1.0
    ):
        """
        Generator for waveforms.
        """

        params = dict(iota=iota, freq=0.0, q=0.0, amp=amp)

        for freq, q in zip(self.bank_freqs, self.bank_qs):
            params["q"] = q
            params["freq"] = freq

            yield pycbc.waveform.get_td_waveform(
                approximant="ringdown",
                f_lower=flow,
                delta_t=deltat,
                duration=duration,
                **params
            )


def ringdown_waveform(**kwargs):
    flow = kwargs["f_lower"]  # Required parameter
    dt = kwargs["delta_t"]  # Required parameter
    dur = kwargs.get("duration", 1)
    freq = kwargs["freq"]  # frequency of ring-down (Hz)
    amp = kwargs["amplitude"]  # amplitude of ring-down
    q = kwargs["q"]  # quality factor of ring-down
    cosiota = np.cos(kwargs["iota"])

    t = np.arange(0, dur, dt)

    h0 = amp * np.exp(-np.pi * freq * t / q) * np.cos(2.0 * np.pi * freq * t)

    hp = (1.0 + cosiota ** 2) * h0
    hc = 2 * cosiota * h0 * 1j

    wf = TimeSeries(hp + hc, delta_t=dt, epoch=0)
    return wf.real(), wf.imag()


# add waveform to PyCBC
pycbc.waveform.add_custom_waveform("ringdown", ringdown_waveform, "time", force=True)


class InjectRingdown:
    """
    Class to inject a ring-down signal into some fake data for a given detector.

    Parameters
    ----------
    detector: str
        Valid name of a gravitational-wave detector, e.g., H1, L1, V1
    starttime: float
        GPS start time for the noise data to be generated.
    duration: float
        Length in seconds of the noise data being generated.
    deltat: float
        Time step for the data being generated.
    injfreq: float
        Frequency of the ring-down signal to be injected (Hz).
    injq: float
        Quality factor of the ring-down signal to be injected.
    injlat: float
        Latitude (in equatorial coordinates) of the signal to be injected
        (rads).
    injlong: float
        Longitude (in equatorial coordinates) of the signal to be injected
        (rads).
    injamp: float
        Initial amplitude of the ring-down signal to be injected.
    injt0: float
        The signal's start time as an offset from the start time of the
        data. This must be a positive value. If this is left as the default
        value of None the signal will be placed so that its geocentric
        arrival time would be that at the centre of the data (so the signal
        will have a minor offset due to time difference between the
        detector and geocentre.)
    injiota: float
        Orientation of the source within respect to the line of sight (rads)
        (defaults to pi/2, i.e., linearly polarised).
    injpsi: float
        Gravitational-wave polarisation angle of the source (rads) (defaults
        to 0).
    psd: callable, FrequencySeries, str, float
        A value setting the PSD from which to generate the noise. PyCBC PSD
        functions or FrequencySeries objects can be passed, or strings giving
        files containing PSDs or valid LALSimulation aliases for analytical
        PSDs can be used, or a single float can be used to have a uniform PSD
        as a function of frequency.
    asd: str, float
        If you have a file containing an ASD rather than a PSD, or want to pass
        a single ASD value, use this rather than the psd option.
    flow: float
        The low frequency cut-off for the data/signal (Hz).
    """

    def __init__(
        self,
        detector,
        starttime,
        duration,
        deltat,
        injfreq,
        injq,
        injlat,
        injlong,
        injamp=1e-23,
        injt0=None,
        injiota=np.pi / 2,
        injpsi=0,
        psd=aLIGOZeroDetHighPower,
        asd=None,
        flow=20.0,
    ):
        # create detector into which the signal will be injected
        self.detector = Detector(detector, reference_time=starttime)

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
        self.ts = noise_from_psd(self.nsamples, self.deltat, self.psd)
        self.ts.start_time = starttime

        # inject the signal into the noise
        self.inject_signal(
            injfreq, injq, injlat, injlong, injamp, injt0, injiota, injpsi
        )

    @property
    def psd(self):
        return self.__psd

    @psd.setter
    def psd(self, psd):
        if callable(psd):
            self.__psd = psd(self.nsamples, self.deltaf, self.flow)
        elif isinstance(psd, FrequencySeries):
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
            if self.is_asd_file:
                self.__psd = FrequencySeries(
                    [psd ** 2] * self.nsamples, delta_f=self.deltaf
                )
            else:
                self.__psd = FrequencySeries([psd] * self.nsamples, delta_f=self.deltaf)
        else:
            raise TypeError("Could not create PSD from supplied input")

    def inject_signal(
        self,
        injfreq,
        injq,
        injlat,
        injlong,
        injamp=1e-23,
        injt0=None,
        injiota=np.pi / 2,
        injpsi=0,
    ):
        """
        Add a ring-down signal into noise. This method can be used multiple
        times to add mulitple signals. The pure signal waveform for each signal
        add is also stored in a `signals` class attribute.

        Parameters
        ----------
        injfreq: float
            The ring-down signal frequency (Hz).
        injq: float
            The ring-down signal quality factor.
        injlat: float
            The source equatorial latitude (rads).
        injlong: float
            The source equatorial longitude (rads).
        injamp: float
            The ring-down signal's peak strain amplitude.
        injt0: float
            The signal's start time as an offset from the start time of the
            data. This must be a positive value. If this is left as the default
            value of None the signal will be placed so that its geocentric
            arrival time would be that at the centre of the data (so the signal
            will have a minor offset due to time difference between the
            detector and geocentre.)
        injiota: float
            The inclination of the source with respect to the line-of-sight
            (rads) (defaults to pi/2, i.e., linearly polarised).
        injpsi: float
            The gravitational wave polarisation angle (rads) (defaults to 0). 
        """

        # if injt0 is None it will be set to half the duration, so the signal
        # is placed within the middle of the data (with minor adjustments from
        # signal arrival time compared to the geocentre)
        if injt0 is None:
            injt0 = self.duration / 2.0

        # create dictionary of signal parameters that have been injected
        params = {
            "freq": injfreq,
            "q": injq,
            "latitude": injlat,
            "longitude": injlong,
            "amplitude": injamp,
            "t0": injt0,
            "iota": injiota,
            "psi": injpsi,
        }

        if not hasattr(self, "inj_params"):
            self.inj_params = [params]
            self.signals = []
        else:
            # append parameters for additional injections
            self.inj_params.append(params)

        # generate waveform
        hp, hc = pycbc.waveform.get_td_waveform(
            approximant="ringdown",
            f_lower=self.flow,
            delta_t=self.deltat,
            duration=self.duration,
            **params
        )

        # get signal as seen in the detector
        signal = self.detector.project_wave(
            hp, hc, params["longitude"], params["latitude"], params["psi"]
        )

        # set signal time within data
        signal.start_time += self.ts.start_time + injt0

        # get indices at which to add signal
        idx0 = int(
            (signal.get_sample_times().data[0] - self.ts.get_sample_times().data[0])
            / self.deltat
        )

        # add signal to the noise
        if idx0 >= 0:
            self.ts[idx0:] += signal[: len(self.ts[idx0:])]
        else:
            endidx = len(signal) + idx0
            if endidx > len(self.ts):
                endidx = len(self.ts)
            self.ts[:endidx] += signal[abs(idx0) : abs(idx0) + endidx]

        # append pure-signals to list
        self.signals.append(signal)
