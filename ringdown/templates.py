import numpy as np
from pycbc.waveform import ringdown_td_approximants, ringdown_fd_approximants


class RingdownTemplateBank:
    """
    A class to generate a template bank for ring-down signals. This uses the
    metric as defined in Creighton, PRD, 60, 022001 (1999) arXiv:gr-qc/9901084.

    Parameters
    ----------
    frange: array_like
        Array containing the minimum and maximum frequency to use (Hz)
    qrange: array_like
        Array containing a minimum and maximum Q values
    taurange: array_like
        Array containing a minimum and maximum decay time value (can use
        instead of Q) (s)
    mm: float
        Metric mismatch to use.
    """

    def __init__(self, frange, qrange=None, taurange=None, mm=0.03):
        self.frange = frange

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
    def frange(self):
        return self.__frange

    @frange.setter
    def frange(self, frange):
        try:
            fmin = np.min(frange)
            fmax = np.max(frange)
        except Exception as e:
            raise TypeError(
                "Could not get minimum and maximum values from frange: {}".format(e)
            )
        
        if fmin >= fmax:
            raise ValueError("Min frequency is greater than max frequency")

        self.__frange = [fmin, fmax]
        self.phirange = np.log(self.frange)

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
                    "i,j->ij", [taumin, taumax], self.frange
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
            phis = np.arange(self.phirange[0], self.phirange[1], dphi)
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
        self, iota=np.pi / 2.0, amp=1e-23, psi=0.0, phi=0.0, flow=20.0, deltat=None, duration=1.0, deltaf=None,
        f_final=None, domain="time",
    ):
        """
        Generator for waveforms.
        """

        params = dict(
            lmns="221",  # ring-down mode
            inclination=iota,
            amp220=amp,
            polarizarion=psi,
            tc=duration / 2,
            phi220=phi,
        )

        for freq, q in zip(self.bank_freqs, self.bank_qs):
            params["tau_220"] = q / (np.pi * freq)
            params["f_220"] = freq

            if domain == "frequency":
                # get frequency domain waveform
                yield ringdown_fd_approximants["FdQNMfromFreqTau"](
                    f_lower=flow,
                    delta_f=deltaf,
                    **params
                )
            else:
                # get time domain waveform
                yield ringdown_td_approximants["TdQNMfromFreqTau"](
                    f_lower=flow,
                    duration=duration,
                    delta_t=deltat,
                    **params
                )
