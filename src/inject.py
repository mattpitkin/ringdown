from pycbc.io import FieldArray
import numpy as np
import tempfile
import shutil
from pycbc.inject import RingdownHDFInjectionSet


class RingdownInjections:
    """
    Create a set of damped sinusoid injections that can be added to data.

    These ring-down signals will just assume one mode, the fundamental
    l=2, m=2, n=0 mode.
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
        inclinations=np.pi / 2.,
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
                ("f_lmn", "<f8"),
                ("lmn", "S3"),
                ("tau_lmn", "<f8"),
                ("amp220", "<f8"),
                ("polarization", "<f8"),
                ("inclination", "<f8"),
                ("ra", "<f8"),
                ("dec", "<f8"),
                ("tc", "<f8"),
            ]
        )

        pnames = ["f_lmn", "tau", "amp220", "ra", "dec", "inclinations", "polarization", "tc"]

        for param, pvalue in zip(
            pnames,
            [freqs, taus, amps, ras, decs, inclinations, psis, tcs],
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
        self.__injections["lmn"] = np.full(self.ninj, "221")  # use 1 22 mode (the 220 mode)

        self.load()

    def load(self, filename=None):
        """
        Load the injection via writing and reading to a HDF5 file. If a
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
            tmpfile = tempfile.mkstemp(suffix="hdf5")[1]
        else:
            tmpfile = filename

        # output the injection
        RingdownHDFInjectionSet.write(tmpfile, self.__injections)

        # re-read in the injections
        self.injections = RingdownHDFInjectionSet(tmpfile)

        if filename is None:
            shutil.rmtree(filename)

    def inject(self, data, detector):
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

        self.injection_set.apply(data, detector_name=detector)