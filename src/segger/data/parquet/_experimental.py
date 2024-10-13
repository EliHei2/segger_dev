from typing import TYPE_CHECKING

if TYPE_CHECKING:  # False at runtime
    import dask, cudf, dask_cudf, pandas as pd


class BackendHandler:
    """
    A class to handle different DataFrame backends for reading and processing
    Parquet files.

    Attributes
    ----------
    _valid_backends : set
        A set of valid backend options ('pandas', 'dask', 'cudf', 'dask_cudf').
    backend : str
        The selected backend for reading Parquet files.

    Methods
    -------
    read_parquet():
        Returns the function to read Parquet files according to the selected
        backend.
    """

    _valid_backends = {
        "pandas",
        "dask",
        "cudf",
        "dask_cudf",
    }

    def __init__(self, backend):
        # Make sure requested backend is supported
        if backend in self._valid_backends:
            self.backend = backend
        else:
            valid = ", ".join(map(lambda o: f"'{o}'", self._valid_backends))
            msg = f"Unsupported backend: {backend}. Valid options are {valid}."
            raise ValueError(msg)

        # Dynamically import packages only if requested
        if self.backend == "pandas":
            import pandas as pd
        elif self.backend == "dask":
            import dask
        elif self.backend == "cudf":
            import cudf
        elif self.backend == "dask_cudf":
            import dask_cudf
        else:
            raise ValueError("Internal Error")

    @property
    def read_parquet(self):
        if self.backend == "pandas":
            return pd.read_parquet
        elif self.backend == "dask":
            return dask.dataframe.read_parquet
        elif self.backend == "cudf":
            return cudf.read_parquet
        elif self.backend == "dask_cudf":
            return dask_cudf.read_parquet
        else:
            raise ValueError("Internal Error")
