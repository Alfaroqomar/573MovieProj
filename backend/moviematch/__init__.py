"""MovieMatch backend package."""

import os

# implicit's ALS is multi-threaded itself; a multi-threaded BLAS underneath it
# oversubscribes the CPU and slows training dramatically.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
