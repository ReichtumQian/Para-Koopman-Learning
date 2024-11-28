import joblib

NJOBS = -1  # -1 means all CPUs


def get_n_jobs():
  global NJOBS
  return NJOBS


def set_n_jobs(n_jobs):
  assert n_jobs == -1 or n_jobs > 0, "n_jobs must be -1 or > 0"
  global NJOBS
  NJOBS = n_jobs
