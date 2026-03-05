"""Data scientist agent package facade."""

from . import dataset_io as ds_datasets
from . import planner as ds_planner
from . import service as _service

for _name in dir(_service):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_service, _name)

del _name
