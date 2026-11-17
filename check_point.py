"""Evaluate every likelihood at one point, outside the sampler.  Run: python check_point.py"""
from cobaya.yaml import yaml_load_file
from cobaya.model import get_model

info = yaml_load_file("likelihood.yaml")
info.pop("output", None)
info.pop("sampler", None)
model = get_model(info)

# LCDM-limit point (q0 = 1.5 Om0 - 1 for j0 = 1):
point = dict(H0=67.4, Om0=0.315, q0=1.5 * 0.315 - 1.0, j0=1.0, ombh2=0.02237)
res = model.logposterior(point, as_dict=True)
print("loglikes:", res["loglikes"])         # one entry per likelihood; -inf marks the culprit
print("derived :", res["derived"])