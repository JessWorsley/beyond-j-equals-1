"""Checks the generated yaml files and the run-naming convention.   python test_make_runs.py"""
import numpy as np
import make_runs as MR
from chain_utils import run_prefix, infer_model

ok = True
def check(name, cond, info=""):
    """Print PASS or FAIL for one test and record a failure in the global ``ok``.

    Parameters
    ----------
    name : str
        Test description.
    cond : bool
        Whether the test passed.
    info : str, optional
        Extra text printed after the name.
    """
    global ok; ok &= bool(cond); print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")

for sn in ("pantheon", "desy5"):
    for model in ("model_1", "model_2", "model_3", "cpl", "lcdm"):
        for mini in (False, True):
            prefix, cfg = MR.build(sn, model, "_t", {"positive_dark": True}, mini)
            names = list(cfg["likelihood"])
            sfx = "CPL" if model == "cpl" else ""
            sn_cls = ("Pantheon" if sn == "pantheon" else "DESY5") + "Likelihood" + sfx
            good = (f"likelihoods.{sn_cls}" in names and f"likelihoods.DESILikelihood{sfx}" in names
                    and f"likelihoods.PlanckLikelihood{sfx}" in names)
            expect_model = {"lcdm": "model_1"}.get(model, model)
            if model != "cpl":
                good &= all(v["model"] == expect_model for v in cfg["likelihood"].values())
            good &= cfg["likelihood"][f"likelihoods.PlanckLikelihood{sfx}"]["positive_dark"] is True
            good &= cfg["output"] == f"chains/{prefix}" + ("_min" if mini else "")
            good &= ("minimize" in cfg["sampler"]) == mini
            good &= infer_model(prefix, []) == {"lcdm": "model_1"}.get(model, model)
            pars = cfg["params"]
            if model == "cpl":
                good &= {"w0", "wa"} <= set(pars) and "q0" not in pars and "prior" in cfg
            elif model == "lcdm":
                good &= pars["j0"] == 1.0 and "value" in pars["q0"]
            else:
                good &= {"q0", "j0"} <= set(pars) and "dq" not in pars
            check(f"{sn:8s} {model:8s} {'min ' if mini else 'mcmc'} yaml", good)

check("run_prefix convention", run_prefix("pantheon", "model_3") == "desi_pantheon_planck_model_3"
      and run_prefix("desy5", "cpl") == "cpl_desy5_run" and run_prefix("pantheon", "cpl") == "cpl_run"
      and run_prefix("desy5", "lcdm") == "lcdm_limit_desy5" and run_prefix("desy5", "model_1", "_zm3") == "desi_desy5_planck_model_1_zm3")
check("infer_model: legacy CPL chain falls back to the given default", infer_model("desi_pantheon_planck_cpl", ["q0", "j0"], fallback="cpl") == "cpl")
print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")