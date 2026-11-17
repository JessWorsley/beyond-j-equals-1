"""
variants_config.py -- reads pipeline.yaml (the manifest of runs).  Used by pipeline.py and compare_variants.py.
"""
import os
import yaml

from chain_utils import run_prefix
from make_runs import effective_tag

MANIFEST = "pipeline.yaml"


def load(path=MANIFEST):
    """Read the run manifest and fill in defaults.

    Parameters
    ----------
    path : str, optional
        Manifest file.  Default is ``MANIFEST`` (``pipeline.yaml``).

    Returns
    -------
    cfg : dict
        Manifest with defaults for ``sn``, ``models``, ``benchmarks``,
        ``best_of`` and ``variants``.

    Raises
    ------
    ValueError
        If two variants have the same name.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("sn", ["pantheon"])
    cfg.setdefault("models", ["model_3"])
    cfg.setdefault("benchmarks", [])
    cfg.setdefault("best_of", None)
    cfg.setdefault("variants", [])
    names = [v["name"] for v in cfg["variants"]]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: variant names must be unique, got {names}")
    return cfg


def variant_tag(v):
    """Return the tag that ends up in a variant's chain prefix.

    Uses the same logic as make_runs.py (``effective_tag``).

    Parameters
    ----------
    v : dict
        Variant entry of the manifest.

    Returns
    -------
    tag : str
        The tag.
    """
    args = list(v.get("args") or [])
    rd = float(args[args.index("--rd-fixed") + 1]) if "--rd-fixed" in args else None
    return effective_tag(v.get("tag", "") or "", rd, "--no-planck" in args)


def z_match_args(v):
    """Return the make_runs.py arguments for a variant's late-time closure.

    Parameters
    ----------
    v : dict
        Variant entry of the manifest.

    Returns
    -------
    args : list of str
        ``['--z-match', 'none']``, or ``['--z-match', <z>]`` plus
        ``'--z-match-negative'`` if the variant allows a negative density.
    """
    z = v.get("z_match")
    if z is None:
        return ["--z-match", "none"]
    return ["--z-match", str(z)] + (["--z-match-negative"] if v.get("z_match_negative") else [])


def make_args(cfg, v):
    """Return the make_runs.py arguments that create all yaml files of a variant.

    Parameters
    ----------
    cfg : dict
        Manifest (see ``load``).
    v : dict
        Variant entry of the manifest.

    Returns
    -------
    args : list of str
        Command-line arguments, always including ``--minimize``.
    """
    a = ["--sn", *cfg["sn"], "--models", *cfg["models"]]
    if v.get("tag"):
        a += ["--tag", v["tag"]]
    a += list(v.get("args") or []) + z_match_args(v) + ["--minimize"]
    if cfg.get("best_of"):
        a += ["--best-of", str(cfg["best_of"])]
    return a


def benchmark_args(cfg):
    """Return the make_runs.py arguments for the benchmark runs (CPL, LCDM limit).

    Parameters
    ----------
    cfg : dict
        Manifest (see ``load``).

    Returns
    -------
    args : list of str
        Command-line arguments, with the full closure (``--z-match none``) and
        ``--minimize``.
    """
    a = ["--sn", *cfg["sn"], "--models", *cfg["benchmarks"], "--z-match", "none", "--minimize"]
    if cfg.get("best_of"):
        a += ["--best-of", str(cfg["best_of"])]
    return a


def runs(cfg, only=None):
    """List every run in the manifest.

    Parameters
    ----------
    cfg : dict
        Manifest (see ``load``).
    only : list of str or None, optional
        Variant names to keep (``'benchmarks'`` for the benchmark runs).  None
        keeps all.

    Returns
    -------
    runs : list of dict
        One dict per run, with keys ``variant``, ``label``, ``sn``, ``model``,
        ``prefix``, ``z_match``, ``checks`` and ``compare``.
    """
    out = []
    for v in cfg["variants"]:
        if only and v["name"] not in only:
            continue
        tag = variant_tag(v)
        for sn in cfg["sn"]:
            for model in cfg["models"]:
                out.append(dict(variant=v["name"], label=v.get("label", v["name"]), sn=sn, model=model,
                                prefix=run_prefix(sn, model, tag), z_match=v.get("z_match"),
                                checks=bool(v.get("checks")), compare=v.get("compare", True)))
    if cfg.get("benchmarks") and (not only or "benchmarks" in only):
        for sn in cfg["sn"]:
            for model in cfg["benchmarks"]:
                out.append(dict(variant="benchmarks", label=model, sn=sn, model=model, prefix=run_prefix(sn, model),
                                z_match=None, checks=False, compare=False))
    return out


def compare_list(sn, model, path=MANIFEST):
    """Return the variants compare_variants.py should show for one dataset and model.

    Parameters
    ----------
    sn : str
        SN dataset.
    model : str
        Closure model.
    path : str, optional
        Manifest file.  Default is ``MANIFEST``.

    Returns
    -------
    variants : list of tuple or None
        ``(label, prefix, z_match)`` per variant, or None if there is no
        manifest or no matching run.
    """
    if not os.path.exists(path):
        return None
    cfg = load(path)
    lst = [(r["label"], r["prefix"], r["z_match"]) for r in runs(cfg)
           if r["sn"] == sn and r["model"] == model and r["compare"]]
    return lst or None