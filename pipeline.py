"""
pipeline.py -- run the whole analysis from the manifest pipeline.yaml: yaml files -> Cobaya runs -> analysis -> comparison.

    python pipeline.py status                      # what exists / is done, per run
    python pipeline.py make                        # write all yaml files (make_runs.py, once per variant)
    python pipeline.py run [--jobs 2]              # cobaya-run every MCMC and minimiser run that is not done yet
    python pipeline.py analyse                     # analyse_chains, plot_chains (+ positivity_check, bound_check where `checks: true`)
    python pipeline.py compare                     # compare_variants, model_comparison, compare_w
    python pipeline.py all                         # make, run, analyse, compare
    python pipeline.py adopt                       # mark chains that already exist as finished (see below)

Options:  --only zm3 base   (variants; `benchmarks` for cpl/lcdm)   --force   --dry-run   --jobs N   --manifest FILE

* Runs that are already done are skipped.  A run counts as done when this script saw cobaya-run finish (chains/pipeline_status.json)
  or, for minimisers, when chains/<prefix>_min.minimum.txt exists.  Chains made by hand before are listed as "found";
  `adopt` marks them as done (do it only if they are finished), `--force` re-runs everything.
* Every command writes a log to logs/.  constants.py is not touched: Z_MATCH goes into the yaml (see make_runs.py).
* The Cobaya runs are sequential by default; `--jobs 4` starts several cobaya-run processes at once.
* model_comparison.py and compare_w.py have their own run lists (RUNS) which you edit by hand.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import variants_config as vc

PY = sys.executable
STATUS = "chains/pipeline_status.json"
RUNS_DIR = "runs"
LOG_DIR = "logs"
_lock = threading.Lock()


# ----------------------------------------------------------------------------------------------------------- status
def load_status():
    """Read the record of finished runs.

    Returns
    -------
    status : dict
        Contents of ``STATUS``, or an empty dict if the file is missing or
        unreadable.
    """
    try:
        with open(STATUS) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def set_done(key):
    """Mark a run as finished in the status file (thread-safe).

    Parameters
    ----------
    key : str
        Chain prefix, with ``_min`` for a minimizer run.
    """
    with _lock:
        st = load_status()
        st[key] = {"done": True, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        os.makedirs("chains", exist_ok=True)
        with open(STATUS, "w") as f:
            json.dump(st, f, indent=1)


def chain_files(prefix):
    """List the MCMC chain files of a run.

    Parameters
    ----------
    prefix : str
        Chain prefix.

    Returns
    -------
    paths : list of str
        Files ``chains/<prefix>.<N>.txt``.
    """
    return glob.glob(f"chains/{prefix}.[0-9]*.txt")


def mcmc_state(prefix, st):
    """Return the state of a run's MCMC.

    Parameters
    ----------
    prefix : str
        Chain prefix.
    st : dict
        Status record (see ``load_status``).

    Returns
    -------
    state : {'done', 'found', 'missing'}
        ``'done'`` if marked finished, ``'found'`` if chain files exist but are
        not marked, otherwise ``'missing'``.
    """
    if st.get(prefix, {}).get("done"):
        return "done"
    return "found" if chain_files(prefix) else "missing"


def min_state(prefix, st):
    """Return the state of a run's minimizer.

    Parameters
    ----------
    prefix : str
        Chain prefix (without ``_min``).
    st : dict
        Status record (see ``load_status``).

    Returns
    -------
    state : {'done', 'missing'}
        ``'done'`` if ``chains/<prefix>_min.minimum.txt`` exists or the run is
        marked finished.
    """
    if os.path.exists(f"chains/{prefix}_min.minimum.txt") or st.get(prefix + "_min", {}).get("done"):
        return "done"
    return "missing"


def yaml_state(prefix):
    """Report whether a run's yaml file exists.

    Parameters
    ----------
    prefix : str
        Chain prefix.

    Returns
    -------
    state : {'yes', 'no'}
        Whether ``runs/<prefix>.yaml`` exists.
    """
    return "yes" if os.path.exists(os.path.join(RUNS_DIR, f"{prefix}.yaml")) else "no"


# ---------------------------------------------------------------------------------------------------------- helpers
def sh(cmd, dry, log=None, env=None, echo=True):
    """Run a command.

    Parameters
    ----------
    cmd : list of str
        Command and arguments.
    dry : bool
        Only print the command.
    log : str, optional
        File for stdout and stderr.  None (default) prints to the console.
    env : dict, optional
        Extra environment variables.
    echo : bool, optional
        Print the command before running it.  Default is True.

    Returns
    -------
    code : int
        Exit code (0 in dry-run mode).
    """
    line = " ".join(cmd)
    if dry:
        print("  [dry-run]", line)
        return 0
    if echo:
        print("  $", line, flush=True)
    full_env = dict(os.environ, **(env or {}))
    if log:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "w") as f:
            f.write(f"$ {line}\n\n")
            f.flush()
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=full_env).returncode
    return subprocess.run(cmd, env=full_env).returncode


# ------------------------------------------------------------------------------------------------------------ stages
def cmd_status(cfg, only):
    """Print a table of which yaml files, chains and minima exist for each run.

    Parameters
    ----------
    cfg : dict
        Manifest (see ``variants_config.load``).
    only : list of str or None
        Variant names to show; None shows all.
    """
    st = load_status()
    rs = vc.runs(cfg, only)
    w = max([len(r["prefix"]) for r in rs] + [10])
    print(f"{'run':{w}s}  yaml  MCMC     minimiser")
    for r in rs:
        print(f"{r['prefix']:{w}s}  {yaml_state(r['prefix']):4s}  {mcmc_state(r['prefix'], st):8s} {min_state(r['prefix'], st)}")
    print("\nMCMC 'found' = chain files exist but this script did not run them: `adopt` marks them finished.")


def cmd_make(cfg, only, dry):
    """Write the yaml files of every variant (and the benchmarks) with make_runs.py.

    Parameters
    ----------
    cfg : dict
        Manifest (see ``variants_config.load``).
    only : list of str or None
        Variant names to make; None makes all.
    dry : bool
        Only print the commands.

    Returns
    -------
    ok : bool
        True if every make_runs.py call succeeded.
    """
    ok = True
    for v in cfg["variants"]:
        if only and v["name"] not in only:
            continue
        print(f"[make] {v['name']}")
        ok &= sh([PY, "make_runs.py", *vc.make_args(cfg, v)], dry, log=None if dry else f"{LOG_DIR}/make_{v['name']}.log", echo=True) == 0
    if cfg.get("benchmarks") and (not only or "benchmarks" in only):
        print("[make] benchmarks")
        ok &= sh([PY, "make_runs.py", *vc.benchmark_args(cfg)], dry, log=None if dry else f"{LOG_DIR}/make_benchmarks.log") == 0
    return ok


def _cobaya(task, dry):
    """Run one cobaya-run job and record it as finished if it succeeds.

    Parameters
    ----------
    task : tuple
        ``(kind, prefix, yaml_file)`` with kind ``'mcmc'`` or ``'min'``.
    dry : bool
        Only print the command.

    Returns
    -------
    code : int
        Exit code of cobaya-run.
    """
    kind, prefix, yaml_file = task
    key = prefix + ("_min" if kind == "min" else "")
    t0 = time.time()
    rc = sh(["cobaya-run", "-f", yaml_file], dry, log=f"{LOG_DIR}/{key}.log")
    if rc == 0 and not dry:
        set_done(key)
    print(f"  [{kind}] {prefix}: {'finished' if rc == 0 else f'FAILED (exit {rc}, see {LOG_DIR}/{key}.log)'}"
          + ("" if dry else f" in {(time.time() - t0) / 60:.1f} min"), flush=True)
    return rc


def cmd_run(cfg, only, dry, force, jobs):
    """Run every MCMC and minimizer that is not done yet.

    Runs whose chains exist but are not marked finished are skipped unless
    `force` is set.

    Parameters
    ----------
    cfg : dict
        Manifest (see ``variants_config.load``).
    only : list of str or None
        Variant names to run; None runs all.
    dry : bool
        Only print the commands.
    force : bool
        Re-run runs that are already done.
    jobs : int
        Number of cobaya-run processes at the same time.

    Returns
    -------
    ok : bool
        True if every run succeeded (or there was nothing to do).
    """
    st = load_status()
    tasks = []
    for r in vc.runs(cfg, only):
        p = r["prefix"]
        for kind, suffix, state in (("mcmc", "", mcmc_state(p, st)), ("min", "_min", min_state(p, st))):
            yf = os.path.join(RUNS_DIR, f"{p}{suffix}.yaml")
            if not os.path.exists(yf):
                print(f"  skip {p}{suffix}: no {yf} (run `pipeline.py make`)")
            elif state == "done" and not force:
                continue
            elif state == "found" and not force:
                print(f"  skip {p}: chain files exist but are not marked finished (`adopt` them, or --force to re-run)")
            else:
                tasks.append((kind, p, yf))
    if not tasks:
        print("[run] nothing to do")
        return True
    print(f"[run] {len(tasks)} cobaya runs, {jobs} at a time (logs in {LOG_DIR}/)")
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        codes = list(ex.map(lambda t: _cobaya(t, dry), tasks))
    return all(c == 0 for c in codes)


def cmd_adopt(cfg, only):
    """Mark chains that already exist as finished.

    Only do this for chains that really are finished.

    Parameters
    ----------
    cfg : dict
        Manifest (see ``variants_config.load``).
    only : list of str or None
        Variant names to adopt; None adopts all.
    """
    st = load_status()
    n = 0
    for r in vc.runs(cfg, only):
        if mcmc_state(r["prefix"], st) == "found":
            set_done(r["prefix"])
            n += 1
            print("  adopted", r["prefix"])
    print(f"[adopt] {n} chains marked as finished")


def cmd_analyse(cfg, only, dry):
    """Run analyse_chains.py and plot_chains.py on every run with chains.

    Runs with ``checks: true`` also get positivity_check.py, and bound_check.py
    for model_3.

    Parameters
    ----------
    cfg : dict
        Manifest (see ``variants_config.load``).
    only : list of str or None
        Variant names to analyse; None analyses all.
    dry : bool
        Only print the commands.

    Returns
    -------
    ok : bool
        Always True.
    """
    st = load_status()
    env = {"MPLBACKEND": "Agg"}
    for r in vc.runs(cfg, only):
        p = r["prefix"]
        if mcmc_state(p, st) == "missing":
            print(f"  skip {p}: no chains")
            continue
        print(f"[analyse] {p}")
        sh([PY, "analyse_chains.py", p], dry, log=f"{LOG_DIR}/analyse_{p}.log", env=env)
        sh([PY, "plot_chains.py", p], dry, log=f"{LOG_DIR}/plot_{p}.log", env=env)
        if r["checks"]:
            sh([PY, "positivity_check.py", p], dry, env=env)
            if r["model"] == "model_3":
                sh([PY, "bound_check.py", p], dry, env=env)
    return True


def cmd_compare(cfg, only, dry):
    """Run compare_variants.py, model_comparison.py and compare_w.py.

    Parameters
    ----------
    cfg : dict
        Manifest (see ``variants_config.load``).
    only : list of str or None
        Unused; the comparisons always cover every run.
    dry : bool
        Only print the commands.

    Returns
    -------
    ok : bool
        Always True.
    """
    env = {"MPLBACKEND": "Agg"}
    for sn in cfg["sn"]:
        for model in cfg["models"]:
            print(f"[compare] variants: {sn} {model}")
            sh([PY, "compare_variants.py", sn, model], dry, env=env)
    print("[compare] model_comparison.py (edit its RUNS list for new tags)")
    sh([PY, "model_comparison.py"], dry, env=env)
    for sn in cfg["sn"]:
        print(f"[compare] compare_w.py {sn} (edit its RUNS list for new tags)")
        sh([PY, "compare_w.py", sn], dry, env=env)
    return True


def main():
    """Parse the command line and run the requested stage (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["status", "make", "run", "analyse", "compare", "all", "adopt"])
    ap.add_argument("--manifest", default=vc.MANIFEST)
    ap.add_argument("--only", nargs="+", default=None, help="variant names from the manifest (or `benchmarks`)")
    ap.add_argument("--force", action="store_true", help="re-run runs that are already done")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    ap.add_argument("--jobs", type=int, default=1, help="number of cobaya-run processes at the same time")
    a = ap.parse_args()
    cfg = vc.load(a.manifest)
    unknown = set(a.only or []) - {v["name"] for v in cfg["variants"]} - {"benchmarks"}
    if unknown:
        sys.exit(f"unknown variant(s) {sorted(unknown)}; the manifest has {[v['name'] for v in cfg['variants']]}")

    c = a.command
    if c == "status":
        cmd_status(cfg, a.only)
    elif c == "adopt":
        cmd_adopt(cfg, a.only)
    elif c == "make":
        cmd_make(cfg, a.only, a.dry_run)
    elif c == "run":
        cmd_run(cfg, a.only, a.dry_run, a.force, a.jobs)
    elif c == "analyse":
        cmd_analyse(cfg, a.only, a.dry_run)
    elif c == "compare":
        cmd_compare(cfg, a.only, a.dry_run)
    elif c == "all":
        if cmd_make(cfg, a.only, a.dry_run):
            cmd_run(cfg, a.only, a.dry_run, a.force, a.jobs)
            cmd_analyse(cfg, a.only, a.dry_run)
            cmd_compare(cfg, a.only, a.dry_run)


if __name__ == "__main__":
    main()