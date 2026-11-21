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
import hashlib
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
    try:
        with open(STATUS) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def yaml_hash(path):
    """Fingerprint of a yaml file (priors, likelihood options ...): a run is only valid for the yaml it was made with."""
    try:
        with open(path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()[:12]
    except OSError:
        return None


def set_done(key, yaml_file=None):
    with _lock:
        st = load_status()
        st[key] = {"done": True, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "yaml_hash": yaml_hash(yaml_file) if yaml_file else None}
        os.makedirs("chains", exist_ok=True)
        with open(STATUS, "w") as f:
            json.dump(st, f, indent=1)


def chain_files(prefix):
    return glob.glob(f"chains/{prefix}.[0-9]*.txt")


def _state(key, st, yaml_file, files_exist):
    """done | stale | found | missing.   stale = recorded as finished, but with a different yaml (or before yaml fingerprints existed):
    e.g. after the priors were changed.  found = output exists but this script never recorded the run (see `adopt`)."""
    rec = st.get(key)
    if rec and rec.get("done"):
        h = yaml_hash(yaml_file)
        return "done" if h is not None and rec.get("yaml_hash") == h else "stale"
    return "found" if files_exist else "missing"


def mcmc_state(prefix, st):
    return _state(prefix, st, os.path.join(RUNS_DIR, f"{prefix}.yaml"), bool(chain_files(prefix)))


def min_state(prefix, st):
    return _state(prefix + "_min", st, os.path.join(RUNS_DIR, f"{prefix}_min.yaml"), os.path.exists(f"chains/{prefix}_min.minimum.txt"))


def yaml_state(prefix):
    return "yes" if os.path.exists(os.path.join(RUNS_DIR, f"{prefix}.yaml")) else "no"


# ---------------------------------------------------------------------------------------------------------- helpers
def sh(cmd, dry, log=None, env=None, echo=True):
    """Run a command; returns the exit code (0 in dry-run mode).  Output goes to `log` if given, else to the console."""
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
    st = load_status()
    rs = vc.runs(cfg, only)
    w = max([len(r["prefix"]) for r in rs] + [10])
    print(f"{'run':{w}s}  yaml  MCMC     minimiser")
    for r in rs:
        print(f"{r['prefix']:{w}s}  {yaml_state(r['prefix']):4s}  {mcmc_state(r['prefix'], st):8s} {min_state(r['prefix'], st)}")
    print("\nfound = output exists but this script did not run it: `adopt` marks it finished (only if the yaml is the one it was made with).")
    print("stale = finished with an older yaml (e.g. before a prior change): `run` repeats it.")


def cmd_make(cfg, only, dry):
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
    kind, prefix, yaml_file = task
    key = prefix + ("_min" if kind == "min" else "")
    t0 = time.time()
    rc = sh(["cobaya-run", "-f", yaml_file], dry, log=f"{LOG_DIR}/{key}.log")
    if rc == 0 and not dry:
        set_done(key, yaml_file)
    print(f"  [{kind}] {prefix}: {'finished' if rc == 0 else f'FAILED (exit {rc}, see {LOG_DIR}/{key}.log)'}"
          + ("" if dry else f" in {(time.time() - t0) / 60:.1f} min"), flush=True)
    return rc


def cmd_run(cfg, only, dry, force, jobs):
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
                print(f"  skip {p}{suffix}: output exists but is not marked finished (`adopt` it, or --force to re-run)")
            else:
                if state == "stale":
                    print(f"  {p}{suffix}: stale (made with an older yaml), will be re-run")
                tasks.append((kind, p, yf))
    if not tasks:
        print("[run] nothing to do")
        return True
    print(f"[run] {len(tasks)} cobaya runs, {jobs} at a time (logs in {LOG_DIR}/)")
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        codes = list(ex.map(lambda t: _cobaya(t, dry), tasks))
    return all(c == 0 for c in codes)


def cmd_adopt(cfg, only):
    """Mark existing output as finished, FOR THE YAML AS IT IS NOW.  Only use this if the yaml files were not changed after the runs."""
    st = load_status()
    n = 0
    for r in vc.runs(cfg, only):
        p = r["prefix"]
        for suffix, state in (("", mcmc_state(p, st)), ("_min", min_state(p, st))):
            if state in ("found", "stale"):
                set_done(p + suffix, os.path.join(RUNS_DIR, f"{p}{suffix}.yaml"))
                n += 1
                print(f"  adopted {p}{suffix}" + (" (was stale)" if state == "stale" else ""))
    print(f"[adopt] {n} runs marked as finished")


def cmd_analyse(cfg, only, dry):
    st = load_status()
    env = {"MPLBACKEND": "Agg"}
    for r in vc.runs(cfg, only):
        p = r["prefix"]
        state = mcmc_state(p, st)
        if state == "missing":
            print(f"  skip {p}: no chains")
            continue
        if state == "stale":
            print(f"  WARNING {p}: chains were made with an older yaml (stale); re-run before using the results")
        print(f"[analyse] {p}")
        sh([PY, "analyse_chains.py", p], dry, log=f"{LOG_DIR}/analyse_{p}.log", env=env)
        sh([PY, "plot_chains.py", p], dry, log=f"{LOG_DIR}/plot_{p}.log", env=env)
        if r["checks"]:
            sh([PY, "positivity_check.py", p], dry, env=env)
            if r["model"] == "model_3":
                sh([PY, "bound_check.py", p], dry, env=env)
    return True


def cmd_compare(cfg, only, dry):
    env = {"MPLBACKEND": "Agg"}
    for sn in cfg["sn"]:
        for model in cfg["models"]:
            print(f"[compare] variants: {sn} {model}")
            sh([PY, "compare_variants.py", sn, model], dry, env=env)
    for tag in cfg.get("model_comparison_tags", [""]):
        print(f"[compare] model_comparison.py {('--tag ' + tag) if tag else '(untagged = base chains)'}")
        sh([PY, "model_comparison.py"] + (["--tag", tag] if tag else []), dry, env=env)
    for sn in cfg["sn"]:
        print(f"[compare] compare_w.py {sn} (edit its RUNS list for new tags)")
        sh([PY, "compare_w.py", sn], dry, env=env)
    print("[compare] compare_plots.py (variants = plot_variants of the manifest)")
    sh([PY, "compare_plots.py"], dry, env=env)
    return True


def main():
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