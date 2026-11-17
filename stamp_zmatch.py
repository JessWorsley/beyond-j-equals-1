"""
stamp_zmatch.py -- record the late-time-closure setting in an existing yaml (does NOT change any chain).

    python stamp_zmatch.py desi_pantheon_planck_model_3_zm3pos 3.0 false     # Z_MATCH = 3, negative frozen density not allowed
    python stamp_zmatch.py desi_pantheon_planck_model_3_zm3    3.0 true
    python stamp_zmatch.py desi_pantheon_planck_model_3        none           # a normal run

For yaml files written before make_runs.py started storing `z_match` (2026-11-16): the chains were run with whatever
constants.py said at the time, and the analysis scripts read the setting back from the yaml.  Without the options they fall
back to constants.py, so a z-match chain is plotted with the full closure.  Stamp the value the chain was RUN with.
Also stamps <prefix>_min.yaml when it exists.
"""
import sys
import yaml


def stamp(path, z_match, allow_negative):
    """Write the z_match options into every likelihood block of a yaml file.

    Parameters
    ----------
    path : str
        yaml file, rewritten in place.
    z_match : float or None
        Z_MATCH the chain was run with.
    allow_negative : bool
        Whether a negative frozen dark density was allowed.
    """
    with open(path) as f:
        info = yaml.safe_load(f)
    n = 0
    for block in info.get("likelihood", {}).values():
        if isinstance(block, dict):
            block["z_match"] = z_match
            block["z_match_allow_negative"] = allow_negative
            n += 1
    with open(path, "w") as f:
        yaml.safe_dump(info, f, sort_keys=False, default_flow_style=False)
    print(f"{path}: z_match = {z_match}, allow_negative = {allow_negative} written to {n} likelihood blocks")


if __name__ == "__main__":
    prefix = sys.argv[1]
    zm = None if sys.argv[2].lower() in ("none", "null") else float(sys.argv[2])
    neg = len(sys.argv) > 3 and sys.argv[3].lower() in ("true", "1", "yes")
    import os
    done = False
    for p in (f"runs/{prefix}.yaml", f"runs/{prefix}_min.yaml"):
        if os.path.exists(p):
            stamp(p, zm, neg)
            done = True
    if not done:
        sys.exit(f"no runs/{prefix}.yaml found")