# Upstream patches for HY-World 2.0 baseline

These patches apply to the **upstream checkout** (e.g. `_bundle_deps/upstream/HY-World-2.0-gh`), not to files inside this repo root.

## `world_gs_trainer_enable_sh_schedule.patch`

Re-enable SH degree ramp during 3DGS training when `sh_degree > 0`:

```text
sh_degree_to_use = min(step // sh_degree_interval, sh_degree)
```

Apply after source setup:

```bash
cd "$HY2_WORLDGEN_ROOT/../.."   # hyworld2 parent = HY-World-2.0-gh root
# or: cd /path/to/HY-World-2.0-gh
patch -p1 < /path/to/HY/patches/world_gs_trainer_enable_sh_schedule.patch
```

Works together with Stage06 defaults in `scripts/env.sh` (`HY2_SH_DEGREE=1`, `HY2_SH_DEGREE_INTERVAL=1500`).
