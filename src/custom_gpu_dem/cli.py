"""
Basic CLI for custom-gpu-dem.

Usage:
  custom-gpu-dem run examples/simple.yaml --steps 500 --vtk-every 100
"""
import argparse
from pathlib import Path

from .config import load_config
from .simulation import DEMSimulation

def main():
    parser = argparse.ArgumentParser(description="custom-gpu-dem runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run a simulation from config")
    run_p.add_argument("config", help="YAML or will use defaults")
    run_p.add_argument("--steps", type=int, default=400)
    run_p.add_argument("--vtk-every", type=int, default=0)
    run_p.add_argument("--out-dir", default="output")
    run_p.add_argument("--log-every", type=int, default=50)

    args = parser.parse_args()

    if args.cmd == "run":
        cfg = DEMConfig()  # TODO: proper load  # fallback not great
        # simple: if file not exist use default config
        try:
            cfg = load_config(args.config)
        except Exception:
            from .config import DEMConfig
            cfg = DEMConfig()
            print("Using default DEMConfig (no valid config file)")

        sim = DEMSimulation(cfg)
        sim.initialize_particles()
        sim.run(args.steps, log_every=args.log_every, vtk_every=args.vtk_every or None, out_dir=args.out_dir)
        sim.save_checkpoint(f"{args.out_dir}/final.npz")
        print("Done. Final checkpoint written.")

if __name__ == "__main__":
    main()
