"""
Master Monthly Rebalancing Pipeline
Chains the three core stages:
1. update_returns.py  (Data Collection & Incremental Update)
2. run_ff3.py         (FF3 Factor Regression & 9 Stock Selection)
3. run_bl_portfolio.py (XGBoost View Generation & Black-Litterman Allocation)
"""

import sys
import time
import argparse
import subprocess
import os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)


def run_cmd(script_name, args_list=None):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    cmd = [sys.executable, script_path] + (args_list or [])
    print(f"\n{'='*75}\n[Pipeline Step] Running: {' '.join(cmd)}\n{'='*75}")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=ROOT_DIR)
    if res.returncode != 0:
        print(f"[Error] Step {script_name} failed with code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[Step Finished] {script_name} in {time.time()-t0:.2f}s\n")


def main():
    parser = argparse.ArgumentParser(description="Run complete monthly quantitative investment pipeline.")
    parser.add_argument('--force-download', action='store_true', help="Force 20-year re-download of raw stock prices")
    parser.add_argument('--t-cut', type=float, default=2.0, help="FF3 t(alpha) gate threshold (default: 2.0)")
    parser.add_argument('--tau', type=float, default=0.1, help="BL tau parameter (default: 0.1)")
    parser.add_argument('--delta', type=float, default=3.0, help="BL risk aversion parameter (default: 3.0)")
    parser.add_argument('--max-weight', type=float, default=0.20, help="Max single stock weight (default: 0.20)")

    args = parser.parse_args()

    t_all = time.time()
    print("*" * 75)
    print(" STARTING MASTER QUANTITATIVE ASSET ALLOCATION PIPELINE")
    print("*" * 75)

    # Step 1: Update returns
    update_args = ['--force'] if args.force_download else []
    run_cmd('update_returns.py', update_args)

    # Step 2: FF3 selection
    ff3_args = ['--t-cut', str(args.t_cut), '--update-factors']
    run_cmd('run_ff3.py', ff3_args)

    # Step 3: Black-Litterman optimization
    bl_args = [
        '--tau', str(args.tau),
        '--delta', str(args.delta),
        '--max-weight', str(args.max_weight)
    ]
    run_cmd('run_bl_portfolio.py', bl_args)

    print("*" * 75)
    print(f"[PIPELINE SUCCESS] Master monthly rebalancing completed in {time.time()-t_all:.2f}s!")
    print("Results are available in results/ and data/ folders.")
    print("*" * 75)


if __name__ == '__main__':
    main()
