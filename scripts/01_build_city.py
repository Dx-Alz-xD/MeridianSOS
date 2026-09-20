"""Build Meridian City and cache it to data/city/meridian.npz.

Usage:  python scripts/01_build_city.py [--seed N]
"""
import argparse
from meridian.config import CFG
from meridian.worldgen.city import build_city, save_city

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=None)
args = ap.parse_args()

print("Building Meridian City...")
city = build_city(CFG, seed=args.seed)
path = save_city(city)
print(f"saved -> {path}")
