#!/usr/bin/env python3
"""
setup_env.py — Environment installer for RAG-end2end-retriever.

Detects the current platform and installs the correct dependencies:
  • Apple Silicon (M-series) : base deps only, skips nvidia-ml-py3
  • NVIDIA GPU (Linux/Windows): base deps + nvidia-ml-py3
  • CPU-only                  : base deps only

Usage:
    python setup_env.py
    python setup_env.py --dry-run   # print what would be installed, don't install
"""

import argparse
import platform
import subprocess
import sys


# ─── Platform detection ──────────────────────────────────────────────────────

def is_apple_silicon() -> bool:
    """True on macOS running on an M-series chip (arm64)."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def has_nvidia_gpu() -> bool:
    """True if nvidia-smi is reachable and reports at least one GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ─── Installer helpers ────────────────────────────────────────────────────────

def pip_install(*packages: str, dry_run: bool = False) -> None:
    cmd = [sys.executable, "-m", "pip", "install", *packages]
    print(f"  $ {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)


def pip_install_requirements(path: str = "requirements.txt", dry_run: bool = False) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-r", path]
    print(f"  $ {' '.join(cmd)}")
    if not dry_run:
        subprocess.run(cmd, check=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them.")
    args = parser.parse_args()

    dry = args.dry_run
    if dry:
        print("[DRY RUN] — no packages will actually be installed.\n")

    # ── Detect ────────────────────────────────────────────────────────────────
    apple = is_apple_silicon()
    nvidia = has_nvidia_gpu()

    print("── Platform detection ───────────────────────────────")
    print(f"  OS            : {platform.system()} {platform.release()}")
    print(f"  Architecture  : {platform.machine()}")
    print(f"  Apple Silicon : {'YES ✓' if apple else 'no'}")
    print(f"  NVIDIA GPU    : {'YES ✓' if nvidia else 'no'}")
    print()

    # ── Step 1: base dependencies ─────────────────────────────────────────────
    print("── Step 1: installing base dependencies (requirements.txt) ──")
    pip_install_requirements("requirements.txt", dry_run=dry)
    print()

    # ── Step 2: platform-specific extras ─────────────────────────────────────
    print("── Step 2: platform-specific extras ────────────────────────")

    if apple:
        print("  Apple Silicon detected.")
        print("  → Skipping nvidia-ml-py3 (requires NVIDIA drivers, not available on M-series).")
        print("  → MPS (Metal Performance Shaders) will be used for training automatically.")
        print("  → CPU will be used for knowledge-base re-encoding.")

    elif nvidia:
        print("  NVIDIA GPU detected.")
        print("  → Installing nvidia-ml-py3 for free-GPU detection during KB re-encoding.")
        pip_install("nvidia-ml-py3==7.352.0", dry_run=dry)

    else:
        print("  No NVIDIA GPU and not Apple Silicon.")
        print("  → Skipping nvidia-ml-py3 (no NVIDIA drivers present).")
        print("  → Training will fall back to CPU. This is slow — consider using a GPU instance.")

    print()

    # ── Step 3: verify torch backend ─────────────────────────────────────────
    print("── Step 3: verifying torch backend ─────────────────────────")
    if not dry:
        try:
            import torch
            cuda_ok = torch.cuda.is_available()
            mps_ok  = torch.backends.mps.is_available()

            if cuda_ok:
                device_name = torch.cuda.get_device_name(0)
                print(f"  ✓ CUDA available  — device: {device_name}")
            elif mps_ok:
                print("  ✓ MPS  available  — Apple Silicon GPU will be used for training")
            else:
                print("  ⚠ No GPU backend available — training will use CPU only")

        except ImportError:
            print("  ⚠ torch not found after install — try running this script again")
    else:
        print("  (skipped in dry-run mode)")

    print()
    print("── Setup complete ───────────────────────────────────────────")
    if apple:
        print("  Run training with:  bash finetune_rag_mps_end2end.sh")
    elif nvidia:
        print("  Run training with:  bash finetune_rag_ray_end2end.sh")
    else:
        print("  Run training with:  python finetune_rag.py  (CPU mode, adjust batch size)")


if __name__ == "__main__":
    main()
