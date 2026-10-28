"""
push_to_github.py
=================
Pushes the committed repository to your GitHub remote.

Usage:
  .venv\\Scripts\\python.exe push_to_github.py <github_repo_url>

Example:
  .venv\\Scripts\\python.exe push_to_github.py https://github.com/myusername/flight-trajectory-predictor.git
"""

import sys
from pathlib import Path
from dulwich import porcelain

if len(sys.argv) < 2:
    print("Usage: python push_to_github.py <github_repo_url>")
    print("Example: python push_to_github.py https://github.com/myusername/flight-trajectory-predictor.git")
    sys.exit(1)

remote_url = sys.argv[1].strip()
repo_dir = str(Path(__file__).resolve().parent)

print(f"Setting remote 'origin' -> {remote_url}")
try:
    porcelain.remote_add(repo_dir, "origin", remote_url)
except Exception:
    pass

print(f"Pushing branch 'main' to {remote_url}...")
try:
    porcelain.push(repo_dir, remote_url, refspecs=[b"refs/heads/main"])
    print("Successfully pushed to GitHub!")
except Exception as e:
    print(f"Push failed: {e}")
    print("\nIf authentication is required, you can also push via standard git:")
    print(f"  git remote add origin {remote_url}")
    print("  git push -u origin main")
