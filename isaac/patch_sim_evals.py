"""Make sim-evals installable from a Gilbreth login node. Idempotent; keeps pyproject.toml.orig.

    python isaac/patch_sim_evals.py /scratch/gilbreth/$USER/sim-evals

Two fixes to sim-evals' pyproject.toml:

1. Isaac Sim packages come straight from NVIDIA's index (pypi.nvidia.com) as real wheels.
   On PyPI they are "wheel-stub" sdists that download the real wheel at build time, choosing
   one compatible with the *build machine*. We build on the host (older glibc) for the
   container (glibc 2.35), so the stubs would find no compatible wheel. Real wheels from
   NVIDIA's index respect `uv sync --python-platform x86_64-manylinux_2_35`.
2. flatdict 4.0.1 (via Isaac Lab) only ships as an sdist whose setup.py imports pkg_resources,
   which recent setuptools no longer includes. Pin an older setuptools for that build only.
"""
import pathlib
import re
import shutil
import sys

# Isaac Sim 5.0 packages sim-evals pulls in (from its uv.lock); more are added from the lock if present.
ISAAC_PACKAGES = [
    "isaacsim", "isaacsim-app", "isaacsim-asset", "isaacsim-benchmark", "isaacsim-code-editor", "isaacsim-core",
    "isaacsim-cortex", "isaacsim-example", "isaacsim-extscache-kit", "isaacsim-extscache-kit-sdk",
    "isaacsim-extscache-physics", "isaacsim-gui", "isaacsim-kernel", "isaacsim-replicator", "isaacsim-rl",
    "isaacsim-robot", "isaacsim-robot-motion", "isaacsim-robot-setup", "isaacsim-ros1", "isaacsim-ros2",
    "isaacsim-sensor", "isaacsim-storage", "isaacsim-template", "isaacsim-test", "isaacsim-utils", "omniverse-kit",
]
BUILD_FIXES = {"flatdict": ["setuptools<75"]}


def patch(root):
    root = pathlib.Path(root)
    pp = root / "pyproject.toml"
    text = pp.read_text()
    orig = text
    names = list(ISAAC_PACKAGES)
    lock = root / "uv.lock"
    if lock.exists():
        for n in re.findall(r'^name = "((?:isaacsim|omniverse-kit)[^"]*)"', lock.read_text(), re.M):
            if n not in names:
                names.append(n)

    # 1. sources: add `<name> = { index = "isaac" }` under [tool.uv.sources] for names not listed yet
    m = re.search(r"^\[tool\.uv\.sources\]\s*$", text, re.M)
    if not m:
        text = text.rstrip() + "\n\n[tool.uv.sources]\n"
        m = re.search(r"^\[tool\.uv\.sources\]\s*$", text, re.M)
    have = set(re.findall(r"^([A-Za-z0-9_.-]+)\s*=", text[m.end():].split("\n[", 1)[0], re.M))
    add = [f'{n} = {{ index = "isaac" }}' for n in names if n not in have]
    if add:
        text = text[: m.end()] + "\n" + "\n".join(add) + text[m.end():]

    # uv only honours [tool.uv.sources] for *direct* dependencies, and these arrive indirectly via
    # isaaclab. Listing them in a dependency group makes them direct, so the lock takes them from
    # NVIDIA's index for the whole project (the lock has one source per package).
    if "[dependency-groups]" not in text:
        body = ",\n".join(f'  "{n}"' for n in names)
        text = text.rstrip() + f"\n\n[dependency-groups]\nisaac-wheels = [\n{body},\n]\n"

    # 2. extra build dependencies for broken sdists
    if "[tool.uv.extra-build-dependencies]" not in text:
        body = "\n".join(f"{k} = {v!r}".replace("'", '"') for k, v in BUILD_FIXES.items())
        text = text.rstrip() + "\n\n[tool.uv.extra-build-dependencies]\n" + body + "\n"

    if text == orig:
        print("sim-evals pyproject.toml already patched")
        return
    if not (root / "pyproject.toml.orig").exists():
        shutil.copy(pp, root / "pyproject.toml.orig")
    with open(pp, "w", newline="\n") as f:
        f.write(text)
    print(f"patched {pp}: {len(add)} Isaac Sim packages -> NVIDIA index; build fixes for {sorted(BUILD_FIXES)}")


if __name__ == "__main__":
    patch(sys.argv[1] if len(sys.argv) > 1 else ".")
