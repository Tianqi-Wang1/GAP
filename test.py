PYTHONNOUSERSITE=1 /home/jincai_guo/.conda/envs/tianqi_GR4CIL/bin/python - <<'PY'
import sys, os, importlib.util, importlib.metadata as md

print("python executable:")
print(sys.executable)

print("\nPYTHONPATH:")
print(os.environ.get("PYTHONPATH"))

print("\nsys.path:")
for p in sys.path:
    print(p)

import torch
print("\ntorch version:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("torch path:", torch.__file__)

spec = importlib.util.find_spec("torchvision")
print("\ntorchvision version:", md.version("torchvision"))
print("torchvision path:", spec.origin)
PY