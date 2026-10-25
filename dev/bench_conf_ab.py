import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from install import autotune

BINARY = Path("build/richengine")
ASSEMBLY = Path(sys.argv[1])
ENVS = [json.loads(arg) for arg in sys.argv[2:]]

prompts, real = autotune._prompts(ASSEMBLY, 8)
print(f"prompts real={real} windows={len(prompts)}", flush=True)

for env in ENVS:
    medians, samples = autotune.measure(
        BINARY, ASSEMBLY, env, prompts=prompts, widths=(1,), repetitions=3
    )
    print(json.dumps({"env": env, "medians": medians, "samples": samples}), flush=True)
