import re
import json
import pathlib

log_path = pathlib.Path(r"C:\Users\Настя\.gemini\antigravity\brain\13717e6b-b272-482a-aa18-9f58b167863b\.system_generated\tasks\task-4557.log")
lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()

print(f"Total log lines: {len(lines)}")

# Let's find sections
for i, line in enumerate(lines):
    if "=== STEP" in line or "--- [STEP" in line or "STAGE" in line or "PROBE" in line or "SOLVER" in line:
        # print some key milestones
        if any(k in line for k in ["Primitive Probing Phase", "Explorer", "DSL Manifest", "Solver", "Candidate", "Circuit broken", "persistent fallback"]):
            print(f"L{i+1}: {line[:120]}")
