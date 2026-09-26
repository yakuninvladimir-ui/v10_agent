import arc_agi
import arcengine
import json
import shutil
import os
from lcld_competition_child import _observation
from v10_agent.observe import normalize_observation
from v10_agent.planning_set import build_planning_set
from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.prompt_builders.solver_prompt import build_solver_prompts
from v10_agent.frame_media import render_grid_png, render_annotated_frame_png
from v10_agent.memory_contours import GameMemory, EpistemicMemory

arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir='local_env_test')
env = arcade.make('ar25')

l0_moves = [arcengine.GameAction.ACTION2] * 10 + [arcengine.GameAction.ACTION3] * 5
for act in l0_moves:
    frame = env.step(act)

l1_moves = [arcengine.GameAction.ACTION3] * 2 + [arcengine.GameAction.ACTION5] * 1 + [arcengine.GameAction.ACTION2] * 8
for act in l1_moves:
    frame = env.step(act)

obs = _observation(frame, 26, 'ar25')
norm = normalize_observation(obs)
grid = norm['grid']

snap = extract_arga_snapshot(grid)
pset = build_planning_set(snap, available_actions=norm['available_actions'])
game_mem = GameMemory(game_id="ar25")
ep_mem = EpistemicMemory(level_id="level_2")

raw_png = render_grid_png(grid)
annotated_png = render_annotated_frame_png(grid, pset)

with open('level2_raw_frame.png', 'wb') as f:
    f.write(raw_png)
with open('level2_annotated_frame.png', 'wb') as f:
    f.write(annotated_png)

# Also copy to brain artifact dir so we can embed in markdown
brain_dir = r"C:\Users\Настя\.gemini\antigravity\brain\f469e745-6367-41e1-ae35-0b0d4956ceaa"
shutil.copy('level2_raw_frame.png', os.path.join(brain_dir, 'level2_raw_frame.png'))
shutil.copy('level2_annotated_frame.png', os.path.join(brain_dir, 'level2_annotated_frame.png'))

sys_p, user_p = build_solver_prompts(
    manifest={'functions': []},
    planning_set=pset,
    epistemic_memory=ep_mem,
    action_budget=40,
    game_memory=game_mem,
    has_image=True,
)

json_str = user_p[user_p.index('{'):user_p.rindex('}')+1]
payload = json.loads(json_str)

print('=== ALL PLANNING OBJECTS IN PROMPT ===')
for o in payload.get('planning_objects', []):
    print(f"{o.get('id')} ({o.get('alias', '')}): color={o.get('color')} bbox={o.get('bbox')} type={o.get('shape_type')} keys={list(o.keys())}")

ascii_patches = []
for o in payload.get('planning_objects', []):
    if 'compact_ascii' in o and o['compact_ascii']:
        ascii_patches.append((o['id'], o.get('alias', ''), o['color'], o['shape_type'], o['compact_ascii']))

print(f'\nTotal ASCII patches in prompt: {len(ascii_patches)}')
for aid, alias, col, st, asc in ascii_patches:
    print(f"\n=== Object {aid} (Alias {alias}): color={col}, type={st} ===")
    print('\n'.join(asc))
