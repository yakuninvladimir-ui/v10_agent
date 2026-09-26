import json
import glob
import os

files = glob.glob('public_games-dataset/*/*.recording.jsonl')

for path in files[:3]:
    game = path.split(os.sep)[1]
    with open(path, 'r', encoding='utf-8') as f:
        l0 = json.loads(f.readline())
        l1 = json.loads(f.readline())
    d0 = l0.get('data', {})
    d1 = l1.get('data', {})
    print(f"Game: {game}")
    print(f"  Frame 0 data keys: {list(d0.keys())}")
    print(f"  Frame 0 action_input: {d0.get('action_input')}")
    print(f"  Frame 1 action_input: {d1.get('action_input')}")
    print(f"  Frame 0 available_actions: {d0.get('available_actions')}")
    print(f"  Frame 0 levels_completed: {d0.get('levels_completed')}, win_levels: {d0.get('win_levels')}")
    print(f"  Frame 0 state: {d0.get('state')}, Frame 1 state: {d1.get('state')}")
