import json
import glob
import os

files = glob.glob('public_games-dataset/*/*.recording.jsonl')

winning_recordings = []
for path in files:
    with open(path, 'r', encoding='utf-8') as f:
        # Read lines, check last line
        for line in f:
            pass
        last_d = json.loads(line).get('data', {})
        if last_d.get('state') == 'WIN' or last_d.get('levels_completed', 0) > 0:
            winning_recordings.append((path, last_d.get('levels_completed'), last_d.get('win_levels'), last_d.get('state')))

print(f"Total recordings with progress: {len(winning_recordings)}")
for p, lvl, win_lvl, state in winning_recordings[:10]:
    game = p.split(os.sep)[1]
    print(f"{game}: levels_completed={lvl}/{win_lvl}, final_state={state}, file={os.path.basename(p)}")
