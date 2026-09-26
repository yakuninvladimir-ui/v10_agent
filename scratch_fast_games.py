import json
import glob
import os

games = sorted(os.listdir('public_games-dataset'))
print(f"Total games: {len(games)}")

for g in games:
    files = glob.glob(f'public_games-dataset/{g}/*.recording.jsonl')
    if not files:
        continue
    fpath = files[0]
    # Read first frame and last frame quickly
    with open(fpath, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        # Seek near end
        f.seek(0, os.SEEK_END)
        size = f.tell()
        # Read last 8KB
        f.seek(max(0, size - 8192))
        lines = f.readlines()
        last_line = lines[-1] if lines else first_line
    
    d0 = json.loads(first_line).get('data', {})
    d_end = json.loads(last_line).get('data', {})
    
    avail = d0.get('available_actions', [])
    win_levels = d0.get('win_levels', 0)
    end_level = d_end.get('levels_completed', 0)
    end_state = d_end.get('state', '')
    
    print(f"{g:6s} | actions: {str(avail):25s} | win_lvls: {win_levels:2d} | completed: {end_level:2d} | end_state: {end_state:12s}")
