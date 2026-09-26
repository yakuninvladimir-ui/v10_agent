import sys
sys.path.insert(0, r'c:\arcprize')
import arc_agi
from arcengine import GameAction
from lcld_competition_child import _current_frame, _observation
from v10_agent.observe import normalize_observation
from v10_agent.arga_lite import extract_arga_snapshot
from v10_agent.planning_set import build_planning_set
from v10_agent.judge import LayeredVerifier
from v10_agent.config import V10Config
from v10_agent.verification import GroundedStep
from v10_agent.types import PropositionSet
from v10_agent.memory_contours import GameMemory

arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir='local_env_test')
env = arcade.make('ar25')
game_mem = GameMemory(game_id='ar25')
game_mem.record_action_effect('ACTION1', 'Object moved UP')
game_mem.record_action_effect('ACTION2', 'Object moved DOWN')
game_mem.record_action_effect('ACTION3', 'Object moved LEFT')
game_mem.record_action_effect('ACTION4', 'Object moved RIGHT')
verifier = LayeredVerifier(V10Config())

# 1. Level 0
print("--- LEVEL 0 ---")
actions_l0 = ['ACTION2']*10 + ['ACTION3']*5
for i, act_name in enumerate(actions_l0, 1):
    f_b = _current_frame(env)
    norm_b = normalize_observation(_observation(f_b, i-1, 'ar25'), crop_border=1)
    snap_b = extract_arga_snapshot(norm_b['grid'])
    snap_b.levels_completed = norm_b.get('levels_completed', 0)
    pset = build_planning_set(snap_b, ['ACTION1','ACTION2','ACTION3','ACTION4','ACTION5'])

    act = getattr(GameAction, act_name)
    env.step(act)
    f_a = _current_frame(env)
    norm_a = normalize_observation(_observation(f_a, i, 'ar25'), crop_border=1)

    step = GroundedStep(step_id=f's{i}', dsl_function='step_action', arguments={}, expected_propositions=PropositionSet.from_iterable([]))
    j = verifier.evaluate_transition(step, snap_b, norm_a, pset, game_mem, {'action_id': act_name})
    levels = norm_a.get("levels_completed", getattr(f_a, 'levels_completed', 0))
    print(f"Step {i:02d} ({act_name}): verdict={j.verdict.name} | levels={levels} | expl={j.explanation[:60]}")

# 2. Level 1
print("\n--- LEVEL 1 ---")
actions_l1 = ['ACTION3', 'ACTION3', 'ACTION5'] + ['ACTION2']*8
for i, act_name in enumerate(actions_l1, 1):
    f_b = _current_frame(env)
    norm_b = normalize_observation(_observation(f_b, i-1, 'ar25'), crop_border=1)
    snap_b = extract_arga_snapshot(norm_b['grid'])
    snap_b.levels_completed = norm_b.get('levels_completed', 0)
    pset = build_planning_set(snap_b, ['ACTION1','ACTION2','ACTION3','ACTION4','ACTION5'])

    act = getattr(GameAction, act_name)
    env.step(act)
    f_a = _current_frame(env)
    norm_a = normalize_observation(_observation(f_a, i, 'ar25'), crop_border=1)

    step = GroundedStep(step_id=f's{i}', dsl_function='step_action', arguments={}, expected_propositions=PropositionSet.from_iterable([]))
    j = verifier.evaluate_transition(step, snap_b, norm_a, pset, game_mem, {'action_id': act_name})
    levels = norm_a.get("levels_completed", getattr(f_a, 'levels_completed', 0))
    print(f"Step {i:02d} ({act_name}): verdict={j.verdict.name} | levels={levels} | expl={j.explanation[:60]}")
