"""Tests for Qwen thinking tags insulation and Phase A Heavy Smoke test pipeline."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from v10_agent.dsl_coder import DSLCoder, extract_code_and_manifest
from v10_agent.explorer_agent import extract_json_block
from v10_agent.llm_advisor import BaseLLMAdvisor, format_multimodal_message
from v10_agent.solver_agent import extract_json_block as extract_solver_block
import phase_a_heavy_smoke


def test_explorer_strips_thinking_tags():
    raw_response = (
        "<think>\n"
        "Let's think about this ARC grid.\n"
        "There is a red box at {0, 0} to {5, 5}.\n"
        '{"temporary_reasoning": true}\n'
        "</think>\n"
        "```json\n"
        '{\n  "schema_version": "v10.env_spec.1",\n  "observations": ["red box detected"]\n}\n'
        "```"
    )
    result = extract_json_block(raw_response)
    assert result is not None
    assert result.get("schema_version") == "v10.env_spec.1"
    assert result.get("observations") == ["red box detected"]


def test_dsl_coder_strips_thinking_tags():
    raw_response = (
        "<think>\n"
        "I should implement a move_right helper function.\n"
        "```python\ndef bad(): pass\n```\n"
        "</think>\n"
        "```python\n"
        "def certified_move(x: int) -> int:\n"
        "    return x + 1\n"
        "```\n"
        "```json\n"
        '{"schema_version": "v10.dsl_manifest.1", "functions": [{"name": "certified_move"}]}\n'
        "```"
    )
    source, manifest = extract_code_and_manifest(raw_response)
    assert source is not None
    assert "certified_move" in source
    assert "def bad():" not in source
    assert manifest is not None
    assert manifest.get("schema_version") == "v10.dsl_manifest.1"


def test_solver_strips_thinking_tags():
    raw_response = (
        "<think>\n"
        "Planning trajectory for agent...\n"
        '{"fake": 1}\n'
        "</think>\n"
        "```json\n"
        '{\n  "schema_version": "v10.trajectory_package.1",\n  "candidates": []\n}\n'
        "```"
    )
    res = extract_solver_block(raw_response)
    assert res is not None
    assert res.get("schema_version") == "v10.trajectory_package.1"


def test_phase_a_heavy_smoke_offline_dry_run():
    # Verify smoke pipeline runs and exits gracefully without crashing when offline
    summary = phase_a_heavy_smoke.run_phase_a_smoke_pipeline()
    assert isinstance(summary, dict)
    assert summary.get("status") in ("skipped_no_wheelhouse", "skipped_no_model", "success")


def test_build_multimodal_smoke_payload():
    payload = phase_a_heavy_smoke._build_multimodal_smoke_payload("test-model")
    assert payload["model"] == "test-model"
    assert payload["chat_template_kwargs"]["reasoning_effort"] == "xhigh"
    assert payload["chat_template_kwargs"]["enable_thinking"] is True
    assert payload["temperature"] == 1.0
    assert payload["top_k"] == 20
    assert payload["max_tokens"] >= 8192
    assert len(payload["messages"]) == 2
    user_content = payload["messages"][1]["content"]
    assert any(item.get("type") == "image_url" for item in user_content)

    # Verify Solver Multimodal specific structures
    assert "_meta" in payload
    meta = payload["_meta"]
    assert meta["has_image"] is True
    assert len(meta["png_bytes"]) > 0
    assert meta["png_bytes"].startswith(b"\x89PNG\r\n\x1a\n")
    assert meta["grid_hash"] == meta["planning_set"].grid_hash

    user_text = next(item["text"] for item in user_content if item.get("type") == "text")
    assert "solver_raw_frame.png is the exact same frame as solver_annotated_frame.png, but without object annotations." in user_text
    assert f'"grid_hash": "{meta["grid_hash"]}"' in user_text
    assert '"planning_objects":' in user_text

    # Verify disabled toggle behavior
    from v10_agent.config import V10Config
    cfg_disabled = V10Config(model_path="test-model", solver_multimodal_enabled=False)
    payload_disabled = phase_a_heavy_smoke._build_multimodal_smoke_payload("test-model", config=cfg_disabled)
    user_content_disabled = payload_disabled["messages"][1]["content"]
    assert not any(item.get("type") == "image_url" for item in user_content_disabled)
    user_text_disabled = user_content_disabled[0]["text"]
    assert "solver_raw_frame.png is the exact same frame" not in user_text_disabled


def test_explorer_strips_muse_glimmer_channel_tags():
    raw_response = (
        "<|start|>assistant to=self<|message|>\n"
        "Let's think step by step about the ARC puzzle.\n"
        "The agent sees a blue pixel at (2, 3).\n"
        '{"internal_hypothesis": "movement", "schema_version": "wrong.schema"}\n'
        "<|eom|>\n"
        "<|start|>assistant to=user<|message|>\n"
        "```json\n"
        "{\n"
        '  "schema_version": "v10.env_spec.1",\n'
        '  "observations": ["blue pixel at 2, 3"]\n'
        "}\n"
        "```\n"
        "<|eot|>"
    )
    result = extract_json_block(raw_response)
    assert result is not None
    assert result.get("schema_version") == "v10.env_spec.1"
    assert result.get("observations") == ["blue pixel at 2, 3"]


def test_dsl_coder_strips_muse_glimmer_channel_tags():
    raw_response = (
        "<|start|>assistant to=self<|message|>\n"
        "Thinking about the DSL implementation:\n"
        "```python\ndef wrong(): return 0\n```\n"
        "<|eom|>\n"
        "<|start|>assistant to=user<|message|>\n"
        "```python\n"
        "def solve_grid(grid):\n"
        "    return grid\n"
        "```\n"
        "```json\n"
        '{"schema_version": "v10.dsl_manifest.1", "functions": [{"name": "solve_grid"}]}\n'
        "```\n"
        "<|eot|>"
    )
    source, manifest = extract_code_and_manifest(raw_response)
    assert source is not None
    assert "def solve_grid" in source
    assert "def wrong" not in source
    assert manifest is not None
    assert manifest.get("schema_version") == "v10.dsl_manifest.1"


def test_solver_parses_atem_and_strips_channel_tags():
    from v10_agent.solver_agent import parse_text_trajectory
    raw_response = (
        "<|start|>assistant to=self<|message|>\n"
        "Planning steps in secret...\n"
        "[TRAJECTORY]\n"
        '[{"action": "WRONG_UP"}]\n'
        "<|eom|>\n"
        "<|start|>assistant to=user<|message|>\n"
        "[INVARIANT_EVOLUTION]\n"
        "Pattern holds across all examples.\n"
        "[HYPOTHESIS]\n"
        "Move right until wall.\n"
        "[TRAJECTORY]\n"
        '[{"action": "RIGHT", "params": {}}]\n'
        "<|eot|>"
    )
    traj_pkg = parse_text_trajectory(raw_response)
    assert traj_pkg is not None
    assert traj_pkg.get("schema_version") == "v10.trajectory_package.1"
    assert traj_pkg.get("hypothesis") == "Move right until wall."
    assert len(traj_pkg["candidates"]) == 1
    steps = traj_pkg["candidates"][0]["steps"]
    assert len(steps) == 1
    assert steps[0]["dsl_function"] == "RIGHT"


def test_solver_parses_atem_invoke_function_calls():
    from v10_agent.solver_agent import parse_text_trajectory
    raw_response = (
        "<|start|>assistant to=self<|message|>Thinking...<|eom|>"
        "<|start|>assistant to=user<|message|>\n"
        '<atem:invoke name="ACTION_STEP">{"action": "DOWN", "x": 4, "y": 5}</atem:invoke>\n'
        "<|eot|>"
    )
    traj_pkg = parse_text_trajectory(raw_response)
    assert traj_pkg is not None
    steps = traj_pkg["candidates"][0]["steps"]
    assert len(steps) == 1
    assert steps[0]["dsl_function"] == "ACTION_STEP"
    assert steps[0]["arguments"]["action"] == "DOWN"
    assert steps[0]["arguments"]["x"] == 4


def test_sanitize_model_response_universal():
    from v10_agent.llm_advisor import sanitize_model_response
    raw = (
        "<|start|>assistant to=self<|message|>Secret internal reasoning<|eom|>"
        "<|start|>assistant to=user<|message|>Hello world<|eot|>"
    )
    assert sanitize_model_response(raw).strip() == "Hello world"

    raw_think = "<think>Hidden thought</think>Direct answer"
    assert sanitize_model_response(raw_think).strip() == "Direct answer"


def test_agent_config_qwen_compatibility():
    from v10_agent.config import AgentConfig
    cfg = AgentConfig()
    # Canonical Qwen 3.8 27B fields
    assert cfg.model_name == "Qwen/Qwen3.8-27B"
    assert cfg.reasoning_strength == "xhigh"
    assert cfg.temperature == 1.0
    assert cfg.top_k == 20
    assert cfg.top_p == 0.95
    assert cfg.enable_thinking is True
    # Backward-compatible qwen aliases
    assert cfg.qwen_model_name == "Qwen/Qwen3.8-27B"
    assert cfg.qwen_temperature == 1.0
    assert cfg.qwen_top_p == 0.95
    assert cfg.qwen_top_k == 20
    assert cfg.qwen_enable_thinking is True


def test_qwen_prompt_builders_thinking_instructions():
    from v10_agent.arga_lite import extract_arga_snapshot
    from v10_agent.planning_set import build_planning_set
    from v10_agent.prompt_builders.explorer_prompt import build_explorer_prompts
    from v10_agent.prompt_builders.coder_prompt import build_coder_prompts
    from v10_agent.prompt_builders.solver_prompt import build_solver_prompts

    grid = [[0, 1, 0]]
    planning_set = build_planning_set(extract_arga_snapshot(grid), ["ACTION1"])
    manifest = {
        "functions": [
            {"name": "test_func", "parameters": [{"name": "obj", "type": "str"}], "docstring": "test doc"}
        ]
    }
    env_spec = {
        "schema_version": "v10.env_spec.1",
        "researched_actions": [{"action_id": "ACTION1", "effect": "moves"}],
    }

    exp_sys, _ = build_explorer_prompts(planning_set)
    coder_sys, _ = build_coder_prompts(env_spec)
    solver_sys, _ = build_solver_prompts(manifest, planning_set)

    for sys_prompt in (exp_sys, coder_sys, solver_sys):
        # Must instruct to think step-by-step
        assert "think" in sys_prompt.lower()
        # Must NOT include the Muse-style Reasoning strength header
        assert "Reasoning strength:" not in sys_prompt
        # Must NOT contain Muse-specific channel syntax
        assert "<|start|>assistant" not in sys_prompt


def test_patch_vllm_muse_reasoning_parser(tmp_path):
    # Create mock vllm reasoning parser file
    vllm_dir = tmp_path / "vllm" / "reasoning"
    vllm_dir.mkdir(parents=True)
    parser_file = vllm_dir / "muse_glimmer_reasoning_parser.py"
    initial_code = (
        "_REASONING_OPEN = 'to=self<|message|>'\n"
        "_EOM = '<|eom|>'\n"
        "class MuseGlimmerReasoningParser(ReasoningParser):\n"
        "    def __init__(self, tokenizer):\n"
        "        pass\n"
    )
    parser_file.write_text(initial_code, encoding="utf-8")

    assert phase_a_heavy_smoke.patch_vllm_muse_reasoning_parser(tmp_path) is True
    updated = parser_file.read_text(encoding="utf-8")
    assert "def reasoning_start_str" in updated
    assert "def reasoning_end_str" in updated
    assert "_REASONING_OPEN" in updated
    assert "_EOM" in updated

    # Idempotent: running again should not duplicate patch
    assert phase_a_heavy_smoke.patch_vllm_muse_reasoning_parser(tmp_path) is False


def test_smoke_send_reasoning_extraction():
    # Verify that response JSON with 'reasoning' (vLLM 0.28 format) is parsed correctly
    body_json = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Final answer text",
                "reasoning": "Step 1: thinking. Step 2: validating.",
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 50}
    }
    # Simulate extraction as done in _smoke_send
    choice = body_json["choices"][0]
    msg = choice["message"]
    content = str(msg.get("content") or "")
    reasoning = str(msg.get("reasoning") or msg.get("reasoning_content") or "")
    assert content == "Final answer text"
    assert reasoning == "Step 1: thinking. Step 2: validating."
    assert len(reasoning) > 0


def test_empty_content_with_reasoning_does_not_return_raw_thoughts():
    """Verify that when vLLM truncates in thinking or emits empty content, thoughts are not returned as response."""
    import io
    from unittest.mock import patch, MagicMock
    from v10_agent.config import V10Config
    from v10_agent.llm_advisor import VLLMAdvisor

    client = VLLMAdvisor(base_url="http://127.0.0.1:1234/v1")
    cfg = V10Config()

    # Case 1: Non-streaming JSON response with empty content and reasoning
    non_streaming_body = json.dumps({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "reasoning": "Let's think step by step. First examine the blue pixel...",
            },
            "finish_reason": "length",
        }]
    }).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.readline.return_value = non_streaming_body.decode("utf-8")
    mock_resp.read.return_value = ""
    mock_resp.__enter__.return_value = io.BytesIO(non_streaming_body)

    with patch("urllib.request.urlopen", return_value=mock_resp.__enter__.return_value):
        res = client.generate("sys", "user", cfg, agent_role="solver")
        assert res == ""  # Must NOT return "Let's think step by step..."!

    # Case 2: Streaming SSE chunks with reasoning only, no content
    sse_lines = (
        'data: {"choices": [{"delta": {"reasoning": "Thinking step 1..."}}]}\n'
        'data: {"choices": [{"delta": {"reasoning": "Thinking step 2..."}}]}\n'
        'data: [DONE]\n'
    ).encode("utf-8")

    with patch("urllib.request.urlopen", return_value=io.BytesIO(sse_lines)):
        res_stream = client.generate("sys", "user", cfg, agent_role="solver")
        assert res_stream == ""  # Must NOT return raw reasoning chunks!


def test_phase_b_smoke_probe_payload_structure():
    """Verify that Phase B model smoke probe disables thinking and is fast/safe."""
    import io
    from unittest.mock import patch, MagicMock
    import phase_a_heavy_smoke

    captured_req: list[dict] = []

    def mock_urlopen(req, *args, **kwargs):
        req_url = req.full_url if hasattr(req, "full_url") else str(req)
        if "models" in req_url:
            return io.BytesIO(json.dumps({"data": [{"id": "Qwen/Qwen3.8-27B"}]}).encode("utf-8"))
        if "chat/completions" in req_url:
            body = json.loads(req.data.decode("utf-8"))
            captured_req.append(body)
            return io.BytesIO(json.dumps({
                "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2}
            }).encode("utf-8"))
        raise ValueError(f"Unexpected url: {req_url}")

    with patch("phase_a_heavy_smoke.urlopen", side_effect=mock_urlopen):
        res = phase_a_heavy_smoke.phase_b_model_smoke_or_die()
        assert res["status"] == "ok"
        assert len(captured_req) == 1
        p = captured_req[0]
        assert p["max_tokens"] <= 256
        assert p["chat_template_kwargs"].get("enable_thinking") is False



