"""Local vLLM Advisor client with multimodal support and offline Mock backend."""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
import urllib.error
import urllib.request
from typing import Any, Mapping

from v10_agent.config import V10Config

logger = logging.getLogger(__name__)


def sanitize_model_response(raw_text: str) -> str:
    """Sanitize model response, removing reasoning traces, special tokens, and ATEM protocol channels.

    Supports:
    - Muse-Glimmer channel format: <|start|>assistant to=self<|message|>...<|eom|>
                                   <|start|>assistant to=user<|message|>...<|eot|>
    - Qwen / DeepSeek thinking tags: <think>...</think>, <thought>...</thought>
    - Special tokens: <|start|>, <|message|>, <|eom|>, <|eot|>, <|begin_of_text|>, <|end_of_text|>
    """
    if not raw_text:
        return ""

    text = str(raw_text)

    # 1. If to=user channel marker is present, the final user message is after it
    user_channel_matches = list(
        re.finditer(r"<\|start\|>\s*assistant\s+to=user\s*<\|message\|>", text, flags=re.IGNORECASE)
    )
    if user_channel_matches:
        text = text[user_channel_matches[-1].end():]
    else:
        user_bare = list(re.finditer(r"(?:^|\n)to=user\s*<\|message\|>", text, flags=re.IGNORECASE))
        if user_bare:
            text = text[user_bare[-1].end():]

    # 2. Remove explicit reasoning blocks:
    # Muse Glimmer ATEM to=self reasoning channel
    text = re.sub(
        r"<\|start\|>\s*assistant\s+to=self\s*<\|message\|>.*?<\|eom\|>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Unclosed to=self up to next assistant tag
    text = re.sub(
        r"<\|start\|>\s*assistant\s+to=self\s*<\|message\|>.*?(?=<\|start\|>)",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Bare to=self<|message|>...<|eom|>
    text = re.sub(
        r"(?:^|\n)to=self\s*<\|message\|>.*?<\|eom\|>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Bare to=self block followed by newline and reasoning
    text = re.sub(
        r"(?:^|\n)to=self\s*\n.*?<\|eom\|>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Standard thinking tags (Qwen, DeepSeek, etc.)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "<think>" in text.lower():
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "<thought>" in text.lower():
        text = re.sub(r"<thought>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)

    # 3. Strip special tokens and protocol markers
    special_patterns = [
        r"<\|start\|>[^<]*<\|message\|>",
        r"<\|start\|>",
        r"<\|message\|>",
        r"<\|eom\|>",
        r"<\|eot\|>",
        r"<\|begin_of_text\|>",
        r"<\|end_of_text\|>",
        r"<\|finetune_right_pad\|>",
        r"<\|image_start\|>",
        r"<\|image_end\|>",
        r"<\|patch\|>",
        r"<\|image\|>",
        r"<\|video\|>",
    ]
    for pat in special_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    # 4. Strip stray prefix markers
    text = re.sub(r"^\s*to=(?:user|self)\s*", "", text, flags=re.IGNORECASE)

    return text.strip()


def _has_image_payload(image_data: Any) -> bool:
    if not image_data:
        return False
    if isinstance(image_data, bytes):
        return len(image_data) > 0
    if isinstance(image_data, (list, tuple)):
        return any(isinstance(item, bytes) and len(item) > 0 for item in image_data)
    if isinstance(image_data, dict):
        return any(isinstance(v, bytes) and len(v) > 0 for v in image_data.values())
    return False


def is_role_vision_enabled(config: V10Config, agent_role: str) -> bool:
    """Determine whether multimodal vision input is enabled for the specified agent role."""
    if agent_role == "solver":
        return getattr(config, "solver_multimodal_enabled", getattr(config, "multimodal_enabled", getattr(config, "qwen_multimodal_enabled", True)))
    if agent_role == "explorer":
        return getattr(config, "explorer_multimodal_enabled", getattr(config, "multimodal_enabled", getattr(config, "qwen_multimodal_enabled", True)))
    if agent_role == "coder":
        return getattr(config, "coder_multimodal_enabled", getattr(config, "multimodal_enabled", getattr(config, "qwen_multimodal_enabled", True)))
    return getattr(config, "multimodal_enabled", getattr(config, "qwen_multimodal_enabled", True))


def format_multimodal_message(
    text_prompt: str,
    image_png_bytes: bytes | list[bytes] | tuple[bytes, ...] | dict[str, bytes] | None = None,
) -> list[dict[str, Any]]:
    """Format OpenAI-compatible chat completion user message with optional base64 image(s)."""
    if not image_png_bytes:
        return [{"role": "user", "content": text_prompt}]

    valid_images: list[bytes] = []
    if isinstance(image_png_bytes, bytes):
        if len(image_png_bytes) > 0:
            valid_images.append(image_png_bytes)
    elif isinstance(image_png_bytes, dict):
        for v in image_png_bytes.values():
            if isinstance(v, bytes) and len(v) > 0:
                valid_images.append(v)
    elif isinstance(image_png_bytes, (list, tuple)):
        for item in image_png_bytes:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], bytes):
                if len(item[1]) > 0:
                    valid_images.append(item[1])
            elif isinstance(item, bytes) and len(item) > 0:
                valid_images.append(item)

    if not valid_images:
        return [{"role": "user", "content": text_prompt}]

    content: list[dict[str, Any]] = [{"type": "text", "text": text_prompt}]
    for img in valid_images:
        b64_image = base64.b64encode(img).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_image}"},
        })

    return [{"role": "user", "content": content}]


class BaseLLMAdvisor:
    """Interface for LLM Advisor backends."""

    def generate_chat(
        self,
        messages: list[dict[str, Any]],
        config: V10Config,
        agent_role: str = "generic",
    ) -> str:
        raise NotImplementedError

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        config: V10Config,
        image_bytes: bytes | list[bytes] | tuple[bytes, ...] | dict[str, bytes] | None = None,
        agent_role: str = "generic",
    ) -> str:
        role_vision_enabled = is_role_vision_enabled(config, agent_role)
        supports_vision = _has_image_payload(image_bytes) and role_vision_enabled

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        user_msg = format_multimodal_message(
            user_prompt,
            image_bytes if supports_vision else None,
        )
        messages.extend(user_msg)
        return self.generate_chat(messages, config=config, agent_role=agent_role)


class VLLMAdvisor(BaseLLMAdvisor):
    """Client for local vLLM OpenAI-compatible server (/v1/chat/completions)."""

    def __init__(self, base_url: str = "http://127.0.0.1:1234/v1", api_key: str = "EMPTY"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def generate_chat(
        self,
        messages: list[dict[str, Any]],
        config: V10Config,
        agent_role: str = "generic",
    ) -> str:
        url = f"{self.base_url}/chat/completions"

        # Use explicitly configured model, or cached model, or discover from /models
        model_id = getattr(config, "model_path", None) or config.qwen_model_path or getattr(self, "_cached_model_id", None)
        if not model_id:
            try:
                models_url = f"{self.base_url}/models"
                req = urllib.request.Request(models_url)
                if self.api_key and self.api_key != "EMPTY":
                    req.add_header("Authorization", f"Bearer {self.api_key}")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    if "data" in data and len(data["data"]) > 0:
                        model_id = data["data"][0]["id"]
                        self._cached_model_id = model_id
            except Exception as exc:
                logger.debug(f"Failed to query /models: {exc}")

        if not model_id:
            model_id = "Qwen/Qwen3.8-27B"

        max_tokens = getattr(config, "max_output_tokens", config.qwen_max_output_tokens)
        is_dashscope = "dashscope" in str(self.base_url).lower()
        if is_dashscope and max_tokens > 8192:
            max_tokens = 8192

        temp = getattr(config, "temperature", config.qwen_temperature)
        top_p = getattr(config, "top_p", config.qwen_top_p)
        top_k = getattr(config, "top_k", config.qwen_top_k)
        pres_penalty = getattr(config, "presence_penalty", config.qwen_presence_penalty)
        rep_penalty = getattr(config, "repeat_penalty", config.qwen_repeat_penalty)
        seed_val = getattr(config, "seed", config.qwen_seed)
        strength = getattr(config, "reasoning_strength", "xhigh")

        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "temperature": temp,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "presence_penalty": pres_penalty,
            "seed": seed_val,
        }

        if top_k > 0:
            payload["top_k"] = top_k
        if rep_penalty > 0:
            payload["repetition_penalty"] = rep_penalty

        chat_kwargs: dict[str, Any] = {
            "reasoning_effort": strength,
            "reasoning_strength": strength,
        }
        if getattr(config, "enable_thinking", True) or getattr(config, "qwen_enable_thinking", True):
            chat_kwargs["enable_thinking"] = True
            payload["enable_thinking"] = True
        payload["chat_template_kwargs"] = chat_kwargs
        payload["reasoning_effort"] = strength

        payload["stream"] = True

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        # Deadline reserve check (Flash Loop Recovery port)
        if hasattr(config, "is_deadline_exceeded") and config.is_deadline_exceeded():
            logger.warning(
                f"Aborting LLM request for role {agent_role!r}: remaining time ({config.remaining_time_seconds():.1f}s) "
                f"is within deadline reserve threshold ({config.deadline_reserve_seconds}s)."
            )
            return "{}"

        base_timeout = getattr(config, "timeout_seconds", config.qwen_timeout_seconds) or 700
        rem = config.remaining_time_seconds() if hasattr(config, "remaining_time_seconds") else None
        if rem is not None:
            available_for_req = max(2.0, rem - getattr(config, "deadline_reserve_seconds", 15.0))
            timeout = min(float(base_timeout), available_for_req)
        else:
            timeout = float(base_timeout)

        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    stream = io.TextIOWrapper(response, encoding="utf-8")
                    content_chunks: list[str] = []
                    reasoning_chunks: list[str] = []
                    first_line = stream.readline()
                    if not first_line:
                        return ""
                    line_stripped = first_line.strip()

                    # Non-streaming JSON response fallback
                    if line_stripped.startswith("{"):
                        rest = stream.read()
                        full_json_str = first_line + rest
                        try:
                            resp_json = json.loads(full_json_str)
                            choices = resp_json.get("choices", [])
                            if choices:
                                msg = choices[0].get("message", {})
                                c = str(msg.get("content") or "")
                                tc = msg.get("tool_calls")
                                if tc and isinstance(tc, list):
                                    tc_blocks = []
                                    for call in tc:
                                        fn = call.get("function", {})
                                        fname = fn.get("name", "")
                                        fargs = fn.get("arguments", "")
                                        tc_blocks.append(f'<atem:invoke name="{fname}">{fargs}</atem:invoke>')
                                    c = f"{c}\n" + "\n".join(tc_blocks)
                                r_val = msg.get("reasoning") or msg.get("reasoning_content")
                                if not c.strip():
                                    if r_val:
                                        logger.warning(
                                            f"vLLM response has empty content while reasoning is present ({len(str(r_val))} chars) for role {agent_role!r}."
                                        )
                                    return ""
                                return sanitize_model_response(c)
                        except Exception:
                            return ""

                    streaming_tool_calls: dict[int, dict[str, str]] = {}
                    sse_buffer = ""

                    # SSE stream chunks
                    def _parse_sse_line(l_str: str) -> bool:
                        nonlocal sse_buffer
                        if not l_str.startswith("data:"):
                            return False
                        d_str = l_str[5:].strip()
                        if d_str == "[DONE]":
                            return True
                        
                        sse_buffer += d_str
                        try:
                            chunk = json.loads(sse_buffer)
                            sse_buffer = ""  # successfully parsed, clear buffer
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                if delta.get("content"):
                                    content_chunks.append(delta["content"])
                                r_chunk = delta.get("reasoning") or delta.get("reasoning_content")
                                if r_chunk:
                                    reasoning_chunks.append(r_chunk)
                                tc = delta.get("tool_calls")
                                if tc and isinstance(tc, list):
                                    for item in tc:
                                        idx = item.get("index", 0)
                                        if idx not in streaming_tool_calls:
                                            streaming_tool_calls[idx] = {"name": "", "arguments": ""}
                                        fn = item.get("function", {})
                                        if fn.get("name"):
                                            streaming_tool_calls[idx]["name"] += fn["name"]
                                        if fn.get("arguments"):
                                            streaming_tool_calls[idx]["arguments"] += fn["arguments"]
                        except json.JSONDecodeError:
                            pass  # incomplete chunk, buffer it
                        except Exception:
                            sse_buffer = ""  # other error, reset to be safe
                        return False

                    if not _parse_sse_line(line_stripped):
                        for line in stream:
                            if _parse_sse_line(line.strip()):
                                break

                    full_content = "".join(content_chunks)
                    if not full_content.strip():
                        if reasoning_chunks:
                            r_chars = sum(len(c) for c in reasoning_chunks)
                            logger.warning(
                                f"vLLM streaming response has empty content while reasoning is present ({r_chars} chars across {len(reasoning_chunks)} chunks) for role {agent_role!r}."
                            )
                        full_content = ""
                    if streaming_tool_calls:
                        tc_blocks = []
                        for _, call_info in sorted(streaming_tool_calls.items()):
                            fn_name = call_info.get("name", "")
                            fn_args = call_info.get("arguments", "")
                            tc_blocks.append(f'<atem:invoke name="{fn_name}">{fn_args}</atem:invoke>')
                        full_content = f"{full_content}\n" + "\n".join(tc_blocks) if full_content else "\n".join(tc_blocks)
                    return sanitize_model_response(full_content)
            except Exception as exc:
                err_body = ""
                if isinstance(exc, urllib.error.HTTPError):
                    try:
                        err_body = f" Body: {exc.read().decode('utf-8', errors='replace')}"
                    except Exception:
                        pass
                if attempt == 0:
                    logger.warning(f"vLLM query attempt 1 failed for role {agent_role!r}: {exc}.{err_body} Retrying once...")
                    time.sleep(1.0)
                    continue
                logger.error(f"vLLM query failed for role {agent_role!r} after retry: {exc}.{err_body}")
                raise


class MockLLMAdvisor(BaseLLMAdvisor):
    """Deterministic in-memory mock backend for offline testing without GPU."""

    def __init__(self) -> None:
        self.responses_by_role: dict[str, list[str]] = {
            "explorer": [],
            "coder": [],
            "solver": [],
            "generic": [],
        }
        self.call_history: list[dict[str, Any]] = []

    def set_response(self, role: str, response: str) -> None:
        self.responses_by_role.setdefault(role, []).append(response)

    def generate_chat(
        self,
        messages: list[dict[str, Any]],
        config: V10Config,
        agent_role: str = "generic",
    ) -> str:
        last_user_text = ""
        sys_text = ""
        for m in messages:
            if m.get("role") == "system":
                sys_text = m.get("content", "")
            elif m.get("role") == "user":
                c = m.get("content", "")
                if isinstance(c, str):
                    last_user_text = c
                elif isinstance(c, list):
                    for part in c:
                        if isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                            last_user_text = part.get("text", "")
                            break

        self.call_history.append({
            "role": agent_role,
            "system_prompt": sys_text,
            "user_prompt": last_user_text,
            "messages": [dict(m) for m in messages],
            "has_image": False,
        })
        queue = self.responses_by_role.get(agent_role, [])
        if queue:
            return queue.pop(0)
        default_queue = self.responses_by_role.get("generic", [])
        if default_queue:
            return default_queue.pop(0)

        if agent_role == "solver_reflection":
            return (
                "<distilled_invariants>\n"
                "- [GOAL]: Level goal is satisfied when targets are covered by pieces and their mirror reflections.\n"
                "- [ENTITIES]: Movable axis acts as a reflection plane; pieces move independently into symmetric target sockets.\n"
                "- [CONTROL]: ACTION5 sequentially shifts active focus across controllable entities.\n"
                "</distilled_invariants>"
            )

        if agent_role == "coder":
            act = "RESET" if "('RESET',)" in last_user_text or "ALL AVAILABLE ENVIRONMENT ACTIONS" not in last_user_text else "ACTION1"
            for candidate in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"):
                if f"- {candidate}" in last_user_text:
                    act = candidate
                    break
            return (
                f"```python\n"
                f"def action1(api):\n"
                f"    return api.declare_environment_action('{act}')\n"
                f"```\n"
                f"```json\n"
                f"{{\n"
                f'  "schema_version": "v10.dsl_manifest.1",\n'
                f'  "functions": [{{"name": "action1", "parameters": []}}]\n'
                f"}}\n"
                f"```"
            )
        if agent_role == "solver":
            act = "RESET" if "('RESET',)" in last_user_text else "ACTION1"
            for candidate in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"):
                if f"- {candidate}" in last_user_text or f"'{candidate}' active" in last_user_text or f"'{candidate}'" in last_user_text:
                    act = candidate
                    break
            return (
                "<invariant_analysis>\n"
                f"- Basic Laws Confirmed: {act} active\n"
                "Invariants & Goal: Advance level\n"
                "Strategy: Call action1\n"
                "</invariant_analysis>\n"
                "<trajectory_1>\n"
                "1. action1()\n"
                "</trajectory_1>\n"
            )
        return "{}"

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        config: V10Config,
        image_bytes: bytes | list[bytes] | tuple[bytes, ...] | dict[str, bytes] | None = None,
        agent_role: str = "generic",
    ) -> str:
        role_vision_enabled = is_role_vision_enabled(config, agent_role)
        has_image = _has_image_payload(image_bytes) and role_vision_enabled
        self.call_history.append({
            "role": agent_role,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "has_image": has_image,
        })
        queue = self.responses_by_role.get(agent_role, [])
        if queue:
            return queue.pop(0)
        default_queue = self.responses_by_role.get("generic", [])
        if default_queue:
            return default_queue.pop(0)

        if agent_role == "coder":
            act = "RESET" if "('RESET',)" in user_prompt or "ALL AVAILABLE ENVIRONMENT ACTIONS" not in user_prompt else "ACTION1"
            for candidate in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"):
                if f"- {candidate}" in user_prompt:
                    act = candidate
                    break
            return (
                f"```python\n"
                f"def action1(api):\n"
                f"    return api.declare_environment_action('{act}')\n"
                f"```\n"
                f"```json\n"
                f"{{\n"
                f'  "schema_version": "v10.dsl_manifest.1",\n'
                f'  "functions": [{{"name": "action1", "parameters": []}}]\n'
                f"}}\n"
                f"```"
            )
        if agent_role == "solver":
            act = "RESET" if "('RESET',)" in user_prompt else "ACTION1"
            for candidate in ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"):
                if f"- {candidate}" in user_prompt or f"'{candidate}' active" in user_prompt or f"'{candidate}'" in user_prompt:
                    act = candidate
                    break
            return (
                "<invariant_analysis>\n"
                f"- Basic Laws Confirmed: {act} active\n"
                "Invariants & Goal: Advance level\n"
                "Strategy: Call action1\n"
                "</invariant_analysis>\n"
                "<trajectory_1>\n"
                "1. action1()\n"
                "</trajectory_1>\n"
            )
        return "{}"


class DashScopeResponsesAdvisor(BaseLLMAdvisor):
    """Client for DashScope Responses API using openai.OpenAI SDK with enable_thinking=True."""

    def __init__(
        self,
        base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        api_key: str = "",
    ):
        self.base_url = (base_url or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1").rstrip("/")
        self.api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def generate_chat(
        self,
        messages: list[dict[str, Any]],
        config: V10Config,
        agent_role: str = "generic",
    ) -> str:
        client = self._get_client()
        raw_model = (
            getattr(self, "_cached_model_id", None)
            or config.qwen_model_path
            or getattr(config, "model_path", None)
            or "qwen3.8-27b"
        )
        if "/" in raw_model:
            tail = raw_model.split("/")[-1].lower()
            if "27b" in tail:
                model_id = "qwen3.8-27b"
            elif "flash" in tail or "3.7" in tail:
                model_id = "qwen3.7-flash"
            else:
                model_id = tail
        else:
            model_id = raw_model

        extra_body: dict[str, Any] = {}
        if config.qwen_enable_thinking:
            extra_body["enable_thinking"] = True
        else:
            extra_body["enable_thinking"] = False

        instructions = ""
        user_input: list[dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "system":
                instructions = m.get("content", "")
            else:
                raw_content = m.get("content")
                if isinstance(raw_content, list):
                    adapted_content: list[dict[str, Any]] = []
                    for part in raw_content:
                        if isinstance(part, dict):
                            p_type = part.get("type")
                            if p_type == "text":
                                adapted_content.append({"type": "input_text", "text": part.get("text", "")})
                            elif p_type == "image_url":
                                img_url = part.get("image_url", "")
                                if isinstance(img_url, dict):
                                    img_url = img_url.get("url", "")
                                adapted_content.append({"type": "input_image", "image_url": str(img_url)})
                            elif p_type in ("input_text", "input_image", "input_file"):
                                adapted_content.append(part)
                            else:
                                adapted_content.append(part)
                        else:
                            adapted_content.append({"type": "input_text", "text": str(part)})
                    user_input.append({"role": m.get("role", "user"), "content": adapted_content})
                else:
                    user_input.append(m)

        for attempt in range(3):
            try:
                resp = client.responses.create(
                    model=model_id,
                    instructions=instructions,
                    input=user_input,
                    extra_body=extra_body,
                )
                answer_text = ""
                reasoning_text = ""
                for item in getattr(resp, "output", []):
                    if getattr(item, "type", None) == "reasoning":
                        for s in getattr(item, "summary", []):
                            reasoning_text += getattr(s, "text", str(s))
                    elif getattr(item, "type", None) == "message":
                        for c in getattr(item, "content", []):
                            answer_text += getattr(c, "text", str(c))

                if reasoning_text:
                    print(f"\n[Reasoning ({agent_role})]")
                    for summary_chunk in [reasoning_text[i:i+500] for i in range(0, min(1500, len(reasoning_text)), 500)]:
                        print(summary_chunk)
                    print(flush=True)
                    logger.info(f"[{agent_role.upper()}] Reasoning preview: {reasoning_text[:300].strip()}...")
                if answer_text:
                    print(f"[Answer ({agent_role})]")
                    print(answer_text[:500] + ("..." if len(answer_text) > 500 else ""), flush=True)
                    return sanitize_model_response(answer_text)
                return str(resp)
            except Exception as exc:
                if attempt < 2:
                    logger.warning(
                        f"DashScopeResponsesAdvisor attempt {attempt+1} failed for {agent_role!r}: {exc}. Retrying in 2s..."
                    )
                    time.sleep(2.0)
                    continue
                logger.error(f"DashScopeResponsesAdvisor failed for {agent_role!r} after retries: {exc}")
                raise


class DashScopeStreamingChatAdvisor(BaseLLMAdvisor):
    """Client for DashScope Streaming ChatCompletions API with reasoning content support."""

    def __init__(
        self,
        base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        api_key: str = "",
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def generate_chat(
        self,
        messages: list[dict[str, Any]],
        config: V10Config,
        agent_role: str = "generic",
    ) -> str:
        client = self._get_client()
        raw_model = getattr(self, "_cached_model_id", None) or config.qwen_model_path or getattr(config, "model_path", None) or "qwen3.8-27b"
        if "/" in raw_model:
            tail = raw_model.split("/")[-1].lower()
            if "27b" in tail:
                model_id = "qwen3.8-27b"
            elif "flash" in tail or "3.7" in tail:
                model_id = "qwen3.7-flash"
            else:
                model_id = tail
        else:
            model_id = raw_model

        extra_body: dict[str, Any] = {}
        if config.qwen_enable_thinking:
            extra_body["enable_thinking"] = True

        for attempt in range(3):
            try:
                kwargs: dict[str, Any] = {
                    "model": model_id,
                    "messages": messages,
                    "stream": True,
                }
                if extra_body:
                    kwargs["extra_body"] = extra_body
                completion = client.chat.completions.create(**kwargs)
                reasoning_chunks: list[str] = []
                content_chunks: list[str] = []

                is_answering = False
                print(f"\n{'=' * 20}Thinking process [{agent_role.upper()}]{'=' * 20}", flush=True)
                for chunk in completion:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                        reasoning_chunks.append(delta.reasoning_content)
                        if not is_answering:
                            try:
                                print(delta.reasoning_content, end="", flush=True)
                            except Exception:
                                pass
                    if hasattr(delta, "content") and delta.content:
                        content_chunks.append(delta.content)
                        if not is_answering:
                            print(f"\n{'=' * 20}Full response [{agent_role.upper()}]{'=' * 20}", flush=True)
                            is_answering = True
                        try:
                            print(delta.content, end="", flush=True)
                        except Exception:
                            pass
                print("", flush=True)

                reasoning_text = "".join(reasoning_chunks)
                answer_text = "".join(content_chunks).strip()

                if reasoning_text:
                    logger.info(
                        f"[{agent_role.upper()}] Reasoning preview ({len(reasoning_text)} chars): {reasoning_text[:200].strip()}..."
                    )
                if answer_text:
                    logger.info(
                        f"[{agent_role.upper()}] Answer preview ({len(answer_text)} chars): {answer_text[:200].strip()}..."
                    )
                    return answer_text
                if reasoning_text:
                    logger.warning(
                        f"[{agent_role.upper()}] Content stream was empty; falling back to reasoning stream ({len(reasoning_text)} chars)"
                    )
                    return reasoning_text
                return "{}"
            except Exception as exc:
                if attempt < 2:
                    logger.warning(
                        f"DashScopeStreamingChatAdvisor attempt {attempt+1} failed for {agent_role!r}: {exc}. Retrying in 2s..."
                    )
                    time.sleep(2.0)
                    continue
                logger.error(f"DashScopeStreamingChatAdvisor failed for {agent_role!r} after retries: {exc}")
                raise


def build_llm_advisor(config: V10Config) -> BaseLLMAdvisor:
    """Factory creating appropriate advisor backend based on configuration."""
    if config.llm_advisor_backend == "fake":
        return MockLLMAdvisor()
    if (
        config.llm_advisor_backend == "dashscope_responses"
        or "protocols/compatible-mode" in (config.qwen_vllm_base_url or "")
    ):
        return DashScopeResponsesAdvisor(
            base_url=config.qwen_vllm_base_url,
            api_key=config.qwen_vllm_api_key,
        )
    if (
        config.llm_advisor_backend in ("dashscope_streaming", "dashscope_chat")
        or "dashscope-intl.aliyuncs.com/compatible-mode/v1" in (config.qwen_vllm_base_url or "")
    ):
        return DashScopeStreamingChatAdvisor(
            base_url=config.qwen_vllm_base_url,
            api_key=config.qwen_vllm_api_key,
        )
    return VLLMAdvisor(
        base_url=config.qwen_vllm_base_url,
        api_key=config.qwen_vllm_api_key,
    )

