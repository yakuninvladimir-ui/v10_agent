"""
vLLM Lifecycle Manager for V10 Agent.
Manages the vLLM server process for Qwen 3.8B FP8.
Ref: Spec 8.2 (vLLM Lifecycle)
"""

import os
import sys
import subprocess
import time
import signal
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class VLLMConfig:
    """Configuration for vLLM server."""
    model_path: str
    tensor_parallel_size: int = 1
    max_num_seqs: int = 4
    max_model_len: int = 131072
    gpu_memory_utilization: float = 0.9
    dtype: str = "auto"
    reasoning_parser: Optional[str] = "qwen3"
    port: int = 8000


class VLLMManager:
    """
    Manages the vLLM OpenAI-compatible API server lifecycle.
    
    Features:
    - Start/stop with proper signal handling
    - Health check polling
    - Bounded log tail reading (critical for Kaggle memory limits)
    """
    
    def __init__(self, model_path: str, config: Optional[VLLMConfig] = None):
        self.model_path = model_path
        self.config = config or VLLMConfig(model_path=model_path)
        self.process: Optional[subprocess.Popen] = None
        self.log_file_path = "/kaggle/working/vllm_qwen.log"
        
    def start(self) -> None:
        """
        Starts the vLLM API server as a subprocess.
        Logs are redirected to self.log_file_path.
        """
        if self.process is not None:
            raise RuntimeError("vLLM server is already running")
        
        # Build command line arguments
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.config.model_path,
            "--tensor-parallel-size", str(self.config.tensor_parallel_size),
            "--max-num-seqs", str(self.config.max_num_seqs),
            "--max-model-len", str(self.config.max_model_len),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            "--dtype", self.config.dtype,
            "--port", str(self.config.port),
        ]
        
        if self.config.reasoning_parser:
            cmd.extend(["--reasoning-parser", self.config.reasoning_parser])
        
        # Open log file for appending
        log_file = open(self.log_file_path, "a")
        
        self.process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=os.environ.copy()
        )
        print(f"[OK] vLLM server started with PID {self.process.pid}")
        
    def wait_for_health(self, timeout: int = 900, poll_interval: float = 5.0) -> bool:
        """
        Polls the /health endpoint until the server is ready.
        
        Args:
            timeout: Maximum seconds to wait
            poll_interval: Seconds between polls
            
        Returns:
            True if healthy, False if timeout
        """
        import urllib.request
        import urllib.error
        
        health_url = f"http://localhost:{self.config.port}/health"
        start_time = time.time()
        
        print(f"Waiting for vLLM health check (timeout={timeout}s)...")
        
        while time.time() - start_time < timeout:
            if self.process and self.process.poll() is not None:
                # Process died
                print(f"[ERROR] vLLM process died with code {self.process.returncode}")
                return False
                
            try:
                req = urllib.request.Request(health_url, method='GET')
                with urllib.request.urlopen(req, timeout=10) as response:
                    if response.status == 200:
                        print("[OK] vLLM server is healthy")
                        return True
            except (urllib.error.URLError, ConnectionRefusedError, TimeoutError):
                pass
            
            time.sleep(poll_interval)
        
        print(f"[ERROR] vLLM health check timed out after {timeout}s")
        return False
        
    def get_bounded_log_tail(self, bytes_limit: int = 12000) -> str:
        """
        Reads only the tail of the log file to avoid memory issues.
        Uses seek from end of file.
        
        Args:
            bytes_limit: Maximum bytes to read from end of file
            
        Returns:
            String containing the last `bytes_limit` bytes of the log
        """
        if not os.path.exists(self.log_file_path):
            return ""
        
        try:
            with open(self.log_file_path, 'rb') as f:
                f.seek(0, 2)  # Seek to end
                file_size = f.tell()
                
                if file_size <= bytes_limit:
                    f.seek(0)
                    content = f.read()
                else:
                    f.seek(file_size - bytes_limit)
                    content = f.read()
                    
                return content.decode('utf-8', errors='replace')
        except Exception as e:
            return f"[Error reading log: {e}]"
            
    def stop(self) -> None:
        """
        Gracefully stops the vLLM server.
        Sends SIGTERM, waits, then SIGKILL if necessary.
        """
        if self.process is None:
            return
            
        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            print("[OK] vLLM server stopped")
        except Exception as e:
            print(f"[WARN] Error stopping vLLM: {e}")
        finally:
            self.process = None

    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.0) -> str:
        """
        Simple generation method using the OpenAI-compatible API.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        import urllib.request
        import json
        
        api_url = f"http://localhost:{self.config.port}/v1/completions"
        
        payload = {
            "model": self.model_path,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            api_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method='POST'
        )
        
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get("choices", [{}])[0].get("text", "")
        except Exception as e:
            return f"[Generation Error: {e}]"
