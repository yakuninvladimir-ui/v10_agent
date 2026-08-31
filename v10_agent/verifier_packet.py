"""
Verifier Packet Module
"""
from typing import Dict, Any

def create_verifier_packet(observation: Dict[str, Any]) -> Dict[str, Any]:
    return {"packet_data": observation}
