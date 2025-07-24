import base64
import hashlib

__all__ = ["b64_to_hex", "hex_to_b64", "b64_md5"]

def b64_to_hex(s: str) -> str:
    return base64.b64decode(s).hex()


def hex_to_b64(s: str) -> str:
    return base64.b64encode(bytes.fromhex(s)).decode()


def b64_md5(s: bytes | str) -> str:
    if isinstance(s, str):
        s = s.encode()
    return hex_to_b64(hashlib.md5(s).hexdigest())