import os, json, base64
import requests

IPFS_API = os.getenv("IPFS_API", "http://127.0.0.1:5001")

def put_json(obj) -> str:
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    files = {"file": ("receipt_chain.json", data, "application/json")}
    r = requests.post(f"{IPFS_API}/api/v0/add", files=files)
    r.raise_for_status()
    cid = r.json()["Hash"]
    return f"ipfs://{cid}/receipt_chain.json"

def get_json(cid_uri: str):
    assert cid_uri.startswith("ipfs://")
    cid = cid_uri[len("ipfs://"):].split("/")[0]
    r = requests.get(f"{IPFS_API}/api/v0/cat?arg={cid}")
    r.raise_for_status()
    return json.loads(r.text)
