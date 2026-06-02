import hashlib
import json
from pathlib import Path


class JsonDiskCache:
    def __init__(self, root_dir, namespace):
        self.root_dir = Path(root_dir)
        self.namespace = str(namespace)
        self.cache_dir = self.root_dir / self.namespace
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"enabled": True, "hits": 0, "misses": 0, "writes": 0, "errors": 0}

    def get(self, key):
        path = self._path(key)
        if not path.exists():
            self.stats["misses"] += 1
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            self.stats["errors"] += 1
            self.stats["misses"] += 1
            return None
        self.stats["hits"] += 1
        return payload

    def set(self, key, payload):
        path = self._path(key)
        tmp = path.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(payload, f, separators=(",", ":"), default=json_default)
            tmp.replace(path)
            self.stats["writes"] += 1
        except Exception:
            self.stats["errors"] += 1
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def diagnostics(self):
        out = dict(self.stats)
        out["root_dir"] = str(self.root_dir)
        out["namespace"] = self.namespace
        return out

    def _path(self, key):
        return self.cache_dir / f"{key}.json"


class DisabledCache:
    stats = {"enabled": False, "hits": 0, "misses": 0, "writes": 0, "errors": 0}

    def get(self, key):
        return None

    def set(self, key, payload):
        return None

    def diagnostics(self):
        return dict(self.stats)


def hash_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)
