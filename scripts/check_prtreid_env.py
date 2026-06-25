#!/usr/bin/env python3
import argparse
import hashlib
import importlib
from pathlib import Path


REQUIRED_MODULES = ["torch", "prtreid", "torchreid", "omegaconf", "yacs"]
PRTREID_MD5 = "9633825232bc89f23a94522c5561650e"
HRNET_MD5 = "58ea12b0420aa3adaa2f74114c9f9721"


def main():
    parser = argparse.ArgumentParser(description="Check PRTReID runtime dependencies and checkpoints.")
    parser.add_argument("--weights-path", default="models/reid/prtreid-soccernet-baseline.pth.tar")
    parser.add_argument("--hrnet-pretrained-path", default="models/reid")
    parser.add_argument("--skip-md5", action="store_true")
    args = parser.parse_args()

    ok = True
    print("== Python modules ==")
    for name in REQUIRED_MODULES:
        try:
            module = importlib.import_module(name)
            print(f"{name}: OK {getattr(module, '__file__', '')}")
        except Exception as exc:
            ok = False
            print(f"{name}: MISSING ({type(exc).__name__}: {exc})")

    print("\n== Checkpoints ==")
    weights = Path(args.weights_path)
    hrnet = Path(args.hrnet_pretrained_path) / "hrnetv2_w32_imagenet_pretrained.pth"
    ok &= check_file(weights, PRTREID_MD5, args.skip_md5)
    ok &= check_file(hrnet, HRNET_MD5, args.skip_md5)

    if not ok:
        print("\nPRTReID is not ready.")
        print("Expected files:")
        print(f"- {weights}")
        print(f"- {hrnet}")
        print("Install the PRTReID/torchreid stack and place the SoccerNet checkpoints above,")
        print("or run a config with prtreid.download_weights=true if the environment supports downloads.")
        raise SystemExit(1)
    print("\nPRTReID environment OK.")


def check_file(path, expected_md5, skip_md5):
    if not path.exists():
        print(f"{path}: MISSING")
        return False
    if not path.is_file():
        print(f"{path}: NOT A FILE")
        return False
    if skip_md5:
        print(f"{path}: OK")
        return True
    actual = md5(path)
    if actual != expected_md5:
        print(f"{path}: BAD MD5 {actual} expected {expected_md5}")
        return False
    print(f"{path}: OK md5={actual}")
    return True


def md5(path):
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
