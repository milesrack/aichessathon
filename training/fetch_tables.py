"""Download the complete three- and four-piece Syzygy set from Lichess."""

import hashlib
import json
import re
import urllib.request
from pathlib import Path


def main() -> None:
    root = Path("weights/syzygy")
    root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for suffix in ("wdl", "dtz"):
        base = f"https://tablebase.lichess.ovh/tables/standard/3-4-5-{suffix}/"
        with urllib.request.urlopen(base, timeout=60) as response:
            index = response.read().decode()
        for name in re.findall(r'href="([KQRBNPv]+\.rtb[wz])"', index):
            if len(name.split(".")[0].replace("v", "")) > 4:
                continue
            path = root / name
            if not path.exists():
                with urllib.request.urlopen(base + name, timeout=60) as response:
                    path.write_bytes(response.read())
            manifest.append(
                {
                    "name": name,
                    "url": base + name,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    # Provenance is development documentation, outside the submitted weights.
    Path("docs/tablebase-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
