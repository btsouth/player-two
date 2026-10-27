"""Bundle the companion window and demo frames into one HTML file for previews.

Usage: python scripts/build_preview.py OUT.html FRAME.jpg [FRAME.jpg ...]
"""

import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image

UI = Path(__file__).resolve().parent.parent / "src/playertwo/ui"


def main() -> None:
    out, frames = Path(sys.argv[1]), []
    for p in sys.argv[2:]:
        im = Image.open(p).convert("RGB")
        im.thumbnail((640, 360))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=78)
        frames.append(base64.b64encode(buf.getvalue()).decode())
    html = (UI / "index.html").read_text()
    html = html.replace('<link rel="stylesheet" href="app.css">',
                        f"<style>{(UI / 'app.css').read_text()}</style>")
    boot = f"window.DEMO_FRAMES={json.dumps(frames)};window.P2_DEMO='loop';"
    html = html.replace('<script src="app.js"></script>',
                        f"<script>{boot}</script><script>{(UI / 'app.js').read_text()}</script>")
    out.write_text(html)


if __name__ == "__main__":
    main()
