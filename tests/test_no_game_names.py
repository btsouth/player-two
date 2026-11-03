"""What Player Two does in a Red Dead take has to be what it does in any game with a view."""

import re
from pathlib import Path

SRC = Path(__file__).parent.parent / "src"
NAMES = re.compile(
    r"\b(red[ _-]?dead|rdr2?|undead nightmare|marston|new austin|west elizabeth|nuevo paraiso"
    r"|armadillo|blackwater|macfarlane|bonnie|thieves'? landing|rathskeller|gaptooth"
    r"|tumbleweed|cochinay|seth briars|irish|dickens|bill williamson|javier escuella)\b",
    re.IGNORECASE)


def test_the_catcher_catches():
    assert NAMES.search("if game == 'Red Dead Redemption':")
    assert NAMES.search("RDR")
    assert NAMES.search("John Marston")
    assert not NAMES.search("the wardrobe")  # "rdr" inside a word is no name


def test_no_red_dead_names_in_src():
    hits = [f"{p}:{n}: {line.strip()}"
            for p in sorted(SRC.rglob("*")) if p.suffix in (".py", ".txt", ".json", ".toml", ".md")
            for n, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1)
            if NAMES.search(line)]
    assert not hits, "\n".join(hits)
