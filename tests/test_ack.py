from playertwo.ack import ack


def test_ack_lines():
    cases = {
        "What are we fighting?": None,
        "Where am I": None,
        "Win this fight for me": "I'll take this fight.",
        "Run away": "I'll try to get us away.",
        "Take me to Leene Square": None,
        "let's go to 600 AD": None,
        "Heal Crono": "I'll heal Crono.",
        "Use Athenian Water on Crono": "I'll use Athenian Water on Crono.",
        "Crono, use Cyclone": "Crono will use Cyclone.",
        "Attack the one on the left": "I'll attack the one on the left.",
        "please heal my party": "I'll heal your party.",
        "Cross the chain bridge to the left": "I'll cross the chain bridge to the left.",
        "do a little dance": None,
        "": None,
    }
    for req, want in cases.items():
        assert ack(req) == want, (req, ack(req))
