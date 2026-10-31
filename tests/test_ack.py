from playertwo.ack import ack


def test_ack_lines():
    cases = {
        "What are we fighting?": "Let me look.",
        "Where am I": "Let me look.",
        "Win this fight for me": "Okay, I'll take this fight.",
        "Run away": "Okay, trying to run.",
        "Take me to Leene Square": None,
        "let's go to 600 AD": None,
        "Heal Crono": "Okay, I'll heal Crono.",
        "Use Athenian Water on Crono": "Okay, I'll use Athenian Water on Crono.",
        "Crono, use Cyclone": "Okay, Crono will use Cyclone.",
        "Attack the one on the left": "Okay, I'll attack the one on the left.",
        "please heal my party": "Okay, I'll heal your party.",
        "Cross the chain bridge to the left": "Okay, I'll cross the chain bridge to the left.",
        "do a little dance": "On it.",
        "": None,
    }
    for req, want in cases.items():
        assert ack(req) == want, (req, ack(req))
