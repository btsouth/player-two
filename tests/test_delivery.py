from playertwo.delivery import companion_line


def test_status_rephrasing_keeps_observed_numbers_and_does_not_invent_a_change():
    assert companion_line("Frog's at 1 HP, Crono and Marle are full.") == (
        "Frog's at 1 HP; Crono and Marle are at full health.")
    assert companion_line("Frog uses Heal.") == "Frog's using Heal."
    assert companion_line("Won the fight: 648 experience and 18 tech points.") == (
        "We won! You earned 648 experience and 18 tech points.")
    assert companion_line("Crono's level increased!") == "Crono went up a level!"


def test_dialogue_keeps_the_exact_words_with_explicit_attribution():
    assert companion_line("Soldier, Stop right there!") == 'The soldier says, "Stop right there!"'
    assert companion_line('Unknown: Take the left road.') == 'Unknown: Take the left road.'


def test_delivery_is_idempotent():
    for line in ["Frog uses Heal.", "That's the fight. We won.", "Soldier, Stop right there!",
                 "Won the fight: 648 experience and 18 tech points."]:
        once = companion_line(line)
        assert companion_line(once) == once


def test_game_quotation_is_not_rewritten_as_a_status():
    line = 'Cid: "Frog uses Heal."'
    assert companion_line(line) == line


def test_internal_action_filler_can_be_silent_but_real_action_and_danger_stay():
    assert companion_line("No target line this turn, so I'll just go for damage.") == ""
    assert companion_line("Spinning Meteor on the legs.") == "Spinning Meteor on the legs."
    assert companion_line("It's coming at you with speed.") == "It's coming at you with speed."


def test_battle_status_style_does_not_change_dialogue_with_the_same_words():
    from playertwo.delivery import read_line_style
    text = "Crono's level increased!"
    assert read_line_style(text, "battle") == "Crono went up a level!"
    assert read_line_style(text, "dialogue") == text
