def test_wide_pan_spreads_for_the_ear_and_keeps_the_side():
    from playertwo.locate import wide_pan

    assert wide_pan(0.29) == 0.54
    assert wide_pan(-0.29) == -0.54
    assert wide_pan(0) == 0
    assert wide_pan(1) == 1
