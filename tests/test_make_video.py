"""The video pipeline's pure parts: windows, gap speed-up, the budget, honest wording."""

import importlib.util
import itertools
import sys
from pathlib import Path

_path = Path(__file__).parents[1] / "scripts" / "make_video.py"
_spec = importlib.util.spec_from_file_location("make_video", _path)
mv = importlib.util.module_from_spec(_spec)
sys.modules["make_video"] = mv  # dataclasses look their module up
_spec.loader.exec_module(mv)


def _cap(a, b, text="line", who="Player Two"):
    return (a, b, who, text)


def _take(events, name="take-A", duration=100.0, top=False):
    caps = mv.demo_captions.captions(events, 0.0)
    return mv.Take(name, Path(f"{name}.mp4"), 0.0, duration, events, caps, top)


def _spoken(text, began, ended):
    return {"t": ended, "kind": "spoken", "text": text, "began": began, "ended": ended}


def test_speech_stays_at_normal_speed_and_only_long_gaps_are_sped():
    cuts = mv.split_speed(10.0, 40.0, [(10.0, 12.0), (30.0, 33.0)], 2.0)
    assert cuts == [(10.0, 12.3, 1.0), (12.3, 29.7, 2.0), (29.7, 33.3, 1.0), (33.3, 40.0, 2.0)]


def test_a_short_gap_is_not_sped():
    assert mv.split_speed(0.0, 6.0, [(0.0, 2.0), (3.0, 6.0)], 3.0) == [(0.0, 6.0, 1.0)]


def test_pieces_cover_the_window_without_a_hole():
    cuts = mv.split_speed(5.0, 25.0, [(4.0, 6.0), (9.0, 10.0), (20.0, 30.0)], 3.0)
    assert cuts[0][0] == 5.0 and cuts[-1][1] == 25.0
    assert all(x[1] == y[0] for x, y in itertools.pairwise(cuts))


def test_a_request_ends_at_its_last_line_plus_what_follows_on():
    caps = [_cap(61.6, 62.7, "Going"), _cap(64.1, 65.2, "We're at"), _cap(66.5, 69.9, "A bright"),
            _cap(90.0, 91.0, "later")]
    own = mv.request_lines(caps, 60.8, 113.0, 64.0)
    assert [c[3] for c in own] == ["Going", "We're at", "A bright"]


def test_a_fight_window_stops_at_the_readers_first_pause():
    caps = [_cap(21.9, 23.7, "Three imps"), _cap(25.5, 27.3, "Marle took"),
            _cap(28.6, 31.1, "Frog took")]
    assert len(mv.request_lines(caps, 19.9, 35.0, 21.8, chain=0.5)) == 1
    assert len(mv.request_lines(caps, 19.9, 35.0, 21.8, chain=2.0)) == 3


def test_a_line_from_before_the_request_is_not_its_own():
    caps = [_cap(10.0, 19.0, "before"), _cap(21.0, 22.0, "answer")]
    assert [c[3] for c in mv.request_lines(caps, 20.0, 40.0, None)] == ["answer"]


def test_a_long_description_is_cut_after_a_whole_caption():
    own = [_cap(33.0, 36.0), _cap(36.0, 40.0), _cap(40.0, 43.0), _cap(43.0, 47.0)]
    kept, left = mv.talk_cut(own, 8.0)
    assert len(kept) == 2 and left == 7.0


def test_the_budget_drops_b3_then_a5_and_no_more():
    lines = {"A5": 8.0, "B1": 80.0, "B3": 9.0}
    assert mv.fit_budget(lines, 3.0, 80.0) == []
    assert mv.fit_budget(lines, 3.0, 85.0) == ["B3"]
    assert mv.fit_budget(lines, 3.0, 92.0) == ["B3", "A5"]
    assert mv.fit_budget(lines, 3.0, 150.0) == ["B3", "A5"]  # the rest is for the caller


def test_the_budget_also_holds_chrono_trigger_to_its_share():
    assert mv.fit_budget({"B1": 115.0, "B3": 9.0}, 3.0, 10.0) == ["B3"]


def test_the_end_card_says_what_the_log_says():
    assert mv.requests_wording([("CT", False), ("CT", False)], True) == \
        "Requests spoken live. Sped-up parts are marked."
    assert mv.requests_wording([("CT", True)], False) == "Requests typed, not spoken, in this cut."
    assert mv.requests_wording([("Chrono Trigger", False), ("Game", True)], True) == \
        "Requests spoken live in Chrono Trigger, typed in Game. Sped-up parts are marked."
    assert "Some requests" in mv.requests_wording([("CT", False), ("CT", True)], False)


def test_overlays_follow_the_line_across_its_pieces():
    pieces = [mv.Piece("a", "ct", take="t", a=0.0, b=4.0),
              mv.Piece("b", "ct", take="t", a=4.0, b=10.0, speed=3.0),
              mv.Piece("c", "ct", take="t", a=10.0, b=14.0)]
    mv.place_overlays(pieces, [mv.Overlay("x", 36, "bl", 3.0, 8.0)])
    assert [(o.f0, o.f1) for o in pieces[0].overlays] == [(3.0, 4.0)]
    assert [(o.f0, o.f1) for o in pieces[1].overlays] == [(0.0, 2.0)]
    assert [(o.f0, o.f1) for o in pieces[2].overlays] == [(0.0, 2.0)]


def _events():
    return [{"t": 5.0, "kind": "heard", "text": "Buy three potions.", "typed": True},
            _spoken("Okay, I'll buy three potions.", 5.1, 6.5),
            {"t": 8.0, "kind": "say", "source": "agent", "text": "Going to the item seller."},
            _spoken("Going to the item seller.", 8.0, 9.5),
            _spoken("Got three potions for 30 gold.", 30.0, 32.0),
            {"t": 30.0, "kind": "agent", "outcome": "Got three", "steps": [], "turns": []},
            {"t": 60.0, "kind": "heard", "text": "Buy three potions.", "typed": True},
            _spoken("Okay, I'll buy three potions.", 60.1, 61.5),
            {"t": 70.0, "kind": "agent", "outcome": "", "steps": [], "turns": []},
            _spoken("Bought.", 70.0, 71.0)]


def test_cut_finds_the_line_and_the_pick_chooses_the_attempt():
    take = _take(_events())
    last = mv.cut_line("A4", take, None, None)
    first = mv.cut_line("A4", take, 1, None)
    assert last.heard == 60.0 and last.b == 71.5
    assert first.heard == 5.0 and first.a == 4.6 and first.b == 32.5
    assert first.typed and first.first_word == 5.1


def test_a_spoken_request_starts_where_the_player_started_talking():
    events = [{"t": 3.0, "kind": "status", "state": "listening"},
              {"t": 5.0, "kind": "heard", "text": "Buy three potions."},
              _spoken("Okay.", 5.1, 6.0), {"t": 6.0, "kind": "agent"}]
    cut = mv.cut_line("A4", _take(events), None, None)
    assert abs(cut.a - 2.7) < 0.001 and not cut.typed


def test_the_walk_is_sped_with_a_label_and_the_lines_are_not():
    take = _take(_events())
    cut = mv.cut_line("A4", take, 1, None)
    pieces = mv.line_pieces(cut, take, 2.0)
    sped = [p for p in pieces if p.speed > 1]
    assert len(sped) == 1 and sped[0].overlays[0].text == "2x"
    assert not any(p.overlays for p in pieces if p.speed == 1)
    assert mv.caption_problems(take.caps, mv.speech_spans(take.events, 0.0), pieces) == []


def test_caption_check_finds_overlap_stale_and_a_caption_with_no_speech():
    caps = [_cap(0.0, 2.0, "Everyone's at full health."), _cap(1.5, 3.0, "Marle took 429, down to 1."),
            _cap(10.0, 11.0, "nobody said this")]
    pieces = [mv.Piece("p", "ct", take="t", a=0.0, b=20.0)]
    found = " | ".join(w for _t, w in mv.caption_problems(caps, [(0.0, 3.0)], pieces))
    assert "overlap" in found and "stale" in found and "no speech" in found


def test_caption_check_finds_speech_sped_up_and_outliving_captions():
    caps = [_cap(1.0, 3.0, "said fast"), _cap(10.0, 14.0, "up too long")]
    pieces = [mv.Piece("p", "ct", take="t", a=0.0, b=5.0, speed=2.0),
              mv.Piece("q", "ct", take="t", a=5.0, b=20.0)]
    found = " | ".join(w for _t, w in mv.caption_problems(caps, [(1.0, 3.0), (10.0, 11.0)], pieces))
    assert "sped up 2x" in found and "outlives" in found


def test_old_logs_put_a_narration_after_its_request():
    events = [{"t": 31.3, "kind": "heard", "text": "What's on the screen?", "typed": True},
              {"t": 33.3, "kind": "status", "state": "speaking"},
              {"t": 33.4, "kind": "say", "source": "describe", "text": "A sunny coast. " * 10},
              {"t": 46.9, "kind": "status", "state": "watching"}]
    fixed = mv.demo_captions.captions(mv._old_narrations(events), 0.0)
    assert min(c[0] for c in fixed if c[2] == "Player Two") >= 33.0
    # A log with the voice's record is left alone.
    played = [*events, _spoken("x", 34.0, 35.0)]
    assert mv._old_narrations(played) is played


def test_a_game_gets_the_no_code_line_only_when_flagged():
    warned = []
    plain = mv.game_pieces("game2", None, False, None, warned)
    flagged = mv.game_pieces("game2", None, True, None, warned)
    assert plain[0].kind == "slate" and not plain[0].overlays
    assert flagged[0].overlays[0].text == mv.NO_CODE


def test_a_game_clip_keeps_to_its_budget_at_normal_speed():
    events = [{"t": 2.0, "kind": "heard", "text": "What's on the screen?", "typed": True},
              *[_spoken(f"Line {i}.", 3.0 + 4 * i, 6.0 + 4 * i) for i in range(12)]]
    warned = []
    pieces = mv.game_pieces("game3", _take(events, "game3"), False, None, warned)
    assert pieces[0].speed == 1.0 and pieces[0].out_s <= 15.0 and not warned


def _plan(a_events, final_rates=None, no_code=frozenset()):
    takes = {"take-A": _take(a_events)}
    return mv.make_plan(takes, {}, set(no_code), final_rates or mv.DRAFT_RATES, {}, {}, {})


def test_final_refuses_placeholders_voice_over_labels_and_draft_rates():
    parts, _dropped, warnings = _plan(_events())
    refused = " | ".join(mv.final_refusals(parts, warnings))
    assert "placeholder: game2" in refused and "placeholder: game3" in refused
    assert "voice-over" in refused and "win-missing" in refused and "goodbye-missing" in refused


def test_the_quote_and_the_honest_lines_are_on_cards():
    parts, _dropped, _warnings = _plan(_events())
    words = " ".join(str(b[0]) for p in parts for b in p.blocks)
    assert "nor defeat Lavos in Chrono Trigger" in words
    assert "Lavos is still ahead" in words and "No blind player has tried it yet" in words
    assert "Muse understands scenes" in words and "common actions" in words
    assert "works out what you want" not in " ".join(
        o.text for p in parts for o in p.overlays)


def test_the_end_card_wording_comes_from_the_shown_requests():
    parts, _dropped, _warnings = _plan(_events())
    assert parts[-1].id == "c-end"
    assert any("typed" in b[0] for b in parts[-1].blocks)


def test_final_refuses_a_speed_up_without_its_label():
    good = mv.Piece("g", "ct", take="t", a=0.0, b=6.0, speed=2.0,
                    overlays=[mv.Overlay("2x", 48, "tr", 0.0, 3.0)])
    bad = mv.Piece("b", "ct", take="t", a=6.0, b=12.0, speed=2.0)
    assert mv.final_refusals([good], []) == []
    assert mv.final_refusals([bad], []) == ["no 2x label all through b"]


def test_captions_at_the_top_clear_the_games_message_bar():
    take = _take([{"t": 1.0, "kind": "heard", "text": "hi", "typed": True}], "take-B", top=True)
    head = mv.ass_text(take)
    assert "PlayResX: 1920" in head and "Style: P2,Inter,48" in head
    assert "DejaVu" not in head


def test_any_speed_gets_a_tempo_filter_ffmpeg_can_take():
    assert mv._tempo(1.0) == ""
    assert mv._tempo(2.0) == "atempo=2,"
    assert mv._tempo(3.0) == "atempo=2.0,atempo=1.5,"
    assert mv._tempo(4.0) == "atempo=2.0,atempo=2,"


def test_a_caption_moment_is_where_it_lands_in_the_video():
    parts = [mv.Piece("c", "hook", kind="card", secs=10.0, blocks=[("x", 10, False)]),
             mv.Piece("p", "ct", take="t", a=100.0, b=106.0, speed=3.0),
             mv.Piece("q", "ct", take="t", a=106.0, b=110.0)]
    assert mv.out_time(parts, "t", 103.0) == 11.0
    assert mv.out_time(parts, "t", 107.0) == 13.0
    assert mv.out_time(parts, "t", 50.0) is None


def test_evidence_never_guesses_a_model_from_request_words():
    events = [{"t": 1, "kind": "heard", "text": "segment with SAM", "typed": True},
              {"t": 2, "kind": "agent", "turns": [], "seconds": 0.1, "steps": [{"tool": "shop"}]}]
    parts = [mv.Piece("p", "ct", take="take-A", a=0, b=4)]
    report = mv.evidence_overlays(parts, {"take-A": _take(events)})
    assert report[0]["layer"] == "Local action"
    assert "SAM" not in report[0]["layer"]
    assert report[0]["badge_from"] == 2
    assert parts[0].overlays[0].f1 == 2  # no completed measurement before it completed


def test_spark_badge_requires_logged_model_identity_and_measured_turns():
    es = [{"t": 0, "kind": "ready", "brain": "muse-spark-1.3"},
          {"t": 1, "kind": "heard", "text": "Where?", "typed": True},
          {"t": 3, "kind": "agent", "seconds": 2, "turns": [{"model_s": 0.8}]}]
    parts = [mv.Piece("p", "ct", take="take-A", a=0, b=5)]
    result = mv.evidence_overlays(parts, {"take-A": _take(es)})
    assert result[0]["layer"] == "Muse Spark" and "0.80 s" in result[0]["timing"]
    no_identity = mv.evidence_overlays([mv.Piece("p", "ct", take="take-A", a=0, b=5)],
                                {"take-A": _take(es[1:])})
    assert no_identity[0]["layer"] == "Player Two"


def test_a_three_minute_overrun_is_refused():
    assert "over 180" in " ".join(mv.final_refusals(
        [mv.Piece("long", "hook", kind="card", secs=180.01)], []))


def test_empty_turns_and_no_steps_do_not_prove_a_local_answer():
    es = [{"t": 1, "kind": "heard", "text": "What are we fighting?"},
          {"t": 3, "kind": "agent", "seconds": 2, "turns": [], "steps": []}]
    result = mv.evidence_overlays([mv.Piece("p", "ct", take="take-A", a=0,b=5)],
                                  {"take-A": _take(es)})
    assert result[0]["layer"] == "Player Two"


def test_missing_fight_gets_a_visible_placeholder_not_a_silent_omission():
    parts, _, _ = _plan(_events())
    assert any(p.id == "cold-missing" and p.kind == "slate" for p in parts)
    assert "placeholder: cold-missing" in mv.final_refusals(parts, [])


def _choice_take():
    es=[_spoken("Fight.",1,1.6),
        {"t":3,"kind":"heard","text":"How are we doing?","typed":True},
        _spoken("Frog is at 1 HP.",3.2,5),
        {"t":8,"kind":"heard","text":"Crono, use Cyclone.","typed":True},
        _spoken("Okay, Crono will use Cyclone.",8.1,10),
        _spoken("Crono used Cyclone.",12,13),
        {"t":16,"kind":"heard","text":"Win this fight for me.","typed":True},
        _spoken("Okay, I'll take this fight.",16.1,18),
        _spoken("Frog uses Heal.",23,24),
        _spoken("Won the fight.",29,30),
        {"t":30,"kind":"agent","seconds":14,"turns":[],"steps":[]}]
    take=_take(es,"take-B",35,True)
    take.edit={"reveal_at":11,"reveal_duration":.25,"normal_until":13.5}
    return take


def test_cold_open_keeps_all_speech_and_time_continuous_through_measured_spin():
    take=_choice_take()
    parts,warnings=mv.cold_open(take)
    assert not warnings and parts[0].speed==1
    assert parts[0].a==.7
    assert any(p.a==11 and p.b==13.5 and p.speed==1 for p in parts)
    assert all(p.black_for==999 for p in parts if p.b<=11)
    assert next(p for p in parts if p.a==11).reveal_s==.25
    assert all(abs(a.b-b.a)<.001 for a,b in itertools.pairwise(parts))
    mv.evidence_overlays(parts,{"take-B":take})
    words=" ".join(o.text for p in parts for o in p.overlays)
    assert all(x in words for x in ["How are we doing?","Crono, use Cyclone.","Win this fight for me."])
    for a,b,who,_ in take.caps:
        if who=="Player Two":
            assert not any(p.speed>1 and min(p.b,b)-max(p.a,a)>.01 for p in parts)


def test_real_win_is_reserved_once_for_the_black_ending_without_repeating_speech():
    take=_choice_take()
    parts,_=mv.cold_open(take)
    opening,ending=mv.reserve_win_line(parts,take)
    assert len(ending)==1 and ending[0].black_for==999
    assert abs(opening[-1].b-ending[0].a)<.001
    assert opening[-1].b<29 and ending[0].a<29<ending[0].b
    assert ending[0].b>=30


def test_sam_overlay_requires_actual_sam_box_not_a_centre_or_fallback():
    es=[{"t":2,"kind":"where","via":"muse-spark","boxes":[[.1,.2,.3,.4]]},
        {"t":3,"kind":"where","via":"sam-3.1","points":[[.5,.5]]}]
    take=_take(es)
    p=mv.Piece("p","ct",take="take-A",a=0,b=5)
    mv.sam_boxes([p],{"take-A":take})
    assert p.overlays==[]
    take.events.append({"t":4,"kind":"where","via":"sam-3.1","boxes":[[.1,.2,.3,.4]]})
    mv.sam_boxes([p],{"take-A":take})
    assert len(p.overlays)==1 and p.overlays[0].where=="box"
    assert p.overlays[0].f0==4


def test_cold_open_requires_actual_spin_measurement_and_request_order():
    take=_choice_take()
    take.edit={}
    parts,warnings=mv.cold_open(take)
    assert any("actual Cyclone frames" in w for w in warnings)
    assert any("cold open incomplete" in w for w in mv.final_refusals(parts,warnings))
    take=_choice_take()
    next(e for e in take.events if e.get("text")=="Crono, use Cyclone.")["t"]=19
    _,warnings=mv.cold_open(take)
    assert any("in that order" in w for w in warnings)


def test_invalid_or_inverted_sam_boxes_are_not_drawn():
    assert not mv.valid_box([.4,.2,.1,.6])
    assert not mv.valid_box([.1,.2,1.2,.6])
    assert not mv.valid_box([.5,.5])


def test_missing_cold_sequence_keeps_its_placeholder_even_if_a_win_is_logged():
    take=_choice_take()
    take.events=[e for e in take.events if e.get("text")!="How are we doing?"]
    opening,_=mv.cold_open(take)
    kept,ending=mv.reserve_win_line(opening,take)
    assert kept==opening and kept[0].kind=="slate" and ending==[]


def test_measured_recorder_delay_aligns_captions_without_rewriting_source(tmp_path,monkeypatch):
    import json
    (tmp_path/"take-A.mp4").write_bytes(b"test")
    (tmp_path/"take-A.mp4.start").write_text("1000")
    es=[{"kind":"spoken","t":1002,"began":1001,"ended":1002,"text":"Fight."}]
    (tmp_path/"take-A.events.jsonl").write_text(json.dumps(es[0])+"\n")
    (tmp_path/"takes.json").write_text(json.dumps({"take-A":{"edit":{"recording_delay":.32}}}))
    monkeypatch.setattr(mv,"probe_s",lambda _:10)
    take=mv.load_take(tmp_path,"take-A",False)
    assert abs(take.caps[0][0]-.68)<.001
    assert take.events==es and (tmp_path/"take-A.mp4.start").read_text()=="1000"


def test_speed_label_stays_clear_of_the_upper_request_chip(tmp_path):
    from PIL import Image
    path=tmp_path/"speed.png"
    mv.label_png(path,mv.Overlay("2x",48,"tr",0,1))
    bbox=Image.open(path).getchannel("A").getbbox()
    assert bbox[0]>mv.W*.7  # caller's legacy top-right slot must still be physically right
    mv.label_png(path,mv.Overlay("2x",48,"speed-shop",0,1))
    assert Image.open(path).getchannel("A").getbbox()[1]>mv.H*.6


def test_shop_completion_is_not_cut_before_its_measured_result_can_be_read():
    es=[{"t":1,"kind":"heard","text":"Buy three potions.","typed":True},
        _spoken("Bought. You have 370 gold left.",3,5),
        {"t":6.2,"kind":"agent","request":"Buy three potions.","turns":[],"steps":[],"seconds":5.2}]
    cut=mv.cut_line("A4",_take(es),None,None)
    assert cut.b>=7.4-.001


def test_shop_scene_repositioning_preserves_continuous_source_and_speed_labels():
    take=_take([])
    take.edit={"shop_menu":[3,7]}
    p=mv.Piece("shop","ct",line="A4",take="take-A",a=0,b=10,speed=2,
               overlays=[mv.Overlay("2x",48,"tr",0,5),
                         mv.Overlay("You: typed\nBuy three potions.",42,"trace",0,5)])
    result=mv.shop_scene_layouts([p],take)
    assert len(result)==1 and (p.a,p.b,p.speed)==(0,10,2)
    assert p.overlays[0].text=="2x" and (p.overlays[0].f0,p.overlays[0].f1)==(0,5)
    assert [o.where for o in p.overlays[1:]]==["trace-shop","trace-shop-menu","trace-shop"]
    assert [(o.f0,o.f1) for o in p.overlays[1:]]==[(0,1.5),(1.5,3.5),(3.5,5)]


def test_tightened_four_times_gaps_never_remove_or_speed_original_speech():
    take=_choice_take()
    take.edit.update(gap_speed=4,gap_max_output=.45,min_gap=.7,speech_pad=.12)
    parts,warnings=mv.cold_open(take)
    assert not warnings
    assert any(p.speed==4 for p in parts)
    assert any(o.text=="Silent gap shortened" for p in parts for o in p.overlays)
    for e in take.events:
        if e.get("kind")!="spoken":
            continue
        covered=sum(max(0,min(p.b,e["ended"])-max(p.a,e["began"])) for p in parts if p.speed==1)
        assert abs(covered-(e["ended"]-e["began"]))<.001
    for p in parts:
        assert not p.speed>1 or p.out_s<=.45+.001
    assert all(p.a>=a.b for a,p in itertools.pairwise(parts))


def test_red_dead_silent_window_uses_game_audio_without_ducking_or_fake_captions():
    look=_take([{"kind":"heard","t":19,"text":"What do you see?","typed":True}],"rdr-look",50)
    look.edit["notes_only_verified"]=True
    quiet=_take([],"rdr-dialogue",100)
    view,dialogue=mv.rdr_pieces(look,quiet)
    assert mv.NO_CODE in [o.text for o in view.overlays]
    assert dialogue.speed==1 and not dialogue.duck and not dialogue.captions
    assert dialogue.out_s==11
    quiet.events.append(_spoken("An interruption.",25,26))
    import pytest
    with pytest.raises(ValueError,match="contains logged"):
        mv.rdr_pieces(look,quiet)


def test_end_card_contains_full_notice_and_goodbye_stays_a_placeholder_until_recorded(tmp_path):
    parts,_,_=mv.make_plan({}, {}, set(), [], {"_folder":str(tmp_path)}, {}, {})
    assert any(p.id=="limit" and p.blocks[0][0]==mv.LIMIT_LINE for p in parts)
    assert any(p.id=="goodbye-missing" and p.kind=="slate" for p in parts)
    end=parts[-1]
    assert mv.SE_NOTICE in [b[0] for b in end.blocks]
    assert mv.LEGAL_LINE in [b[0] for b in end.blocks]
    mv.card_png(tmp_path/"end.png",end.blocks,end=True)


def test_black_captions_are_large_and_use_a_distinct_player_two_colour():
    text=mv.ass_text(_take([_spoken("Fight.",1,2)]),"black")
    assert "Inter,80" in text and "\\1c&HD7EFFF&" in text


def test_goodbye_slot_requires_real_ordered_request_and_spoken_exchange():
    import pytest
    take=_take([{"kind":"heard","t":2,"text":"Goodbye.","typed":True},
                _spoken("Bye for now.",2.1,3),_spoken("Have fun.",3.2,4)],"goodbye",10)
    p=mv.goodbye_piece(take)
    assert p.a==1.7 and p.b==4.3 and p.black_for==999
    take.events=take.events[:-1]
    with pytest.raises(ValueError,match="spoken records"):
        mv.goodbye_piece(take)


def test_red_dead_no_code_claim_requires_a_verified_manifest_flag():
    view,_=mv.rdr_pieces(_take([],"rdr-look",50),_take([],"rdr-dialogue",100))
    assert mv.NO_CODE not in [o.text for o in view.overlays]


def test_a_short_four_times_gap_encodes_with_a_real_silent_audio_track(tmp_path):
    import subprocess
    source=tmp_path/"source.mp4"
    subprocess.run(["ffmpeg","-v","error","-f","lavfi","-i","color=s=64x64:r=60:d=1",
                    "-f","lavfi","-i","anullsrc=r=48000:cl=stereo","-t","1",
                    "-c:v","libx264","-c:a","aac",str(source)],check=True)
    take=mv.Take("t",source,0,1,[],[])
    p=mv.Piece("gap","hook",take="t",a=.2,b=.8,speed=4,captions=False)
    mv.encode_clip(p,{"t":take},{},tmp_path,tmp_path/"gap.mp4","libx264")
    import json
    streams=json.loads(subprocess.check_output(["ffprobe","-v","error","-show_streams",
                                               "-of","json",str(tmp_path/"gap.mp4")]))["streams"]
    assert any(s["codec_type"]=="audio" and int(s["nb_frames"])>0 for s in streams)
    assert abs(mv.probe_s(tmp_path/"gap.mp4")-.15)<.06
