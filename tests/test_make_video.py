"""The video pipeline's pure parts: windows, gap speed-up, the budget, honest wording."""

import importlib.util
import itertools
import shutil
import sys
from pathlib import Path

import pytest

_path = Path(__file__).parents[1] / "scripts" / "make_video.py"
_spec = importlib.util.spec_from_file_location("make_video", _path)
mv = importlib.util.module_from_spec(_spec)
sys.modules["make_video"] = mv  # dataclasses look their module up
_spec.loader.exec_module(mv)

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


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
    assert "voice-over" in refused and "cold-missing" in refused and "goodbye-missing" in refused and "nav-pending" in refused


def test_the_quote_and_the_honest_lines_are_on_cards():
    parts, _dropped, _warnings = _plan(_events())
    words = " ".join([str(b[0]) for p in parts for b in p.blocks]+[o.text for p in parts for o in p.overlays])
    assert "nor defeat Lavos in Chrono Trigger" in words
    assert "Not beaten yet" in words and "No blind player has tried it yet" in words
    assert mv.GENERAL_LINE in words and mv.RETRO_QUOTE in words
    assert mv.RETRO_QUOTE == "A blind person still cannot ... nor defeat Lavos in Chrono Trigger."
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
    assert "PlayResX: 1920" in head and "Style: P2,Inter,52" in head
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


def test_black_captions_are_large_and_white_for_player_two():
    text=mv.ass_text(_take([_spoken("Fight.",1,2)]),"black")
    assert "Inter,80" in text and "\\1c&HFFFFFF&" in text


def test_goodbye_slot_requires_real_ordered_request_and_spoken_exchange():
    import pytest
    take=_take([{"kind":"heard","t":2,"text":"Goodbye.","typed":True},
                _spoken("Bye for now.",2.1,3),_spoken("Have fun.",3.2,4)],"goodbye",10)
    p=mv.goodbye_piece(take)
    assert p.a==1.6 and p.b==4.3 and p.black_for==999
    take.events=take.events[:-1]
    with pytest.raises(ValueError,match="spoken records"):
        mv.goodbye_piece(take)


def test_red_dead_no_code_claim_requires_a_verified_manifest_flag():
    view,_=mv.rdr_pieces(_take([],"rdr-look",50),_take([],"rdr-dialogue",100))
    assert mv.NO_CODE not in [o.text for o in view.overlays]


@needs_ffmpeg
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


def test_real_spoken_goodbye_keeps_the_request_from_listening_start():
    take=_take([{"kind":"status","t":1,"state":"listening"},
                {"kind":"heard","t":3,"text":"Goodbye."},
                _spoken("Bye for now.",3.2,4),_spoken("Have fun.",4.1,5)],"goodbye",10)
    assert abs(mv.goodbye_piece(take).a-.7)<.001


def test_red_dead_request_and_badge_clear_the_actual_lower_left_hud(tmp_path):
    from PIL import Image
    path=tmp_path/"rdr.png"
    mv.label_png(path,mv.Overlay("You: typed\nWhat do you see?\nMuse Spark / scene\nGeneration 2.24 s",42,"trace-rdr",0,1))
    assert Image.open(path).getchannel("A").getbbox()[3]<780


def test_mh_story_is_one_run_with_false_narration_only_in_muted_uncaptioned_gap():
    import pytest
    take=_take([{"kind":"mh","name":"mh_muse","read":"wheel","seconds":3.94,"t":1},
                _spoken("It's coming at you with technical.",12,14)],"game2",30)
    take.edit={"story":{"normal":[[2,5],[20,23]],"gap_speed":28}}
    parts=mv.mh_story_pieces(take)
    assert all(a.b==b.a for a,b in itertools.pairwise(parts))
    assert len(parts)==3 and parts[1].speed==28 and not parts[1].captions
    assert parts[1].a<=12 and parts[1].b>=14
    assert not any("3.94" in o.text for o in parts[0].overlays)  # compute time is not response latency
    assert not any(mv.NO_CODE in o.text for p in parts for o in p.overlays)
    take.edit["story"]["normal"]=[[2,15],[20,23]]
    with pytest.raises(ValueError,match="False MH line"):
        mv.mh_story_pieces(take)


def test_caption_sheet_does_not_label_intentionally_muted_uncaptioned_speech():
    take=_take([_spoken("omit false line",1,2),_spoken("keep true line",5,6)],"game2",8)
    parts=[mv.Piece("muted","game2",take="game2",a=0,b=3,speed=4,captions=False),
           mv.Piece("normal","game2",take="game2",a=3,b=8)]
    shots=mv.caption_shots(parts,{"game2":take})
    assert shots and all("omit false" not in label for _,label in shots)
    assert any("keep true" in label for _,label in shots)


def test_new_opening_anchor_must_be_an_actual_prior_spoken_line():
    take=_choice_take()
    take.events[0]["text"]="Frog took 493, down to 1."
    take.caps=mv.demo_captions.captions(take.events,0)
    take.edit["opening_line"]="Frog took 493, down to 1."
    parts,warnings=mv.cold_open(take)
    assert not warnings and parts[0].a==.7
    take.edit["opening_line"]="Imagined line."
    _,warnings=mv.cold_open(take)
    assert any("original opening speech record missing" in w for w in warnings)


def test_field_goodbye_uses_real_video_and_preserves_request_lead_in():
    take=_take([{"kind":"heard","t":2,"text":"Goodbye.","typed":True},
                _spoken("Bye for now.",2.1,3),_spoken("Have fun.",3.2,4)],"goodbye",10)
    take.edit["field"]=True
    p=mv.goodbye_piece(take)
    assert p.black_for==0 and p.layout=="fight" and p.a==1.6 and p.b==4.3


def test_win_pays_off_before_identity_and_only_goodbye_is_saved_for_the_end(tmp_path):
    import dataclasses
    take=_choice_take()
    take.events += [_spoken("Crono's level increased!",30.5,31.5),
                    {"kind":"heard","t":32,"text":"Goodbye.","typed":True},
                    _spoken("Bye for now.",32.2,33),_spoken("Have fun.",33.2,34)]
    take.caps=mv.demo_captions.captions(take.events,0)
    bye=dataclasses.replace(take,name="goodbye",edit={"field":True})
    parts,_,_=mv.make_plan({"take-B":take,"goodbye":bye},{},set(),[],{"_folder":str(tmp_path)},{},{})
    identity=next(i for i,p in enumerate(parts) if p.id=='identity')
    assert any(p.take==take.name and p.a<=29<p.b for p in parts[:identity])
    assert not any(p.take==take.name and not p.freeze and p.a<=29<p.b for p in parts[identity:])
    farewell=parts[-2]
    assert farewell.id=='goodbye' and farewell.a>31 and farewell.b>=34 and farewell.black_for==0
    assert parts[-1].secs==5 and parts[identity].freeze
    assert not any('The same fight.' in o.text for p in parts for o in p.overlays)


def test_fair_description_keeps_its_real_reply_after_a_two_second_location_pause():
    take=_take([{"kind":"heard","t":1,"text":"What does it look like?","typed":True},
                _spoken("Leene Square.",2,3),_spoken("A cozy market.",5.1,7),
                _spoken("Striped stalls with fruit.",7.1,10)],"take-A",20)
    cut=mv.cut_line("A3b",take,None,12)
    assert cut.b==10.5


def test_shortened_description_ends_before_the_next_spoken_sentence():
    take=_take([{"kind":"heard","t":1,"text":"What does it look like?","typed":True},
                _spoken("Leene Square.",2,3),_spoken("Striped stalls and fruit.",5.1,8),
                _spoken("An omitted character description.",8,18)],"take-A",25)
    cut=mv.cut_line("A3b",take,None,7)
    assert cut.left_out>0 and cut.b==8


def test_player_two_captions_have_no_repeated_speaker_prefix():
    text=mv.ass_text(_take([_spoken("A real reply.",1,2)]),"map")
    assert "Player Two:" not in text and "A real reply." in text


def test_sam_sequence_requires_logged_geometry_and_retains_unducked_actor_audio(monkeypatch):
    monkeypatch.setattr(mv,"rdr_bench",lambda _: ("Verified bench", "proof.json"))
    import pytest
    take=_take([{"kind":"where","t":3,"via":"sam-3.1","seconds":1.21,"boxes":[[0,.1,.5,.9]]},
                {"kind":"where","t":9,"via":"sam-3.1","seconds":1.08,"boxes":[[.1,.2,.8,.7]]}],"rdr-sam",30)
    take.edit={"view":[0,2],"horse":[2,7],"wagon":[8,12],"dialogue":[13,23],
               "ping_boosts":{"horse":[[6,6.05,3.5]],"wagon":[[11,11.05,3.5]]}}
    ps=mv.rdr_sam_pieces(take)
    mv.sam_boxes(ps,{take.name:take})
    tree=ps[1];boxes=[o for o in tree.overlays if o.where=="box"]
    assert len(boxes)==1 and boxes[0].f0==1 and abs(boxes[0].f1-4.05)<.001
    assert ps[-1].captions is False and ps[-1].duck is False
    take.events[0]['via']='muse-spark'
    with pytest.raises(ValueError,match=r"actual sam-3\.1"):
        mv.rdr_sam_pieces(take)


@needs_ffmpeg
def test_ducked_stems_retain_and_boost_the_last_stereo_ping(tmp_path,monkeypatch):
    import array
    import math
    import subprocess
    import wave
    for key,value in {"W":64,"H":64,"GAME_W":64,"GAME_H":64}.items():
        monkeypatch.setattr(mv,key,value)
    source=tmp_path/"source.mp4"
    subprocess.run(["ffmpeg","-v","error","-f","lavfi","-i","color=s=64x64:r=60:d=2",
                    "-c:v","libx264",str(source)],check=True)
    for stem in ("game","voice"):
        samples=array.array("h")
        for i in range(96000):
            tone=int(4000*math.sin(2*math.pi*1200*i/48000)) if stem=="voice" and 1.90<=i/48000<1.95 else 0
            samples.extend([tone,tone//4])
        with wave.open(str(source.with_suffix(f".{stem}.wav")),"wb") as wav:
            wav.setparams((2,2,48000,0,"NONE","not compressed"));wav.writeframes(samples.tobytes())
    take=mv.Take("t",source,0,2,[],[])
    p=mv.Piece("ping","game3",take="t",a=0,b=2,captions=False,voice_boosts=[[1.895,1.955,3.5]])
    target=tmp_path/"ping.mp4"
    mv.encode_clip(p,{"t":take},{},tmp_path,target,"libx264")
    raw=subprocess.check_output(["ffmpeg","-v","error","-i",str(target),"-vn","-f","f32le","-ac","2","-ar","48000","-"])
    decoded=array.array("f",raw)
    assert len(decoded)>=96000*2
    left=decoded[int(1.91*48000)*2:int(1.94*48000)*2:2]
    right=decoded[int(1.91*48000)*2+1:int(1.94*48000)*2:2]
    assert max(map(abs,left))>.35 and max(map(abs,right))<.16


@needs_ffmpeg
def test_join_removes_per_clip_aac_padding_instead_of_accumulating_it(tmp_path):
    import array
    import subprocess
    source=tmp_path/"short.mp4"
    subprocess.run(["ffmpeg","-v","error","-f","lavfi","-i","color=s=64x64:r=60:d=0.1",
                    "-f","lavfi","-i","sine=frequency=1200:sample_rate=48000:duration=0.1",
                    "-c:v","libx264","-c:a","aac",str(source)],check=True)
    joined=tmp_path/"joined.mp4"
    mv.join_pieces([source]*20,joined,tmp_path)
    raw=subprocess.check_output(["ffmpeg","-v","error","-i",str(joined),"-vn","-f","f32le","-ac","1","-ar","48000","-"])
    samples=array.array("f",raw)
    assert 2<=len(samples)/48000<2.03  # one master AAC tail, never twenty clip tails
    assert abs(mv.probe_s(joined)-2)<.03
    assert max(map(abs,samples[int(1.95*48000):int(1.98*48000)]))>.05


def test_steady_time_summary_is_mathematical_and_guards_reject_tampering():
    ps=[mv.Piece('one','hook',take='t',a=0,b=2),mv.Piece('two','hook',take='t',a=4,b=8,speed=2)]
    mv.steady_timing(ps,'cold')
    assert mv.valid_timing_group(ps,'cold')
    assert all(any(o.text=='8 s shown in 4 s · Cuts + speed-ups' for o in p.overlays) for p in ps)
    assert mv.final_refusals(ps,[])==[]
    ps[1].overlays[0].f1-=.1
    assert any('no 2x label' in x for x in mv.final_refusals(ps,[]))
    mv.steady_timing(ps,'cold');ps[1].b=9
    assert not mv.valid_timing_group(ps,'cold')


def test_recorded_clip_benchmark_is_recomputed_and_absent_proof_drops_the_shot(tmp_path):
    import json
    assert mv.rdr_bench(tmp_path)==('','')
    root=tmp_path/'rdr-bench';root.mkdir()
    (root/'words.json').write_text(json.dumps([[0,10,'actor'],[10,20,'words']]))
    for name,end in [('before',12.7),('after',1.1)]:
        (root/f'{name}.t0.json').write_text(json.dumps({'t0':100,'pos':3.05}))
        (root/f'{name}.events.jsonl').write_text(json.dumps(_spoken('line',100,100+end))+'\n')
    text,proof=mv.rdr_bench(tmp_path)
    assert '12.7 s' in text and '1.1 s' in text and Path(proof).is_file()
    stamp=Path(proof).stat().st_mtime_ns;mv.rdr_bench(tmp_path)
    assert Path(proof).stat().st_mtime_ns==stamp
    (root/'after.events.jsonl').write_text(json.dumps(_spoken('line',100,102))+'\n')
    assert mv.rdr_bench(tmp_path)==('','')


def test_new_founder_and_limit_slots_remain_final_requirements():
    parts=[mv.Piece('identity','hook',kind='card',secs=8.5),mv.Piece('limit','outro',kind='card',secs=6)]
    errors=mv.final_refusals(parts,[])
    assert 'missing voice-over audio: identity' in errors and 'missing voice-over audio: limit' in errors


def test_muted_request_intro_still_counts_as_a_shown_typed_request():
    take=_take([{'kind':'heard','t':1,'text':"Where's it weak?",'typed':True}],'game2',4)
    p=mv.Piece('intro','game2',take='game2',a=0,b=3,speed=4,captions=False,
               overlays=[mv.Overlay("You: typed\nWhere's it weak?",42,'trace-mh',0,.75)])
    assert mv._shown_requests({'game2':take},[p])==[('game2',True)]


def test_full_cut_does_not_fall_back_to_the_old_unproven_silence_claim(tmp_path):
    parts,_,_=mv.make_plan({'rdr-look':_take([],'rdr-look'),
                           'rdr-dialogue':_take([],'rdr-dialogue')},{},set(),[],{'_folder':str(tmp_path)},{},{})
    assert any(p.id=='game3' and p.kind=='slate' for p in parts)
    assert not any('stays quiet' in o.text for p in parts for o in p.overlays)


def test_final_checks_a_wait_cut_summary_even_without_acceleration():
    ps=[mv.Piece('ask','game3',take='t',a=0,b=1),mv.Piece('reply','game3',take='t',a=1.3,b=3)]
    mv.steady_timing(ps,'horse')
    assert mv.final_refusals(ps,[])==[]
    ps[1].a=1.1
    assert 'unsupported time summary: horse' in mv.final_refusals(ps,[])


def test_final_rechecks_benchmark_evidence_on_a_loaded_cut(tmp_path):
    p=mv.Piece('actors','game3',take='t',a=0,b=5,captions=False,proof=str(tmp_path/'rdr-bench/verified.json'),
               overlays=[mv.Overlay('Recorded-clip benchmark\nTalking over actors, same clip: 12.7 s before this fix, 1.1 s after.',42,'bench',0,5)])
    assert 'unsupported benchmark caption: actors' in mv.final_refusals([p],[])


def test_clean_mh_cut_keeps_all_shown_audio_at_normal_speed():
    take=_take([{'kind':'heard','t':1,'text':"Where's it weak?",'typed':True},
                {'kind':'mh','name':'mh_muse','read':'wheel','seconds':4.19,'t':2},
                _spoken('A correct reply.',2.2,3.2)],'game2',30)
    take.edit={'story':{'normal':[[2,5],[20,23]],'cuts_only':True,'carry_requests':True}}
    ps=mv.mh_story_pieces(take)
    assert len(ps)==2 and all(p.speed==1 and p.captions and p.audio_mode=='mix' for p in ps)
    assert not any('muted' in o.text.lower() for p in ps for o in p.overlays)
    assert mv.valid_timing_group(ps,'mh-cuts')


def test_live_acceleration_cannot_speed_a_spoken_line(tmp_path):
    import pytest
    take=_take([_spoken('Keep me at real speed.',.5,1.5)],'t',3)
    p=mv.Piece('fast','ct',take='t',a=0,b=2,speed=5,captions=False,audio_mode='live')
    with pytest.raises(ValueError,match='Live accelerated audio'):
        mv.encode_clip(p,{'t':take},{},tmp_path,tmp_path/'bad.mp4','libx264')


def test_how_bridge_keeps_one_voice_clock_and_requires_real_windows():
    import pytest
    takes={n:_take([],n,100) for n in ('take-A','game2','rdr-sam')}
    takes['rdr-sam'].edit={'bridge':[{'take':n,'a':10,'b':10+d} for n,d in
                                    [('take-A',3),('game2',3),('rdr-sam',3)]]}
    ps=mv.how_bridge(takes,{'how':'voice.wav'},{'how':9})
    assert [p.take for p in ps]==['take-A','game2','rdr-sam']
    assert [p.vo_offset for p in ps]==[0,3,6] and sum(p.out_s for p in ps)==9
    assert all(p.vo_prompt==mv.GENERAL_LINE and p.audio_mode==('mix' if p.take=='game2' else 'game') and not p.captions for p in ps)
    with pytest.raises(ValueError,match='exceeds'):
        mv.how_bridge(takes,{}, {'how':9.01})
    takes['game2'].duration=11
    with pytest.raises(ValueError,match='outside'):
        mv.how_bridge(takes,{}, {})
    takes['game2'].duration=100; takes['game2'].events=[_spoken('Do not hide me.',11,12)]
    with pytest.raises(ValueError,match='must not omit'):
        mv.how_bridge(takes,{}, {})


@needs_ffmpeg
def test_bridge_uses_the_later_voice_samples_instead_of_restarting(tmp_path):
    import subprocess
    import wave

    import numpy as np
    source=tmp_path/'source.mp4';vo=tmp_path/'vo.wav'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=s=64x64:r=60:d=0.5',
                    '-f','lavfi','-i','anullsrc=r=48000:cl=stereo','-t','0.5',
                    '-c:v','libx264','-c:a','aac',str(source)],check=True)
    n=np.arange(24000)/48000
    samples=np.concatenate([np.sin(2*np.pi*440*n),np.sin(2*np.pi*880*n)])*.2
    with wave.open(str(vo),'wb') as f:
        f.setparams((1,2,48000,0,'NONE','not compressed'));f.writeframes((samples*32767).astype('<i2').tobytes())
    take=mv.Take('t',source,0,.5,[],[])
    p=mv.Piece('bridge','games',take='t',a=0,b=.5,captions=False,audio_mode='game',vo=str(vo),vo_offset=.5)
    out=tmp_path/'out.mp4';mv.encode_clip(p,{'t':take},{},tmp_path,out,'libx264')
    raw=subprocess.check_output(['ffmpeg','-v','error','-i',str(out),'-f','f32le','-ac','1','-ar','48000','-'])
    decoded=np.frombuffer(raw,np.float32)[2400:19200];phase=np.arange(len(decoded))/48000
    power=lambda hz:abs(np.mean(decoded*np.exp(-2j*np.pi*hz*phase)))
    assert power(880)>10*power(440)


def test_answer_clock_excludes_compute_and_acknowledgement():
    take=_take([{'kind':'heard','t':1,'text':'Where am I?','typed':True},
                _spoken('Let me look.',1.1,1.7),_spoken('On the world map,',2,3),
                {'kind':'agent','t':1.44,'seconds':.44,'turns':[],'steps':[{'tool':'where_am_i'}],
                 'outcome':'On the world map, near the fair.'}])
    ps=[mv.Piece('p','ct',take='take-A',a=0,b=4)]
    proof=mv.evidence_overlays(ps,{'take-A':take})
    assert proof[0]['answer_seconds']==1 and proof[0]['badge_from']==2
    assert any('Answer 1.0 s' in o.text for o in ps[0].overlays)
    assert not any('0.44' in o.text for o in mv.render_overlays(ps[0]))


def test_black_opening_keeps_its_explanation_clear_and_marks_the_reveal():
    take=_choice_take();ps,_=mv.cold_open(take);mv.steady_timing(ps,'cold')
    assert mv.valid_timing_group(ps,'cold')
    for p in ps:
        if p.black_for==999:
            assert any(o.text=='This is the game with the picture off.' for o in p.overlays)
            assert not any(o.where.startswith('time-') for o in p.overlays)
    assert any(o.text=='Picture on.' for p in ps for o in p.overlays)
    last=ps[-1];last.overlays=[o for o in last.overlays if not o.where.startswith('time-')]
    assert not mv.valid_timing_group(ps,'cold')


def test_context_panel_keeps_request_separate_without_stacking_text():
    p=mv.Piece('p','ct',take='t',a=0,b=5,overlays=[
        mv.Overlay('You: typed\nGo.\nMuse Spark\nAnswer 1.0 s',34,'trace',0,5),
        mv.Overlay('155 s shown in 29 s',34,'time-nav',0,5),
        mv.Overlay('Not scripted.',34,'nav-recovery',1,4)])
    shown=mv.render_overlays(p)
    assert len([o for o in shown if o.where.startswith('trace')])==1
    for at in [.5,2,4.5]:
        assert len([o for o in shown if o.f0<=at<o.f1 and not o.where.startswith('trace')])==1
    assert any('Answer 1.0 s' in o.text and 'Not scripted.' in o.text for o in shown)


def test_purchase_hold_must_be_explicit_and_from_the_same_nearby_result():
    p=mv.Piece('held','ct',line='A4',take='t',a=10,b=12,freeze=True,hold_at=9,
               overlays=[mv.Overlay('Purchase result held',34,'trim',0,2)])
    assert mv.final_refusals([p],[])==[]
    p.hold_at=0
    assert 'unsupported held result: held' in mv.final_refusals([p],[])


def test_muse_latency_does_not_credit_a_local_place_prefix():
    take=_take([{'kind':'heard','t':1,'text':'What does it look like?','typed':True},
                _spoken('Town Square.',1.5,2),_spoken('A busy market.',3,4)])
    take.edit={'scene_prefix':'Town Square.'}
    e={'kind':'say','source':'atmosphere','text':'Town Square. A busy market.'}
    assert mv.event_answer(take,1,e)==(3,2)


def test_menu_claim_requires_muse_turns_and_shows_its_own_keyboard_press():
    es=[]
    for t,text in [(10,'Read this menu.'),(22,'Open the journal.'),(34,'Read it.')]:
        es+=[{'kind':'heard','t':t,'text':text,'typed':True},
             {'kind':'agent','t':t+5,'request':text,'turns':[{'model_s':1}],
              'steps':[{'tool':'press','args':{'buttons':['down','down','down']},'t':2},
                       {'tool':'press','args':{'buttons':['a']},'t':4.5}]}]
    take=_take(es,'rdr-menu',50)
    take.edit={'backend':'keyboard','story':[[10,15],[26,28],[34,40]]}
    assert len(mv.rdr_menu_pieces(take))==3
    with pytest.raises(ValueError,match='edit must show its own journal press'):
        mv.menu_records(take,[mv.Piece('hidden','game3',take='rdr-menu',a=30,b=40)])
    mv.menu_records(take,[mv.Piece('shown','game3',take='rdr-menu',a=26,b=28)])
    take.edit['story']=[[10,15],[27,28],[34,40]]
    with pytest.raises(ValueError,match='show its own journal press'):
        mv.rdr_menu_pieces(take)


def test_menu_context_stays_left_of_the_page_and_above_the_request(tmp_path):
    from PIL import Image
    p=tmp_path/'menu.png'
    mv.label_png(p,mv.Overlay('32 s shown in 12 s · Cuts\nNo code for this game. Muse reads the menu, presses, and reads the page.\nMuse Spark Answer 3.1 s',34,'context-menu',0,5))
    bbox=Image.open(p).getchannel('A').getbbox()
    assert bbox[2]<500 and bbox[3]<520


def test_pacing_blocks_unlisted_short_shots_and_cannot_waive_partial_speech():
    take=_take([_spoken('A complete line.',1,3)],'t',10)
    p=mv.Piece('bad','ct',take='t',a=2,b=3)
    _report,blocked=mv.pacing_report([p],{'t':take},{})
    assert 'shot:bad' in blocked and any(x.startswith('partial:') for x in blocked)
    exceptions={x:'Not a valid waiver.' for x in blocked}
    _,blocked=mv.pacing_report([p],{'t':take},exceptions)
    assert 'shot:bad' in blocked and any(x.startswith('partial:') for x in blocked)


def test_paced_captions_condense_text_without_modifying_source_speech():
    take=_take([_spoken('Head and stomach.',1,3),_spoken('Three parts you can break.',4,6)],'t',10)
    original=list(take.events);take.edit={'paced':True}
    rows=mv.caption_rows(take)
    assert rows[1][2]=='Checking three parts…' and take.events==original


def test_paced_health_summary_waits_for_both_real_health_facts():
    take=_take([_spoken("Frog's at 1 HP, Crono and Marle are full.",1,3),
                _spoken('Crono took 779, down to 1.',3,5)],'t',10)
    take.edit={'paced':True};rows=mv.caption_rows(take)
    assert len(rows)==1 and rows[0][0]==3 and rows[0][2]=='Frog and Crono: 1 HP.'
