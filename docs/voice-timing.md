# Transcription timing

Spoken `heard` events include measurements from the real Muse Voice Transcribe stream:

- `transcribe_s`: utterance start to final transcript, including connection and streamed speech.
- `transcribe_audio_s`: seconds of PCM supplied, including pre-roll and ending silence.
- `transcribe_finalize_s`: utterance finish to final transcript. For hands-free, the local VAD has already waited for ending silence before this interval starts.
- `utterance_started_at` / `utterance_ended_at`: hands-free VAD bounds in wall time. Start includes the captured pre-roll; the cutter uses it to keep the spoken wake name.
- `raw_transcript`: the full hands-free transcript before removing the wake name.

These are wall-clock stream measurements, not server compute time or last-word-to-response latency. Last-word-to-first-answer also includes local VAD endpointing, routing and speech generation.

A corresponding `model_stage` event identifies Muse Voice Transcribe explicitly. The video pipeline displays its measured duration as `Stream … s`. Older logs without a stage do not acquire an inferred transcription badge.

An urgent named stop can execute from a partial transcript before the stream finishes. That early `heard` event retains its measured audio start but has no invented completed duration; the completed stage follows when the final transcript arrives. Typed requests carry no transcription duration.
