import tempfile
import os
try:
    from faster_whisper import WhisperModel
    FWHISPER_AVAILABLE = True
except Exception:
    FWHISPER_AVAILABLE = False

def transcribe_audio_file(file_bytes: bytes, lang_hint=None) -> str:
    if not FWHISPER_AVAILABLE:
        return "[Audio transcription unavailable: please install faster-whisper]"
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(tmp_path, language=lang_hint)
    texts = [seg.text for seg in segments]
    try:
        os.remove(tmp_path)
    except Exception:
        pass
    return " ".join(texts).strip()
