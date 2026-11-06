"""Synthesize the narration sentence by sentence with Piper (fr_FR-siwis-medium, downloaded on first use).

Writes out/video/audio/<section>.wav (sentences joined by GAP seconds of silence) and out/video/audio/timings.json:
{section: {"speech": seconds, "sentences": [[start, end, text], ...]}}, times from the start of the speech.
"""
import io
import json
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

from narration import SECTIONS, spoken

GAP = 0.40
VOICE = "fr_FR-siwis-medium"
voices = Path("out/video/voices")
out = Path("out/video/audio")
out.mkdir(parents=True, exist_ok=True)
if not (voices / f"{VOICE}.onnx").exists():
    voices.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "piper.download_voices", VOICE, "--download-dir", str(voices)], check=True)


def say(text):
    raw = subprocess.run([sys.executable, "-m", "piper", "-m", str(voices / f"{VOICE}.onnx"), "-f", "/dev/stdout",
                          "--length-scale", "1.05"],
                         input=text.encode(), check=True, capture_output=True).stdout
    with wave.open(io.BytesIO(raw)) as w:
        rate = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    return rate, pcm


timings = {}
for name, text in SECTIONS:
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    chunks, marks, clock, rate = [], [], 0.0, None
    for s in sentences:
        rate, pcm = say(spoken(s))
        marks.append([round(clock, 3), round(clock + len(pcm) / rate, 3), s])
        chunks += [pcm, np.zeros(int(GAP * rate), np.int16)]
        clock += len(pcm) / rate + GAP
    pcm = np.concatenate(chunks[:-1])
    with wave.open(str(out / f"{name}.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm.tobytes())
    timings[name] = {"speech": len(pcm) / rate, "sentences": marks}
    print(f"{name:11s} {len(pcm) / rate:6.1f} s, {len(sentences)} sentences")
print(f"total {sum(v['speech'] for v in timings.values()):.1f} s")
(out / "timings.json").write_text(json.dumps(timings, indent=1, ensure_ascii=False))
(out / "durations.json").write_text(json.dumps({k: v["speech"] for k, v in timings.items()}, indent=1))
