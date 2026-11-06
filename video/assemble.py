"""Assemble the final video: every section's picture with its narration (delayed by its lead), concatenated,
loudness-normalized, with French subtitles from the sentence timings as a soft track (and an .srt beside it).

Reads the narration (tts.py), the scenes (scenes.py) and the refraction clip (refraction_clip.py) under out/video;
writes out/video/ondes_dans_un_milieu.mp4 and .srt.
"""
import json
import subprocess
from pathlib import Path

from narration import timing

ORDER = [("intro", "Intro"), ("model", "Model"), ("sum", "Sum"), ("sheet", "OneSheet"), ("slab", "Slab"),
         ("spacetime", "Spacetime"), ("front", "Front"), ("colors", "Colors"), ("slowlight", "SlowLight"),
         ("refraction", None), ("summary", "Summary"), ("credits", "Credits")]
OUT = Path("out/video")
TIMINGS = json.loads((OUT / "audio/timings.json").read_text())
SCHEDULE = timing({k: v["speech"] for k, v in TIMINGS.items()})
build = OUT / "build"
build.mkdir(parents=True, exist_ok=True)


def duration(path):
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                 str(path)], capture_output=True, text=True, check=True).stdout)


def srt_time(t):
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h:02d}:{m:02d}:{int(s):02d},{int(round((s % 1) * 1000)) % 1000:03d}"


segments, cues, clock = [], [], 0.0
for name, scene in ORDER:
    video = OUT / ("clips/refraction.mp4" if scene is None else f"media/videos/scenes/1080p30/{scene}.mp4")
    lead = SCHEDULE[name][0]
    dur = duration(video)
    seg = build / f"{len(segments):02d}_{name}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-i", str(OUT / f"audio/{name}.wav"),
                    "-filter_complex", f"[1:a]adelay={int(lead * 1000)}:all=1,apad,aformat=sample_rates=48000:"
                    f"channel_layouts=stereo[a]", "-map", "0:v", "-map", "[a]", "-t", f"{dur:.3f}",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "30",
                    "-c:a", "aac", "-b:a", "192k", str(seg)], check=True)
    for start, end, text in TIMINGS[name]["sentences"]:
        cues.append((clock + lead + start, clock + lead + end, text))
    segments.append(seg)
    clock += duration(seg)
    print(f"{name:11s} {dur:6.2f} s  -> ends at {clock:7.2f} s")

(build / "list.txt").write_text("".join(f"file '{s.name}'\n" for s in segments))
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(build / "list.txt"),
                "-c", "copy", str(build / "joined.mp4")], check=True)
srt = "".join(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{text}\n\n" for i, (a, b, text) in enumerate(cues, 1))
srt_path, video_path = OUT / "ondes_dans_un_milieu.srt", OUT / "ondes_dans_un_milieu.mp4"
srt_path.write_text(srt)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(build / "joined.mp4"), "-i", str(srt_path),
                "-map", "0:v", "-map", "0:a", "-map", "1:s", "-c:v", "copy",
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                "-c:s", "mov_text", "-metadata:s:s:0", "language=fre", "-metadata:s:a:0", "language=fre",
                "-metadata", "title=Les ondes dans un milieu — personne ne ralentit", "-metadata", "artist=Claude",
                "-metadata", "comment=Écrit et réalisé par Claude (Anthropic). Simulation : Mario Geiger. "
                "Idée : Quentin Talon.", "-movflags", "+faststart", str(video_path)], check=True)
print(f"total {duration(video_path):.2f} s, {len(cues)} subtitles")
