from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from openai import OpenAI
from pathlib import Path
import shutil
import os
import ffmpeg

api_key = os.environ["OPENAI_API_KEY"]

app = FastAPI()
openai = OpenAI(api_key=api_key)

SUBTITLES_DIR = Path("subtitles")
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("output")

UPLOAD_DIR.mkdir(exist_ok=True)
SUBTITLES_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_subtitle_section_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)

    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


@app.post("/translate")
async def transcribe(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    subs_path = SUBTITLES_DIR / file.filename
    output_path = OUTPUT_DIR / file.filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_to_transcript = open(file_path, "rb")
    transcription = openai.audio.transcriptions.create(
        file=file_to_transcript,
        model="whisper-1",
        response_format="verbose_json",
        timestamp_granularities=["segment"]
    )

    segments_texts = [f"{i} | {segment.text}" for i,
                      segment in enumerate(transcription.segments)]
    combined_text = "\n".join(segments_texts)
    translation = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a professional translator."},
            {"role": "user", "content": f"Translate this English text into Portuguese, return only the translation without any additional text: {
                combined_text}"},
        ]
    )

    translated_segments = translation.choices[0].message.content.split("\n")
    with open(subs_path.with_suffix(".srt"), "w") as subs:
        for segment, translated_segment in zip(transcription.segments, translated_segments):
            id = segment.id
            start_str = generate_subtitle_section_time(segment.start)
            end_str = generate_subtitle_section_time(segment.end)
            text = translated_segment.split(" | ")[-1]

            subs.write(f"{id + 1}\n{start_str} --> {end_str}\n{text}\n\n")

    ffmpeg.input(file_path).output(filename=output_path, vf=f"subtitles={
        subs_path.with_suffix(".srt")}").run()

    return FileResponse(output_path)
