from gtts import gTTS
from openai import OpenAI
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import moviepy as mp
import numpy as np
import os
import gc
import uuid
import shutil
import tempfile

app = FastAPI()

client = OpenAI(api_key="") # Need to replace

os.makedirs("static", exist_ok=True)

def extract_audio(video_path: str, audio_path: str):
    """Extract audio from video and save as WAV."""
    try:
        video = mp.VideoFileClip(video_path)
        if not video.audio:
            video.close()
            raise ValueError("Video has no audio")
        video.audio.write_audiofile(audio_path, codec='aac', bitrate='128k')
        video.close()
        gc.collect()
    except Exception as e:
        raise Exception(f"Audio extraction failed: {e}")

def transcribe_audio(audio_path: str, return_dict: bool = False):
    """Transcribe audio to text using Whisper."""
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json"
            )
        if return_dict:
            return transcript
        else:
            return transcript.text
    except Exception as e:
        raise Exception(f"Whisper API error: {e}")

def translate_to_tamil(text: str) -> str:
    """Translate text to Tamil using OpenAI API with news-style tone."""
    try:
        if not text.strip():
            raise ValueError("No text to translate")
        prompt = (
            "You are a professional Tamil news anchor. Translate the following text into Tamil, "
            "ensuring natural, formal, and human-like sentence structure suitable for a news broadcast. "
            "Maintain a polite and engaging tone while avoiding any unverified information. "
            "Correct any grammatical or syntactical errors in the translated text. \n\n"
            f"{text}"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a Tamil news translation expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500, # token limit
            temperature=0.5 # creativity
        )
        tamil_text = response.choices[0].message.content.strip()
        return tamil_text
    except Exception as e:
        raise Exception(f"OpenAI API error: {e}")

def generate_news_script(tamil_text: str, output_path: str) -> str:
    """Generate Tamil news script."""
    try:
        news_script = (
            f"{tamil_text}\n\n"
        ) # If needed opening and closing text will modify but will affect the timeline
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(news_script)
        return news_script
    except Exception as e:
        raise Exception(f"Error generating news script: {e}")

def generate_tamil_audio(text: str, output_path: str):
    """Generate Tamil audio from text using gTTS."""
    try:
        tts = gTTS(text=text, lang='ta')
        tts.save(output_path)
    except Exception as e:
        raise Exception(f"gTTS error: {e}")

def replace_audio(video_path: str, tamil_audio_path: str, output_path: str):
    """Replace video audio with Tamil audio."""
    try:
        video = mp.video.io.VideoFileClip.VideoFileClip(video_path)
        tamil_audio = mp.audio.io.AudioFileClip.AudioFileClip(tamil_audio_path)
        
        video.audio = tamil_audio
        
        video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            bitrate="5000k"
        )
        
        video.close()
        gc.collect()
        if hasattr(tamil_audio, 'close'):
            tamil_audio.close()
    except Exception as e:
        raise Exception(f"MoviePy error: {e}")
    
# WIP
"""def add_subtitle(video_path: str, subtitle_video_path: str, transcript, font_path: str):
    clip = mp.video.io.VideoFileClip.VideoFileClip(video_path)
    subs = []

    for segment in transcript.segments:
        text = segment.text.strip()
        start = segment.start
        end = segment.end
        duration = end - start

        subtitle = (mp.video.VideoClip.TextClip(
                        font=font_path,
                        text=text,
                        font_size=int(36),
                        color='white',
                        stroke_color='black',
                        stroke_width=int(1)
                        ).get_frame(0)
        )
        
        sub_clip = (mp.video.VideoClip.ImageClip(subtitle)
                        .set_start(start)
                        .set_duration(duration)
                        .set_position(("center", "bottom"))
        )

        subs.append(sub_clip)

    final = mp.video.compositing.CompositeVideoClip.CompositeVideoClip([clip, *subs])
    final.write_videofile(subtitle_video_path, codec="libx264", audio_codec="aac")

    clip.close()
    final.close()
    gc.collect()

    print(f"Subtitled video saved at {subtitle_video_path}")"""
    
@app.post("/translate-video")
async def translate_video(file: UploadFile = File(...)):
    """
    Upload a video, translate its audio to Tamil with a news-style tone,
    and return the translated video and news script.
    """
    # Validation
    if not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only MP4 files are supported")

    # Temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            video_path = os.path.join(temp_dir, "input_video.mp4")
            with open(video_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            # Define paths for temporary and output files
            audio_path = os.path.join(temp_dir, "temp_audio.m4a")
            tamil_audio_path = os.path.join(temp_dir, "temp_tamil_audio.mp3")
            output_video_path = os.path.join(temp_dir, "output_tamil_video.mp4")
            subtitled_video_path = os.path.join(temp_dir, "output_with_subtitles.mp4")
            news_script_path = os.path.join(temp_dir, "news_script_tamil.txt")

            print("Extracting audio...")
            extract_audio(video_path, audio_path)

            print("Transcribing audio...")
            original_text = transcribe_audio(audio_path)

            print("Translating to Tamil...")
            tamil_text = translate_to_tamil(original_text)

            print("Generating news script...")
            generate_news_script(tamil_text, news_script_path)

            print("Generating Tamil audio...")
            generate_tamil_audio(tamil_text, tamil_audio_path)

            print("Replacing audio...")
            replace_audio(video_path, tamil_audio_path, output_video_path)

            # WIP
            """print("Adding subtitles...")
            font_path = "fonts/Latha.ttf"
            transcript = transcribe_audio(audio_path, return_dict=True)
            add_subtitle(
                video_path=output_video_path,
                subtitle_video_path=subtitled_video_path,
                transcript=transcript,
                font_path=font_path
            )"""

            uid = str(uuid.uuid4())
            final_video_name = f"translated_{uid}.mp4"
            final_video_path = os.path.join("static", final_video_name)
            shutil.copyfile(output_video_path, final_video_path)

            # Return download URL (user can GET from /download/video?filename=...)
            return JSONResponse(content={
                "download_url": f"/download/video?filename={final_video_name}"
            })

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
        finally:
            await file.close()

@app.get("/download/video")
def get_translated_video(filename: str):
    video_path = os.path.join("static", filename)

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename="translated_video.mp4"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)