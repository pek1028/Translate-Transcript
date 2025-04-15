from gtts import gTTS
from openai import OpenAI
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import moviepy as mp
import os
import shutil
import tempfile

app = FastAPI()

client = OpenAI(api_key="openai-api-key") # Need to replace

def extract_audio(video_path: str, audio_path: str):
    """Extract audio from video and save as WAV."""
    try:
        video = mp.VideoFileClip(video_path)
        if not video.audio:
            video.close()
            raise ValueError("Video has no audio")
        video.audio.write_audiofile(audio_path, codec='aac', bitrate='128k')
        video.close()
    except Exception as e:
        raise Exception(f"Audio extraction failed: {e}")

def transcribe_audio(audio_path: str) -> dict:
    """Transcribe audio to text using Whisper."""
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json"
            )
        return transcript
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
            temperature=0.5
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
        ) # If needed opening and closing text will modify
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
        video = mp.VideoFileClip(video_path)
        tamil_audio = mp.AudioFileClip(tamil_audio_path)
        if tamil_audio.duration > video.duration:
            tamil_audio = tamil_audio.subclip(0, video.duration)
        video = video.set_audio(tamil_audio)
        video.write_videofile(output_path, codec="libx264", audio_codec="aac")
        video.close()
        tamil_audio.close()
    except Exception as e:
        raise Exception(f"MoviePy error: {e}")
    
def add_subtitle(video_path: str, subtitle_video_path: str, transcript: dict, font: str = "Arial-Unicode-MS"):
    """Add subtitles in the video."""
    clip = mp.VideoFileClip(video_path)
    subs = []

    for segment in transcript.get("segments", []):
        text = segment["text"].strip()
        start = segment["start"]
        end = segment["end"]

        subtitle = (mp.TextClip(
                        text,
                        fontsize=36,
                        font=font,
                        color='white',
                        stroke_color='black',
                        stroke_width=1.5)
                    .set_position(('center', 'bottom'))
                    .set_start(start)
                    .set_end(end))
        
        subs.append(subtitle)

    final = mp.CompositeVideoClip([clip, *subs])
    final.write_videofile(subtitle_video_path, codec="libx264", audio_codec="aac")

    print(f"Subtitled video saved at {subtitle_video_path}")

    
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
            transcript_dict = transcribe_audio(audio_path)
            original_text = transcript_dict.get("text", "")

            print("Translating to Tamil...")
            tamil_text = translate_to_tamil(original_text)

            print("Generating news script...")
            generate_news_script(tamil_text, news_script_path)

            print("Generating Tamil audio...")
            generate_tamil_audio(tamil_text, tamil_audio_path)

            print("Replacing audio...")
            replace_audio(video_path, tamil_audio_path, output_video_path)

            print("Adding subtitles...")
            add_subtitle(output_video_path, subtitled_video_path)

            return {
                "video": FileResponse(
                    subtitled_video_path,
                    media_type="video/mp4",
                    filename="output.mp4"
                ),
                "script": FileResponse(
                    news_script_path,
                    media_type="text/plain",
                    filename="script.txt"
                )
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
        finally:
            await file.close()

@app.get("/")
async def root():
    return {"message": "test"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)