import json
import os
import mimetypes
import google.generativeai as genai
from pydantic import BaseModel, ValidationError
from typing import List, Optional

# FastAPI app to expose HTTP endpoints for the frontend
from fastapi import FastAPI, UploadFile, File, Form, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- Pydantic Models for Structured Output ---
class DetailedTip(BaseModel):
    """A model for a single piece of feedback with a timestamp."""
    original_timestamp: str
    transcribed_timestamp: str
    suggestion: str

class FeedbackReport(BaseModel):
    """The overall feedback report structure."""
    score: float
    overall_feedback: str
    word_repetition_score: float
    word_repetition_count: int
    speaking_pace_score: float
    speaking_pace_count: int
    filler_words_score: float
    voice_clarity_score: float
    filler_words_count: int
    repetitive_words_list: List[str]
    detailed_tips: List[DetailedTip]

# --- API Configuration ---
# IMPORTANT: Replace "YOUR_GEMINI_API_KEY" with your actual Gemini API key.
# For security, it's recommended to use environment variables instead of hardcoding the key.
# from dotenv import load_dotenv
# load_dotenv()
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
try:
    API_KEY = "AIzaSyDMQM-Ez9GFs7aad_yAaQbSNJDlou1qElU"
    if not API_KEY or API_KEY == "YOUR_GEMINI_API_KEY":
        print("Please set GEMINI_API_KEY environment variable before running the server.")
    else:
        genai.configure(api_key=API_KEY)
except Exception as e:
    print(f"Error configuring API key: {e}")


# --- 1. & 2. Gemini API: Content Generation ---
def generate_script_from_topic(topic, duration):
    """
    Generates a presentation script for a specific duration using the Gemini API.
    """
    print(f"\n🤖 Generating a {duration}-minute script for topic: '{topic}' with Gemini...")
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"""
    Generate a script for a {duration}-minute session on the topic: '{topic}'.
    Adapt your style to the user's intent expressed in the topic text (e.g., seminar, presentation, introduction, interview, lesson, pitch, demo). Do not assume a default style.
    The script should be divided into a logical number of parts appropriate for the duration.
    Provide a generic timestamp range (e.g., "00:00-00:05") for each part.
    Your response MUST be a valid JSON object with a single key "script".
    The value of "script" should be a list of dictionary objects.
    Each dictionary should have two keys: "timestamp" (string) and "line" (string).
    Example format:
    {{
      "script": [
        {{"timestamp": "00:00-00:05", "line": "Introduction line tailored to the requested context."}},
        {{"timestamp": "00:06-00:12", "line": "Next point."}}
      ]
    }}
    """
    try:
        response = model.generate_content(prompt)
        # Clean up the response to ensure it's valid JSON
        json_response = response.text.strip().replace("```json", "").replace("```", "")
        data = json.loads(json_response)
        # Add a title for consistency, though it's not used elsewhere
        data['title'] = f"Presentation on {topic}"
        return data
    except Exception as e:
        print(f"An error occurred during script generation: {e}")
        print("Falling back to a default script.")
        # Fallback to a default script if API fails
        return {
            "title": "Default Topic",
            "script": [
                {"timestamp": "00:00-00:05", "line": "API call failed. This is a default script."},
                {"timestamp": "00:06-00:12", "line": "Please check your API key and network connection."},
            ]
        }

# --- 3. Get Audio File from User ---
def get_audio_from_user():
    """
    Gets the path to a recorded audio file from the user.
    """
    print("\n🎤 Please provide the path to your recorded speech (e.g., speech.wav or speech.mp3).")
    # audio_path = input("Enter file path: ")
    audio_path = "speech.wav"
    while not os.path.exists(audio_path):
        print("Error: File not found. Please enter a valid path.")
        audio_path = input("Enter file path: ")
    return audio_path

# --- 4. Gemini API: Audio Transcription ---
def transcribe_audio_with_timestamps(audio_path):
    """
    Uploads an audio file and uses the Gemini API to transcribe it with timestamps.
    """
    print(f"\n🔄 Processing and transcribing audio from '{audio_path}' with Gemini...")
    try:
        # Use a model that supports audio input, like gemini-1.5-flash
        model = genai.GenerativeModel('gemini-2.5-flash')

        # Read the file into memory
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        
        # Get the mime type
        mime_type, _ = mimetypes.guess_type(audio_path)
        if mime_type is None:
            # Default to a common audio format if guessing fails
            mime_type = "audio/wav"

        # Create a part dictionary for the audio blob
        audio_part = {
            "mime_type": mime_type,
            "data": audio_bytes
        }

        # Prompt for transcription with timestamps
        prompt = f"""
        Please transcribe the following audio.
        Provide a timestamp range for each transcribed sentence or significant pause.
        Your response MUST be a valid JSON object with a single key "transcription".
        The value of "transcription" should be a list of dictionary objects.
        Each dictionary should have two keys: "timestamp" (string) and "line" (string).
        Example format:
        {{
          "transcription": [
            {{"timestamp": "00:00-00:05", "line": "The first transcribed sentence."}},
            {{"timestamp": "00:06-00:12", "line": "The second transcribed sentence."}}
          ]
        }}
        """

        # Make the API call with the audio data directly in the prompt
        response = model.generate_content([prompt, audio_part])

        # Clean up the response to ensure it's valid JSON
        json_response = response.text.strip().replace("```json", "").replace("```", "")
        data = json.loads(json_response)

        return data

    except Exception as e:
        print(f"An error occurred during audio transcription: {e}")
        print("Falling back to dummy transcription.")
        # Fallback if transcription fails
        return {
            "transcription": [
                {"timestamp": "00:00-00:05", "line": "Audio transcription failed."},
                {"timestamp": "00:06-00:11", "line": "Please check the audio file format and your API setup."},
            ]
        }

def transcribe_audio_bytes(mime_type: str, audio_bytes: bytes):
    """
    Transcribe audio already loaded as bytes using Gemini model.
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        audio_part = {
            "mime_type": mime_type or "audio/webm",
            "data": audio_bytes,
        }
        prompt = (
            "Please transcribe the following audio.\n"
            "Provide a timestamp range for each transcribed sentence or significant pause.\n"
            "Your response MUST be a valid JSON object with a single key \"transcription\".\n"
            "The value of \"transcription\" should be a list of dictionary objects with keys \"timestamp\" and \"line\"."
        )
        response = model.generate_content([prompt, audio_part])
        json_response = response.text.strip().replace("```json", "").replace("```", "")
        data = json.loads(json_response)
        return data
    except Exception:
        return {
            "transcription": [
                {"timestamp": "00:00-00:05", "line": "Audio transcription failed."},
                {"timestamp": "00:06-00:11", "line": "Please check the audio file format and your API setup."},
            ]
        }

# --- 5. & 6. Gemini API: Comparison and Feedback ---
def get_feedback_from_gemini(original_script, transcription_data) -> FeedbackReport:
    """
    Uses the Gemini API to compare the original script with the transcription
    and generate a score and detailed feedback using a Pydantic schema.
    """
    print("\n🧠 Analyzing your speech with Gemini for feedback...")
    model = genai.GenerativeModel('gemini-2.5-flash')

    original_json = json.dumps(original_script, indent=2)
    transcribed_json = json.dumps(transcription_data['transcription'], indent=2)

    prompt = f"""
    You are a public speaking coach. Your task is to analyze a user's speech delivery by comparing an original script to their transcribed speech and providing a detailed analysis.

    Here is the original script:
    ```json
    {original_json}
    ```

    Here is the user's transcribed speech:
    ```json
    {transcribed_json}
    ```

    Please perform the following actions and return the response in the requested JSON format.
    1.  **Overall Score (0-10)**: Provide a numeric score based on how well the user followed the script.
    2.  **Overall Feedback**: Give a very short, one-sentence overall feedback on the performance.
    3.  **Speech Analysis**:
        - **Word Repetition Score (0-10)**: Score how often the user unnecessarily repeats words.
        - **Speaking Pace Score (0-10)**: Based on timestamps and text, score the speaking pace.
        - **Filler Words Score (0-10)**: Score the usage of filler words (e.g., 'um', 'ah', 'like'). Higher score means fewer fillers.
        - **Voice Clarity Score (0-10)**: Based on the coherence of the transcribed text, estimate speech clarity.
        - **Filler Words Count**: Provide the total count of identified filler words.
        - **Repetitive Words List**: Provide a list of words that were repeated unnecessarily.
    4.  **Detailed Tips**: For each entry in the original script, find the corresponding part in the user's transcription. Provide a brief, one-sentence suggestion for each comparison.

    Your response MUST be a valid JSON object that strictly adheres to the following structure.
    {{
      "score": <float>,
      "overall_feedback": "<string>",
      "word_repetition_score": <float>,
      "word_repetition_count": <int>,
      "speaking_pace_count": <int>,
      "speaking_pace_score": <float>,
      "filler_words_score": <float>,
      "voice_clarity_score": <float>,
      "filler_words_count": <int>,
      "repetitive_words_list": ["<string>", "<string>"],
      "detailed_tips": [
        {{
          "original_timestamp": "<string from original script>",
          "transcribed_timestamp": "<corresponding string from transcription, or 'N/A' if not found>",
          "suggestion": "<string>"
        }}
      ]
    }}
    Do not include any other text or formatting like markdown backticks.
    """
    try:
        # Use generation_config to ask for a JSON response
        generation_config = genai.types.GenerationConfig(
            response_mime_type="application/json"
        )
        response = model.generate_content(prompt, generation_config=generation_config)
        
        json_response = response.text.strip().replace("```json", "").replace("```", "")
        
        feedback_data = json.loads(json_response)
        # Coerce missing or malformed fields with safe defaults
        coerced = {
            "score": float(feedback_data.get("score", 0.0) or 0.0),
            "overall_feedback": feedback_data.get("overall_feedback", "") or "",
            "word_repetition_score": float(feedback_data.get("word_repetition_score", 0.0) or 0.0),
            "word_repetition_count": int(feedback_data.get("word_repetition_count", 0) or 0),
            "speaking_pace_score": float(feedback_data.get("speaking_pace_score", 0.0) or 0.0),
            "speaking_pace_count": int(feedback_data.get("speaking_pace_count", 0) or 0),
            "filler_words_score": float(feedback_data.get("filler_words_score", 0.0) or 0.0),
            "voice_clarity_score": float(feedback_data.get("voice_clarity_score", 0.0) or 0.0),
            "filler_words_count": int(feedback_data.get("filler_words_count", 0) or 0),
            "repetitive_words_list": feedback_data.get("repetitive_words_list") or [],
            "detailed_tips": feedback_data.get("detailed_tips") or [],
        }
        # Ensure types for list fields
        if not isinstance(coerced["repetitive_words_list"], list):
            coerced["repetitive_words_list"] = []
        if not isinstance(coerced["detailed_tips"], list):
            coerced["detailed_tips"] = []
        return FeedbackReport(**coerced)

    except (ValidationError, json.JSONDecodeError) as e:
        print(f"Error validating or parsing the feedback response: {e}")
    except Exception as e:
        print(f"An error occurred during feedback generation: {e}")
    
    # Fallback to a default feedback report on error
    return FeedbackReport(
        score=0.0,
        overall_feedback="Could not generate valid feedback due to an API or parsing error.",
        word_repetition_score=0.0,
        word_repetition_count=0,
        speaking_pace_score=0.0,
        speaking_pace_count=0,
        filler_words_score=0.0,
        voice_clarity_score=0.0,
        filler_words_count=0,
        repetitive_words_list=[],
        detailed_tips=[]
    )


def display_feedback_report(feedback_data: FeedbackReport):
    """
    Displays the feedback report (as a Pydantic object) in a user-friendly format.
    """
    print("\n📊 Generating your feedback report...")
    print("="*50)
    print("        PRESENTATION FEEDBACK REPORT")
    print("="*50)

    print(f"Overall Score: {feedback_data.score:.2f} / 10.0\n")
    print("--- Overall Feedback ---")
    print(f"{feedback_data.overall_feedback}\n")

    print("--- Speech Analysis ---")
    print(f"Speaking Pace Score:     {feedback_data.speaking_pace_score:.2f} / 10.0")
    print(f"Voice Clarity Score:     {feedback_data.voice_clarity_score:.2f} / 10.0")
    print(f"Filler Words Score:      {feedback_data.filler_words_score:.2f} / 10.0 (Count: {feedback_data.filler_words_count})")
    repetitive_words_str = ", ".join(feedback_data.repetitive_words_list) if feedback_data.repetitive_words_list else "None"
    print(f"Word Repetition Score:   {feedback_data.word_repetition_score:.2f} / 10.0 (Words: {repetitive_words_str})\n")


    print("--- Detailed Comparison & Tips ---")
    tips = feedback_data.detailed_tips
    if not tips:
        print("No specific tips were generated.")
    else:
        for tip in tips:
            print(f"Original [{tip.original_timestamp}] vs. Your Speech [{tip.transcribed_timestamp}]:")
            print(f"  - Suggestion: {tip.suggestion}\n")

    print("="*50)


# ------------------------ FastAPI Server ------------------------
app = FastAPI(title="Voice Coach Backend")

# Allowed origins list for validation
ALLOWED_ORIGINS = [
    "https://frontend-53528.web.app",
    "https://frontend-53528.firebaseapp.com",
    "https://backend-0d8r.onrender.com",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Enable CORS for local frontend and deployed frontend
# Enable CORS with proper OPTIONS handling - using specific origins for Firebase
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

@app.get("/health")
def health():
    return {"status": "ok"}

# Explicit OPTIONS handler for /generate_script to handle preflight requests
# This MUST be defined before the POST route to ensure it's matched first
@app.options("/generate_script")
async def options_generate_script(request: Request):
    """Handle OPTIONS preflight for /generate_script"""
    origin = request.headers.get("Origin")
    # Validate origin or use the first allowed origin as fallback
    if origin and origin in ALLOWED_ORIGINS:
        allow_origin = origin
    else:
        allow_origin = ALLOWED_ORIGINS[0] if ALLOWED_ORIGINS else "*"
    
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": allow_origin,
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "3600",
            "Access-Control-Allow-Credentials": "true",
        }
    )

# Explicit OPTIONS handler for /analyze to handle preflight requests
# This MUST be defined before the POST route to ensure it's matched first
@app.options("/analyze")
async def options_analyze(request: Request):
    """Handle OPTIONS preflight for /analyze"""
    origin = request.headers.get("Origin")
    # Validate origin or use the first allowed origin as fallback
    if origin and origin in ALLOWED_ORIGINS:
        allow_origin = origin
    else:
        allow_origin = ALLOWED_ORIGINS[0] if ALLOWED_ORIGINS else "*"
    
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": allow_origin,
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
            "Access-Control-Max-Age": "3600",
            "Access-Control-Allow-Credentials": "true",
        }
    )

class GenerateRequest(BaseModel):
    topic: str
    duration: Optional[int] = 3

@app.post("/generate_script")
def api_generate_script(payload: GenerateRequest):
    data = generate_script_from_topic(payload.topic, payload.duration or 3)
    return data

@app.post("/analyze")
async def api_analyze(
    audio: UploadFile = File(...),
    original_script_json: str = Form(...),
):
    # Parse original script JSON (expects { "script": [ {timestamp, line}, ... ] })
    try:
        original_container = json.loads(original_script_json)
        if isinstance(original_container, list):
            original_script = original_container
        else:
            original_script = original_container.get("script", [])
    except Exception:
        original_script = []

    audio_bytes = await audio.read()
    # Save uploaded audio to backend/speech.wav
    try:
        target_path = os.path.join(os.path.dirname(__file__), 'speech.wav')
        with open(target_path, 'wb') as f:
            f.write(audio_bytes)
    except Exception as e:
        print(f"Failed to save audio file: {e}")
    mime_type = audio.content_type or mimetypes.guess_type(audio.filename or "")[0] or "audio/webm"

    transcription_data = transcribe_audio_bytes(mime_type, audio_bytes)
    feedback = get_feedback_from_gemini(original_script, transcription_data)

    # Convert Pydantic model to dict for JSON response
    return json.loads(feedback.model_dump_json())


if __name__ == "__main__":
    # Run: python app.py
    uvicorn.run(app, host="0.0.0.0", port=8000)

