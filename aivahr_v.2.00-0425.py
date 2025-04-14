import sqlite3
import cv2
import numpy as np
import whisper
import sounddevice as sd
from silero_vad.utils import VADIterator
import torch
from deepface import DeepFace
import io
import base64
from PIL import Image
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import DuckDuckGoSearchRun, Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import requests
from bs4 import BeautifulSoup
from scipy.io.wavfile import write
import subprocess
import tempfile
import os
from queue import Queue
import threading

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect("chat_history.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    human_input TEXT,
                    ai_output TEXT
                 )''')
    conn.commit()
    conn.close()

# Save conversation to SQLite
def save_conversation(session_id, human_input, ai_output):
    conn = sqlite3.connect("chat_history.db")
    c = conn.cursor()
    c.execute("INSERT INTO history (session_id, human_input, ai_output) VALUES (?, ?, ?)",
              (session_id, human_input, ai_output))
    conn.commit()
    conn.close()

# Load conversation from SQLite into memory
def load_conversation_to_memory(memory, session_id):
    conn = sqlite3.connect("chat_history.db")
    c = conn.cursor()
    c.execute("SELECT human_input, ai_output FROM history WHERE session_id = ? ORDER BY id", (session_id,))
    rows = c.fetchall()
    for human_input, ai_output in rows:
        memory.chat_memory.add_user_message(human_input)
        memory.chat_memory.add_ai_message(ai_output)
    conn.close()

# Define Bible Scraper tool
def scrape_bible_info(query):
    url = f"https://www.biblegateway.com/quicksearch/?quicksearch={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        verses = soup.find_all("div", class_="search-result")[:2]
        result = "\n".join([verse.get_text(strip=True) for verse in verses]) or "No verses found."
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# Convert frame to base64 for MiniCPM-o
def frame_to_base64(frame):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Extract audio from video (for MP4)
def extract_audio_from_video(video_source, output_path, duration=None):
    try:
        cmd = [
            "ffmpeg", "-i", video_source, "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1", output_path
        ]
        if duration:
            cmd.insert(3, "-t")
            cmd.insert(4, str(duration))
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path
    except Exception as e:
        return f"Error extracting audio: {str(e)}"

# Transcribe audio with Whisper
def transcribe_audio(audio_path):
    try:
        model = whisper.load_model("tiny")
        result = model.transcribe(audio_path, fp16=False)
        os.remove(audio_path)
        return result["text"].strip() or "No speech detected."
    except Exception as e:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return f"Error transcribing audio: {str(e)}"

# Face recognition with DeepFace
def verify_face(frame, reference_image="you.jpg"):
    try:
        # Save frame temporarily
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_frame:
            cv2.imwrite(temp_frame.name, frame)
            result = DeepFace.verify(
                img1_path=temp_frame.name,
                img2_path=reference_image,
                model_name="VGG-Face",
                distance_metric="cosine",
                enforce_detection=True
            )
            os.remove(temp_frame.name)
        return result["verified"], result["distance"]
    except Exception as e:
        return False, float("inf")

# Voice-activated capture with face and keyphrase
def capture_face_voice_keyphrase(video_source=0, keyphrase="hey bot", max_silence=0.5, max_duration=15):
    sample_rate = 16000
    vad = VADIterator(sample_rate=sample_rate, threshold=0.5)
    audio_queue = Queue()
    video_frames = []
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        return None, None, "Error: Could not open video source."

    def audio_callback(indata, frames, time, status):
        audio_queue.put(indata.copy())

    # Start audio recording
    stream = sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback)
    stream.start()

    # Variables
    audio_buffer = []
    speech_detected = False
    face_detected = False
    silence_duration = 0
    start_time = cv2.getTickCount()
    fps = 2
    frame_interval = 1 / fps
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (384, int(384 * frame.shape[0] / frame.shape[1])))
            frame_count += 1

            # Check face every 0.5s (~1 frame at 2 FPS)
            if frame_count % int(0.5 / frame_interval) == 0:
                verified, distance = verify_face(frame)
                if verified and distance < 0.4:  # Tight threshold for security
                    face_detected = True
                    video_frames.append(frame)
                else:
                    face_detected = False
                    video_frames = []  # Reset if face not matched

            # Keep ~4s of frames (8 frames)
            if len(video_frames) > 8:
                video_frames.pop(0)

            # Process audio if face detected
            if face_detected:
                while not audio_queue.empty():
                    chunk = audio_queue.get()
                    vad_result = vad(chunk.flatten(), sample_rate)
                    if vad_result and vad_result.get("start"):
                        speech_detected = True
                    if speech_detected:
                        audio_buffer.append(chunk)

                    # Check for silence
                    if speech_detected and (not vad_result or vad_result.get("end")):
                        silence_duration += len(chunk) / sample_rate
                        if silence_duration >= max_silence:
                            break
                    else:
                        silence_duration = 0

                if speech_detected and silence_duration >= max_silence:
                    break

            # Timeout
            elapsed = ((cv2.getTickCount() - start_time) / cv2.getTickFrequency())
            if elapsed >= max_duration:
                break

        stream.stop()
        stream.close()
        cap.release()

        if not audio_buffer or not face_detected:
            return None, video_frames, "No face or speech detected."

        # Save and transcribe audio
        audio_data = np.concatenate(audio_buffer, axis=0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            write(temp_audio.name, sample_rate, audio_data)
            user_input = transcribe_audio(temp_audio.name)

        # Check keyphrase
        if not user_input.lower().startswith(keyphrase.lower()):
            return None, video_frames, f"No keyphrase '{keyphrase}' detected."

        # Remove keyphrase from input
        user_input = user_input[len(keyphrase):].strip() or "What did you say?"

        return user_input, video_frames, None
    except Exception as e:
        stream.stop()
        stream.close()
        cap.release()
        return None, video_frames, f"Error capturing: {str(e)}"

# Define tools
tools = [
    Tool(
        name="DuckDuckGo Search",
        func=lambda q: DuckDuckGoSearchRun(max_results=3).run(q),
        description="Search the web using DuckDuckGo."
    ),
    Tool(
        name="Bible Scraper",
        func=scrape_bible_info,
        description="Scrape Bible-related information."
    )
]

# Initialize ChatOllama with MiniCPM-o
llm = ChatOllama(
    model="minicpm-o",
    temperature=0.7,
    base_url="http://localhost:11434",
    keep_alive="1h",
    num_ctx=2048
)

# Set up memory
memory = ConversationBufferMemory(return_messages=True)

# Define agent prompt
agent_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "history", "tools", "tool_names"],
    template="""
You are a helpful assistant with access to tools: {tool_names}.
Talk like a chill friend who loves tech.
Handle requests based on input:
- If input mentions past conversation, recall from history.
- If input is casual, chat normally.
- If input asks for search or Bible verses, use tools.
- If input describes a video or scene, analyze provided frames or transcribed audio.

Tools:
{tools}

Chat History:
{history}

User Input:
{input}

Agent Scratchpad:
{agent_scratchpad}

Respond with:
- Thought: Your reasoning.
- Action: Tool to use or "Finish" for no tool.
- Action Input: Tool input, omit if "Finish".
- Final Answer: [response] if "Finish", omit for tools.
"""
)

# Create ReAct agent
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors=True)

# Face, voice, keyphrase chat
def chat_with_face_voice_keyphrase(session_id, video_source=0, keyphrase="hey bot", max_silence=0.5, max_duration=15):
    print("Bot: Waiting for your face and keyphrase... ", end="", flush=True)
    full_output = ""

    # For MP4, process as single clip
    if isinstance(video_source, str) and video_source.endswith((".mp4", ".avi")):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            audio_result = extract_audio_from_video(video_source, temp_audio.name)
            if "Error" in audio_result:
                print(audio_result)
                return audio_result
            user_input = transcribe_audio(temp_audio.name)
            cap = cv2.VideoCapture(video_source)
            video_frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (384, int(384 * frame.shape[0] / frame.shape[1])))
                verified, _ = verify_face(frame)
                if verified:
                    video_frames.append(frame)
                if len(video_frames) > 8:
                    video_frames.pop(0)
            cap.release()
            if not video_frames:
                error = "No matching face detected in video."
                print(error)
                return error
            if not user_input.lower().startswith(keyphrase.lower()):
                error = f"No keyphrase '{keyphrase}' detected."
                print(error)
                return error
            user_input = user_input[len(keyphrase):].strip() or "What did you say?"
            error = None
    else:
        user_input, video_frames, error = capture_face_voice_keyphrase(
            video_source, keyphrase, max_silence, max_duration
        )

    if error:
        print(error)
        if "No face or speech" in error or "keyphrase" in error:
            user_input = input("[No face/keyphrase, type input]: ")
            if user_input.lower() == "exit":
                print("Saving and exiting...")
                return
            # Capture frames for context
            cap = cv2.VideoCapture(video_source)
            video_frames = []
            for _ in range(4):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (384, int(384 * frame.shape[0] / frame.shape[1])))
                video_frames.append(frame)
            cap.release()
        else:
            save_conversation(session_id, "face_voice_input", error)
            return error

    # Process input
    try:
        frame_b64 = [frame_to_base64(f) for f in video_frames]
        messages = [
            SystemMessage(content="Talk like a chill tech friend. Analyze video frames or audio if provided."),
            HumanMessage(content=user_input, images=frame_b64)
        ]

        print("Bot: ", end="", flush=True)
        for event in agent_executor.stream({"input": user_input}):
            if "output" in event:
                chunk = event["output"]
                print(chunk, end="", flush=True)
                full_output += chunk
            elif "actions" in event:
                for action in event["actions"]:
                    print(f"[Using {action.tool}...]", end="", flush=True)
        print()
        save_conversation(session_id, user_input, full_output)
        return full_output
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        save_conversation(session_id, user_input, error_msg)
        return error_msg

# Main loop
if __name__ == "__main__":
    init_db()
    session_id = "user_session_001"
    load_conversation_to_memory(memory, session_id)
    
    print("Chatbot with face recognition and keyphrase ready! Say 'Hey, bot' to start, or type 'exit'.")
    video_source = 0  # Or "path/to/video.mp4"
    keyphrase = "hey bot"
    max_silence = 0.5
    max_duration = 15
    while True:
        result = chat_with_face_voice_keyphrase(
            session_id, video_source=video_source, keyphrase=keyphrase,
            max_silence=max_silence, max_duration=max_duration
        )
        if isinstance(result, str) and "exit" in result.lower():
            break
