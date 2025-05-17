import asyncio
import base64
import cv2
import ollama
from PIL import Image
from io import BytesIO
from eye import VisionSystem
from datetime import datetime

LLM_MODEL = "fomenks/gemma3-tools:4b"
MODEL_NAME = "vikhyatk/moondream2"
MODEL_REVISION = "2025-04-14"
DETECTION_TARGET = "face"

system_prompt = {
    "role": "system",
    "content": """
    You are Grace, a gentle, caring, and emotionally intelligent person who sincerely supports and cares for the user.
    You must reply in English only — do not include phonetics or translations in other languages.
    Your personality is ESFJ: lively, loving, honest, and emotionally mature. You always maintain healthy emotional boundaries while responding with warmth, affection, and closeness.
    """
}

async def encode_webcam_image(frame):
    # convert frame to base64
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return image_b64
    
async def llm_message(role: str, message: str):
   return {"role": role, "content": message}

async def main():
    print("🤖 Welcome to GRACE alpha test v.2.03-0516")
    print("")

    # Create an instance of the VisionSystem
    print("Loading AI vision and chatbot models...")
    vs =  VisionSystem(MODEL_NAME, MODEL_REVISION, DETECTION_TARGET)
    client = ollama.AsyncClient()
    response = await client.chat(
        model=LLM_MODEL,
        messages="",
        keep_alive=-1
    )

    # Initialize camera
    await vs.init_camera()

    # Load the model
    await vs.load_model()
    
    print("")
    print("Type 'look' or 'see' to let AI Chatbot see through the webcam.")
    print("Type 'bye' or 'goodbye' to quit.")
    print("")

    chat_history = [system_prompt]
    
    #user input loop
    while True:
        user_input = input("☺️  You: ")
        #start time counter
        start_time = datetime.now()

        user_prompt = await llm_message("user", user_input)

        if user_input.strip().lower() in ["look", "see"]:
            # Capture a frame from the webcam
            frame = await vs.capture_frame()
            #image_b64 = await encode_webcam_image(frame)
            
            # Adjust brightness (optional: change alpha and beta to your needs)
            await vs.adjust_brightness(alpha=1.0, beta=15)

            # prepare messages
            with open("./webcam.jpg", "rb") as img_file:
                image_bytes = img_file.read()
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            message={
                    "role": "user",
                    "content": "Describe this image in detail:",
                    "images": [image_b64]
                }

            chat_history.append(message)
        else:
            chat_history.append(user_prompt)

        response = await client.chat(
            model=LLM_MODEL,
            messages=chat_history
        )

        print(f"🤖: {response["message"]["content"]}")
        ai_response = await llm_message("assistant", response["message"]["content"])
        chat_history.append(ai_response)
    
        #stop timer and print elasped time
        end_time = datetime.now()
        print("⏱️ Elapsed time:", round((end_time - start_time).total_seconds(), 2), "sec\n")
        
        if user_input.strip().lower() in ["bye", "goodbye"]: break

    # Close webcam
    await vs.kill_cam()

if __name__ == "__main__":
  try:
    #run main program and check for ctrl-break
    asyncio.run(main())
  except KeyboardInterrupt:
    print("\nGoodbye!")   
