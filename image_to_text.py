import google.generativeai as genai
from PIL import Image
import os

# --- Configuration ---
API_KEY = ""  # Replace with your actual Gemini API Key
IMAGE_PATH = "" #path to your image file
# Use a current, recommended vision model
MODEL_NAME = "gemini-1.5-flash-latest"
# -------------------

def read_text_from_image(api_key, image_path):
    """
    Uses a Gemini model to perform OCR on a local image file.
    """
    if not api_key or "YOUR_GEMINI_API_KEY" in api_key:
        print("Error: Please replace 'YOUR_GEMINI_API_KEY' with your actual Gemini API key.")
        return

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return

    try:
        # Configure the Gemini client
        genai.configure(api_key=api_key)

        # Load the image using PIL
        img = Image.open(image_path)
        #convert to greyscale
        # img = img.convert("L")
        #now increase the contrast
        # img = img.point(lambda x: 0 if x < 128 else 255, '1')
        # Optionally, save the processed image for debugging
        img.save("processed_image.png")
        # Select the Gemini model
        model = genai.GenerativeModel(MODEL_NAME)

        # Prepare the prompt and image for the model
        prompt = "Extract the exact text written in this image. ONLY the time on 7 segment display."
        
        # Generate content
        response = model.generate_content([prompt, img])

        # Print the extracted text
        print("\nDetected text:")
        print(response.text)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    read_text_from_image(API_KEY, IMAGE_PATH)