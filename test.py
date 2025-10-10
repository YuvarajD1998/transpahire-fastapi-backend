import asyncio
import os
import google.genai as genai
from google.genai import types

# Make sure your GEMINI_API_KEY is set as an environment variable
# Example in terminal: export GEMINI_API_KEY="your_api_key"

API_KEY = "AIzaSyBiAYBW8n8oYaRuAUI6jtNwKHzhw8ihUwM"
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable not set!")

async def main():
    # Initialize the Gemini client
    client = genai.Client(api_key=API_KEY)

    # Example prompt
    prompt = "Write a short greeting message from a friendly AI."

    try:
        # Async call to generate content
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.5,
                    max_output_tokens=100
                ),
            ),
        )

        print("Gemini Response:")
        print(response.text)

    except Exception as e:
        print("Error calling Gemini:", e)

if __name__ == "__main__":
    asyncio.run(main())
