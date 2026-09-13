import os
import getpass

import google.generativeai as genai

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    api_key = getpass.getpass("Enter your Gemini API key: ")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-3.6-flash")

print ("chatbot is ready to answer your questions. Type 'exit' to quit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Exiting the chatbot. Goodbye!")
        break

    response = model.generate_content(user_input)
    print("Chatbot:", response.text)