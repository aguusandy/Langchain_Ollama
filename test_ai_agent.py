
from ai_agent import ChatBot

thread_name = "test_thread"
language = "Spanish"

bot = ChatBot(thread_id=thread_name, language=language)

print(f"ChatBot initialized with thread ID: {thread_name} and language: {language}")

question1 = "Hola, como estas ?"
response1 = bot.answer(question1)
print(f"Question: {question1} -> Response: {response1}\n")


question2 = "Eres un asistente virtual? Puedes decirme algo destacable de los agentes AI"
response2 = bot.answer(question2)
print(f"Question: {question2} -> Response: {response2}\n")