"""
Testing the RagBot
"""
from rag import RagBot 

rag = RagBot()
rag.load_pdfs(["Andino_Agustin_Propuesta_PFC_2024.pdf", "Informe_PFC_Andino_Agustin.pdf"])
# rag.interactive()

question1 = "Cual es el nombre del proyecto?"
response1 = rag.ask(question1)
print(f"Question: {question1}\nResponse: {response1}")

question2 = "Cual es el objetivo principal del proyecto?"
response2 = rag.ask(question2)
print(f"Question: {question2}\nResponse: {response2}")
