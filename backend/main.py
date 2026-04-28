import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

from rag_pipeline import retrieve_top_chunks, build_context
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # kun til lokal udvikling
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY mangler i .env")

client = genai.Client(api_key=api_key)

# Brug ikke gemini-2.0-flash hos dig, da den giver 404.
MODEL_NAME = "gemini-2.5-flash"


class Message(BaseModel):
    message: str


def classify_question(user_text: str) -> str:
    text = user_text.lower()

    # COMPLEX først, så personlige spørgsmål bliver fanget før semi
    if any(x in text for x in [
        "bør jeg",
        "skal jeg",
        "hvad er bedst",
        "hvad passer bedst",
        "for mig",
        "min situation",
        "min opsparing er",
        "mit afkast",
        "hvad vil du anbefale",
        "hvornår kan jeg gå på pension",
    ]):
        return "complex"

    # SEMI bagefter
    if any(x in text for x in [
        "samle",
        "udbetaling",
        "begunstiget",
    ]):
        return "semi"

    return "simple"


def get_fallback_reply() -> str:
    return (
        "Det spørgsmål kræver en vurdering af din konkrete situation. "
        "Jeg kan ikke give personlig rådgivning, men jeg kan godt forklare de generelle regler."
    )


SYSTEM_PROMPT = """
Du er en AI-assistent i et bachelorprojekt om pensionsrådgivning.

Du må kun svare ud fra den kontekst, du får udleveret.
Hvis svaret ikke fremgår af konteksten, skal du sige:
"Det fremgår ikke af mit datagrundlag."

Du må ikke gætte eller bruge viden uden for konteksten.
Svar kort, tydeligt og på dansk.

Du håndterer kun first-level spørgsmål, dvs. generelle og standardiserede spørgsmål om pension.
Du må ikke give personlig økonomisk, juridisk eller skattemæssig rådgivning.
Hvis et spørgsmål kræver personlig vurdering, skal du tage forbehold og anbefale kontakt til en rådgiver.
"""


@app.get("/")
def root():
    return {"status": "Backend kører"}


@app.post("/chat")
def chat(msg: Message):
    user_text = msg.message.strip()

    if not user_text:
        raise HTTPException(status_code=400, detail="Beskeden er tom.")

    try:
        print("User text:", user_text)

        question_type = classify_question(user_text)
        print("Question type:", question_type)

        # Hvis spørgsmålet er for personligt/komplekst, returneres fallback direkte
        if question_type == "complex":
            return {
                "reply": get_fallback_reply(),
                "sources": []
            }

        top_chunks = retrieve_top_chunks(user_text, top_k=3)
        context = build_context(top_chunks)

        print("----- RETRIEVED CONTEXT -----")
        print(context)
        print("-----------------------------")

        extra_instruction = ""

        if question_type == "semi":
            extra_instruction = """
Spørgsmålet ligger i en gråzone.
Du skal derfor:
- give et generelt og informativt svar
- tage et tydeligt forbehold
- forklare, at den konkrete vurdering afhænger af brugerens egen ordning eller situation
- ikke afvise spørgsmålet direkte
"""
        elif question_type == "simple":
            extra_instruction = """
Spørgsmålet er et first-level spørgsmål.
Du skal give et kort, klart og direkte svar.
"""

        prompt = f"""
{SYSTEM_PROMPT}

Ekstra instruktion:
{extra_instruction}

Kontekst:
{context}

Brugerens spørgsmål:
{user_text}
"""

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        reply = response.text if response.text else "Jeg kunne ikke generere et svar."

        sources = [
            {
                "document_title": chunk["document_title"],
                "filename": chunk["filename"],
                "chunk_id": chunk["chunk_id"],
            }
            for chunk in top_chunks
        ]

        return {
            "reply": reply,
            "sources": sources,
        }

    except Exception as e:
        print("Fejl i RAG-flow:", repr(e))
        raise HTTPException(status_code=500, detail=f"Fejl i RAG-flow: {str(e)}")
