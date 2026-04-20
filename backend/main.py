import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fint i udvikling
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("GEMINI_API_KEY")
print("API key loaded:", bool(api_key))

if not api_key:
    raise RuntimeError("GEMINI_API_KEY mangler i miljøvariablerne.")

client = genai.Client(api_key=api_key)


class Message(BaseModel):
    message: str


def classify_question(user_text: str) -> str:
    text = user_text.lower()

    # 1. SEMI først (vigtig!)
    if any(x in text for x in [
        "skat",
        "samle",
        "udbetaling",
        "begunstiget",
    ]):
        return "semi"

    # 2. COMPLEX bagefter
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

    return "simple"

def build_prompt(user_text: str, question_type: str) -> str:
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

    return f"""
Du er en AI-assistent i et bachelorprojekt om pensionsrådgivning.

Du håndterer kun first-level spørgsmål, dvs. generelle og standardiserede spørgsmål om pension.

Regler:
- Svar altid på dansk
- Svar kort, klart og i et letforståeligt sprog
- Forklar kun generelle pensionsbegreber, regler og processer
- Giv ikke personlig økonomisk, juridisk eller skattemæssig rådgivning
- Hvis spørgsmålet er uklart, så stil ét kort opklarende spørgsmål
- Hvis spørgsmålet handler om brugerens konkrete situation, så sig tydeligt, at det kræver individuel vurdering
- Hvis spørgsmålet ligger i gråzonen, så giv et generelt svar med et tydeligt forbehold
- Opfind ikke fakta eller regler, du ikke er sikker på
- Undgå unødvendige introer som "Her er en kort forklaring", medmindre det giver mening

Ekstra instruktion:
{extra_instruction}

Brugerens spørgsmål:
{user_text}
"""


def get_fallback_reply() -> str:
    return (
        "Det spørgsmål kræver en vurdering af din konkrete situation. "
        "Jeg kan desværre ikke give personlig rådgivning, men jeg kan godt forklare de generelle regler, hvis du ønsker det."
    )


@app.get("/")
def root():
    return {"status": "Backend kører"}


@app.post("/chat")
def chat(msg: Message):
    user_text = msg.message.strip()

    if not user_text:
        raise HTTPException(status_code=400, detail="Beskeden må ikke være tom.")

    try:
        print("User text:", user_text)

        question_type = classify_question(user_text)
        print("Question type:", question_type)

        if question_type == "complex":
            return {"reply": get_fallback_reply()}

        prompt = build_prompt(user_text, question_type)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        reply = response.text if response.text else "Jeg kunne ikke generere et svar."

        return {"reply": reply}

    except Exception as e:
        print("Fejl ved Gemini-kald:", repr(e))
        raise HTTPException(
            status_code=500,
            detail=f"Fejl ved kald til AI-modellen: {str(e)}"
        )