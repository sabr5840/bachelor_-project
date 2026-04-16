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
    allow_origins=["*"],  # kun fint i udvikling
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

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            #TODO - system prompt ind skal tilrettes efter behøv, for nu er den generel og ikke specifik for pensionsrådgivning eller andet data
            contents=(
                "Du er en AI-assistent i et bachelorprojekt om pensionsrådgivning. "
                "Du skal svare kort, klart og på dansk. "
                "Du må gerne forklare generelle pensionsbegreber, men du må ikke udgive dig for at være en menneskelig rådgiver. "
                "Hvis spørgsmålet kræver personlig økonomisk vurdering, skal du tage forbehold.\n\n"
                f"Brugerens spørgsmål: {user_text}"
            )
        )

        reply = response.text if response.text else "Jeg kunne ikke generere et svar."

        return {"reply": reply}

    except Exception as e:
        print("Fejl ved Gemini-kald:", repr(e))
        raise HTTPException(
            status_code=500,
            detail=f"Fejl ved kald til AI-modellen: {str(e)}"
        )