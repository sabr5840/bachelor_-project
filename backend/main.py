import os
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai

from rag_pipeline import retrieve_top_chunks, build_context

load_dotenv()

app = FastAPI()

from pydantic import BaseModel, Field
from typing import List

class ChatMessage(BaseModel):
    role: str
    content: str

class Message(BaseModel):
    message: str
    history: List[ChatMessage] = Field(default_factory=list)

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

Ved generelle definitionsspørgsmål, fx "hvad er ratepension?", skal du svare neutralt og ikke skrive "hos PenSam" eller "hos os".

Hvis spørgsmålet handler om en konkret handling, service eller vejledning hos PenSam, fx at samle pension, ændre begunstigelse, finde overblik eller kontakte rådgiver, må du formulere svaret i en PenSam-kontekst.
I den situation må du skrive som PenSam, fx "hos PenSam", "hos os", "vi kan hjælpe" og "kontakt os", men kun hvis det er i overensstemmelse med den givne kontekst.
Hvis et spørgsmål kan forstås bredt, skal du starte med en generel forklaring og derefter præcisere relevante særlige tilfælde fra konteksten.

Hvis spørgsmålet er generelt (fx "kan jeg få pension udbetalt som engangsbeløb"),
skal du tydeligt afgrænse svaret og forklare, at det afhænger af typen af pension.
Undgå at starte med "Ja", hvis svaret ikke gælder alle tilfælde.

Hvis et spørgsmål kan forstås bredt, skal du starte med en generel forklaring og derefter præcisere relevante særlige tilfælde fra konteksten.

Ved hvorfor-spørgsmål skal du starte med en direkte årsagsforklaring i første sætning, før du uddyber med regler eller eksempler.

Hvis spørgsmålet er generelt (fx "kan jeg få pension udbetalt som engangsbeløb"),
skal du tydeligt afgrænse svaret og forklare, at det afhænger af typen af pension.
Undgå at starte med "Ja", hvis svaret ikke gælder alle tilfælde.
"""


@app.get("/")
def root():
    return {"status": "Backend kører"}


@app.post("/chat")
def chat(msg: Message):
    user_text = msg.message.strip()

    if not user_text:
        raise HTTPException(status_code=400, detail="Beskeden er tom.")

    conversation_history = "\n".join(
        [f"{m.role}: {m.content}" for m in msg.history[-6:]]
    )
    
    try:
        print("User text:", user_text)

        question_type = classify_question(user_text)
        print("Question type:", question_type)

        # Hvis spørgsmålet er for personligt/komplekst, returneres fallback direkte
        if question_type == "complex":
            extra_instruction = """
        Spørgsmålet kræver en personlig vurdering.

        Du skal:
        - først give et kort generelt svar baseret på konteksten
        - derefter tydeligt skrive, at det afhænger af brugerens situation
        - anbefale kontakt til en rådgiver
        - undgå at give konkret personlig rådgivning
        """

       # Use fewer chunks for simple questions and more chunks for semi questions with broader context needs
        if question_type == "simple":
           top_k = 3
        else:
           top_k = 5

        top_chunks = retrieve_top_chunks(user_text, top_k=top_k)

retrieval_query = f"""
Tidligere samtale:
{conversation_history}

 

Nyeste spørgsmål:
{user_text}
"""

 

if question_type == "simple":
    top_k = 3
else:
    top_k = 5

 

top_chunks = retrieve_top_chunks(retrieval_query, top_k=top_k)

 

if not top_chunks:
    return {
        "reply": "Det fremgår ikke af mit datagrundlag.",
        "sources": []
    }

 

context = build_context(top_chunks)
        context = build_context(top_chunks)

        print("----- RETRIEVED CONTEXT -----")
        print(context)
        print("-----------------------------")

        extra_instruction = ""

        if question_type == "semi":
            extra_instruction = """

Spørgsmålet ligger i en gråzone.
Du skal derfor:
- give et kort og klart svar
- forklare kort hvordan det typisk gøres, hvis konteksten indeholder det
- inkludere ét kort forbehold
- anbefale kontakt til en rådgiver, hvis spørgsmålet kræver personlig vurdering
- undgå lange forklaringer og lister
- Hvis konteksten indeholder en selvbetjeningsside eller login-løsning, må du kort forklare, hvor brugeren kan finde oplysningerne.

Eksempel på godt forbehold:
"Det er en god idé at tjekke, om du mister vigtige vilkår eller dækninger, og kontakte en rådgiver ved tvivl."
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

        Tidligere samtale:
        {conversation_history}

        Brugerens nyeste spørgsmål:
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

