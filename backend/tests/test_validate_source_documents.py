import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SOURCE_DIR = Path("data/source_documents")

REQUIRED_SECTIONS = [
    "TITEL:",
    "KILDER:",
    "BESKRIVELSE:",
    "REGLER:",
    "BETINGELSER:",
    "UDBETALING / FUNKTION:",
    "SKAT:",
    "FORDELE:",
    "ULEMPER:",
    "RÅDGIVNING:",
    "FALDGRUBER:",
]

def test_all_txt_files_follow_template():
    files = list(SOURCE_DIR.rglob("*.txt"))

    assert files, "Ingen txt-filer fundet"

    errors = []

    for file in files:
        content = file.read_text(encoding="utf-8")

        for section in REQUIRED_SECTIONS:
            if section not in content:
                errors.append(f"{file} mangler sektion: {section}")

    assert not errors, "\n".join(errors)

from rag_pipeline import retrieve_top_chunks

def test_retrieves_ratepension_for_ratepension_question():
    chunks = retrieve_top_chunks("Hvad er en ratepension?", top_k=3)

    filenames = [chunk["filename"] for chunk in chunks]

    assert any("ratepension" in name for name in filenames)

from main import classify_question

def test_simple_question():
    assert classify_question("Hvad er aldersopsparing?") == "simple"

def test_complex_question():
    assert classify_question("Bør jeg vælge ratepension?") == "complex"

def test_semi_question():
    assert classify_question("Hvordan fungerer udbetaling?") == "semi"