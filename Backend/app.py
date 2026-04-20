from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
from pathlib import Path
import shutil

from sbrain_translator import OntologyDrivenTranslator
from sbrain_learning_memory import SBrainLearningMemory

app = FastAPI(title="S-Brain v6.2 - Aerospace Standards Translator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

translator = OntologyDrivenTranslator()
memory = SBrainLearningMemory()

@app.get("/")
async def root():
    return {"message": "S-Brain v6.2 is running with Learning Memory"}

@app.post("/translate")
async def translate(
    file: UploadFile = File(...),
    from_std: str = Form(...),
    to_std: str = Form(...)
):
    temp_path = Path(f"temp_{file.filename}")
    with temp_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = translator.translate(
            xml_input=str(temp_path),
            from_std=from_std,
            to_std=to_std
        )
        return {
            "xml": result["xml_string"],
            "coverage": result.get("coverage", 0),
            "log": result.get("log", [])[:30]
        }
    finally:
        temp_path.unlink(missing_ok=True)

@app.post("/confirm-mapping")
async def confirm_mapping(
    from_std: str = Form(...),
    original_tag: str = Form(...),
    new_tag: str = Form(...),
    to_std: str = Form(...),
    example_value: str = Form(None)
):
    memory.record_correction(from_std, original_tag, new_tag, to_std, example_value)
    return {"status": "learned", "message": "Mapping permanently stored"}

@app.get("/memory-stats")
async def memory_stats():
    return memory.get_stats()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
