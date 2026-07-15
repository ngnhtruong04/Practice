from pathlib import Path

def save_output(text: str):
    
    Path("outputs").mkdir(exist_ok=True)
    
    Path("outputs/answer.txt").write_text(
        text,
        encoding="utf-8"
    )