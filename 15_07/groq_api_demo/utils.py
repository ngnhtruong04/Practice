from pathlib import Path

def save_output(
    text: str, 
    filename: str = "outputs/response.txt",
)-> None: 
    
    path = Path(filename)
    
    path.parent.mkdir(
        parents = True,
        exist_ok = True,
    )
    
    path.write_text(
        text, 
        encoding = "utf-8",
    )