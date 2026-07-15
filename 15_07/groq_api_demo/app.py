from client import generate_answer
from utils import save_output

def main() -> None:
    prompt = input("Enter Prompt:\n")
    
    answer = generate_answer(prompt)
    
    print("\nAI response:\n")
    
    print(answer)
    
    save_output(answer)
    
if __name__ == "__main__":
    main()
