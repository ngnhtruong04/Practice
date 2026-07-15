from client import generate_answer
from prompt_loader import load_prompt
from utils import save_output

system_prompt = load_prompt(
    "prompt/system.txt"
)

user_prompt = input("Prompt: ")

answer = generate_answer(
    system_prompt = system_prompt,
    user_prompt = user_prompt,
    temperature = 0.3,
    top_p = 1
)

print(answer)

save_output(answer)