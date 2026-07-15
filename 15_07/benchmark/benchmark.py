import csv
import time

from client import client

prompt = "Write a slogan for an AI startup"

temperatures = [
    0,
    0.3,
    0.7,
    1,
]

with open (
    "benchmark.csv",
    "w",
    newline = "",
    encoding  = "utf-8",
) as f:
    writer = csv.writer(f)
    
    writer.writerow(
        [
            "temperature",
            "latency",
            "tokens",
            "response",
        ]
    )
    
    for temp in temperatures:
        
        start = time.time()
        
        response = client.chat.completions.create(
            
            model = "llama-3.3-70b-versatile",
            
            messages = [
                {
                    "role":"user",
                    "content": prompt
                }
            ],
            
            temperature = temp,
            
            top_p = 1,
        )
        
        latency = round (
            time.time() - start,
            3,
        )
        
        writer.writerow(
            [
                temp,
                latency,
                response.usage.total_tokens,
                response.choices[0].message.content,
            ]
        )