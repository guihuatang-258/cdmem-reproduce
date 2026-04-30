from openai import OpenAI
import os

client = OpenAI(
    base_url=os.getenv(
        'OPENAI_API_BASE_URL') if 'OPENAI_API_BASE_URL' in os.environ else None,
    api_key=os.getenv('OPENAI_API_KEY'),
)


result = client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': 'hello how are you',
        }
    ],
    model="minimax/minimax-m2.7",
    # extra_body={"reasoning": {"enabled": False}}
)
print(result)
print(result.choices[0].message.content)
