from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    base_url=os.getenv('OPENAI_API_BASE_URL'),
    # base_url or (os.getenv('OPENAI_API_BASE_URL') if 'OPENAI_API_BASE_URL' in os.environ else None),
    api_key=os.getenv('OPENAI_API_KEY')
    # api_key or os.getenv('OPENAI_API_KEY'),
)


result = client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': '说hello\nworld，返回json格式',
        }
    ],
    model="qwen/qwen3.6-flash",
    # stop="\n",
    response_format={'type': 'json_object'},
    extra_body={"reasoning": {"enabled": False}}
)
print(result)
print(result.choices[0].message.content)
