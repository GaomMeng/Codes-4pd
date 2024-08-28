from openai import OpenAI 

client = OpenAI(api_key = "2cad6973f83f47f58e54282199541496", 
                base_url="http://modelhub.4pd.io/learnware/models/openai/4pd/api/v1")

res = client.chat.completions.create(
    model="public/qwen2-7b-instruct-awq@main", 
    messages=[{ "role": "user", "content": "除了中文和英语，你还会说哪些语言，请列举出来，我要学习：" }],
    temperature=1, 
    max_tokens=1000, 
    top_p=1, 
    stop=None, 
)

print(res.choices[0].message.content)