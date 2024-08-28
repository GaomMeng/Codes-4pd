from openai import OpenAI 

client = OpenAI(api_key = "2cad6973f83f47f58e54282199541496", 
                base_url="http://modelhub.4pd.io/learnware/models/openai/4pd/api/v1")

def generate_promot(from_lang, to_lang, prompt):
    # return f"Translate from {from_lang} to {to_lang}: {prompt}"
    return f"There is a text:\n'{prompt}'\nFirst, appropriately supplement the context of this passage to increase the amount of information. Second, translate the context first, then based on the context, translate the following sentence from {from_lang} to {to_lang}. Third, do not return or display any steps, contexts, etc., just return the translated content and print it.\nTranslation:" 

def generate_response(prompt):
    res = client.chat.completions.create(
        model="public/qwen2-7b-instruct-awq@main", 
        messages=[{ "role": "user", "content": prompt }],
        temperature=1, 
        max_tokens=1000, 
        top_p=1, 
        stop=None, 
    )
    # print(f'\n\n   {res}  \n\n')
    return res.choices[0].message.content

def translate(message):
    from_lang = message['from']
    to_lang = message['to']
    texts = message['texts']
    translated = []
    for text in texts:
        prompt = generate_promot(from_lang, to_lang, text)
        print(prompt)
        response = generate_response(prompt)
        translated.append(response)
    message['translated'] = translated
    return message

if __name__ == '__main__':
    message = {
                "data": [
                    {
                    "from": "zh",
                    "to": "en",
                    "texts": [
                        "最近如何我的朋友？",
                        "你看起来不错嘛。",
                    ],
                    "translated": [
                        [
                        "How are you my friend?"
                        ],
                        [
                        "You look good."
                        ]
                    ]
                    }
                ]
                }
    data_all = message['data']
    for data in data_all:
        translate(data)
    print(message)
        

