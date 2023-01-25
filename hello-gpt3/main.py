import openai
import config
openai.api_key = config.OPENAI_API_KEY

prompt="""quais são as bibliotecas de deep learning mais comuns?
exiba uma lista:
"""

response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    temperature=0.5,
    top_p=1.0,
    max_tokens=500)

response = response.choices[0]['text'].strip()
print(response)
