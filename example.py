'''
source:
https://ollama.com/library/deepseek-r1
available models, e.g.
deepseek-r1:1.5b
deepseek-r1:14b

install via ollama, e.g.:
ollama run deepseek-r1:1.5b
'''

from chat_assist import Client

ai_model = "deepseek-r1:1.5b"
client = Client(host='http://localhost:11434')

def get_response(prompt):
    response = client.generate(model=ai_model, prompt=prompt, stream=False)
    return response["response"]

def stream_response(prompt):
    stream = client.generate(model=ai_model, prompt=prompt, stream=True)
    for chunk in stream:
        print(chunk['response'], end='', flush=True)

error = '''
Traceback (most recent call last):
  File "<pyshell#1>", line 1, in <module>
    a/b
NameError: name 'b' is not defined
'''

prompt = f"How can I solve following error: {error}"


#stream_response(prompt)
print(get_response(prompt))