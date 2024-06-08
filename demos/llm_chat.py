from requests import Session


class LLMChat:
    def __init__(self, username: str, password: str):
        self.session = Session()
        self.history = [{
            'role': 'system',
            'content': 'You are a helpful bot having a conversation with a user. You are a senior Data Scientist and support the user in all work related questions and tasks. You answer honest and short. And never lose your temper.'
        }]
    
    def send_message(self, message: str):
        self.history.append({'role': 'user', 'content': message})
        res = self.session.post('http://localhost:8000/models/Qwen2-0.5B-Instruct', json={'inputs': self.history, 'parameters': {'use_cache': False, 'max_new_tokens': 50}})
        res.raise_for_status()
        res = res.json()
        self.history.append({'role': 'assistant', 'content': res[0]['generated_text']})
        return res[0]['generated_text']

    def close(self):
        self.session.close()


if __name__ == '__main__':
    llm = LLMChat('username', 'password')
    while True:
        message = input('You: ')
        response = llm.send_message(message)
        print(f'Bot: {response}')
        if message == 'exit':
            llm.close()
            break
