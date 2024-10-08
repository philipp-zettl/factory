from requests import Session
from argparse import ArgumentParser
from urllib.parse import urljoin


class bcolors:
    HEADER = '\033[95m'  #]
    OKBLUE = '\033[94m'  #]
    OKCYAN = '\033[96m'  #]
    OKGREEN = '\033[92m' #]
    WARNING = '\033[93m' #]
    FAIL = '\033[91m'   #]
    ENDC = '\033[0m'    #]
    BOLD = '\033[1m'    #]
    UNDERLINE = '\033[4m' #]


class LLMChat:
    def __init__(self, host: str, prompt: str | None = None, model_name: str = 'phi-3-mini-128k-chat'):
        self.session = Session()
        self.host_url = host
        self.history = [{
            'role': 'system',
            'content': prompt or 'You are a helpful bot having a conversation with a user. You are a senior Data Scientist and support the user in all work related questions and tasks. You answer honest and short. And never lose your temper.'
        }]
        self.model_name = model_name
    
    def send_message(self, message: str):
        self.history.append({'role': 'user', 'content': message})
        res = self.session.post(
            urljoin(urljoin(self.host_url, 'models/'), self.model_name),
            json={
                'inputs': self.history,
                'parameters': {'use_cache': False, 'max_new_tokens': 250}
            }
        )
        res.raise_for_status()
        res = res.json()
        self.history.append({'role': 'assistant', 'content': res[0]['generated_text']})
        return res[0]['generated_text']

    def close(self):
        self.session.close()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='phi-3-mini-128k-chat')
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--host', type=str, default='http://localhost:8001')
    return parser.parse_args()


def main():
    args = parse_args()
    llm = LLMChat(args.host, args.prompt, args.model)
    while True:
        message = input(f'{bcolors.OKBLUE}You: ')
        print(f'{bcolors.ENDC}')
        if message == 'exit':
            llm.close()
            break
        response = llm.send_message(message)
        print(f'{bcolors.OKGREEN}Bot: {response}{bcolors.ENDC}')

if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        print(bcolors.ENDC, end='')
        raise e
