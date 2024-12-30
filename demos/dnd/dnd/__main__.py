from requests import Session
from argparse import ArgumentParser
from urllib.parse import urljoin


class bcolors:
    HEADER = '\033[95m'  #]
    BLUE = '\033[94m'  #]
    CYAN = '\033[96m'  #]
    GREEN = '\033[92m' #]
    OKBLUE = '\033[94m'  #]
    OKCYAN = '\033[96m'  #]
    OKGREEN = '\033[92m' #]
    WARNING = '\033[93m' #]
    FAIL = '\033[91m'   #]
    ENDC = '\033[0m'    #]
    BOLD = '\033[1m'    #]
    UNDERLINE = '\033[4m' #]


class LLMChat:
    def __init__(self, host: str, prompt: str | None = None, model_name: str = 'Qwen'):
        self.session = Session()
        self.host_url = host
        self.history = [{
            'role': 'system',
            'content': prompt
        }]
        self.model_name = model_name

    def send_message(self, message: str):
        self.history.append({'role': 'user', 'content': message})
        res = self.session.post(
            urljoin(urljoin(self.host_url, 'models/'), self.model_name),
            json={
                'inputs': self.history,
                'parameters': {
                    'use_cache': False, 'max_new_tokens': 128, 'temperature': 0.3
                }
            }
        )
        res.raise_for_status()
        res = res.json()
        self.history.append({'role': 'assistant', 'content': res[0]['generated_text']})
        return res[0]['generated_text']

    def chat(self, message: str):
        return self.send_message(message)

    def close(self):
        self.session.close()


def main():
    from time import sleep
    import random

    # Initialize participants as DnD characters
    participants = {
        "Alice": LLMChat(
            host="http://localhost:8001",
            prompt="You are Alice, a bold elven ranger with unmatched archery skills and a deep connection to nature. Speak and act as your character in the campaign, focusing on protecting your friends and the wilderness.",
            model_name="Qwen"
        ),
        "Bob": LLMChat(
            host="http://localhost:8001",
            prompt="You are Bob, a gruff dwarven fighter who prefers a direct approach. You are loyal to your friends but love cracking jokes and taking on challenges head-on. Speak and act as your character.",
            model_name="Qwen"
        ),
        "DungeonMaster": LLMChat(
            host="http://localhost:8001",
            prompt="You are the Dungeon Master, narrating a Dungeons & Dragons campaign. Describe the world vividly, set the stage for challenges, and react to the players' actions. Keep the story engaging and dynamic. Answer in short sentences",
            model_name="Qwen"
        )
    }
    color_map = {
        "Alice": bcolors.OKGREEN,
        "Bob": bcolors.OKBLUE,
        'DungeonMaster': bcolors.WARNING
    }

    # Campaign setup
    conversation_turns = 20  # Number of turns in the campaign
    current_speaker = "DungeonMaster"  # Start with the DM
    message = (
        "You find yourselves in a dimly lit tavern. A mysterious hooded figure approaches your table and speaks: "
        "'Brave adventurers, the kingdom is in peril. Will you accept the quest to retrieve the Crystal of Eternity?'"
    )  # Initial narrative by the DM

    print(f"{current_speaker}: {message}")

    # Synchronize chat histories across all participants
    for participant in participants.values():
        participant.history.append({'role': 'system', 'content': message})

    for turn in range(conversation_turns):
        # Get the current speaker's LLM instance
        speaker_instance = participants[current_speaker]
        color = color_map.get(current_speaker, bcolors.HEADER)

        # Send the message and get the response
        response = speaker_instance.chat(message)
        print(f"{color}{current_speaker}: {response}{bcolors.ENDC}")

        # Update all participants' histories with the new message
        for participant in participants.values():
            participant.history.append({'role': 'user' if current_speaker != 'DungeonMaster' else 'assistant', 'content': message})
            participant.history.append({'role': 'assistant' if current_speaker != 'DungeonMaster' else 'user', 'content': response})

        # If DM, set up responses from players
        if current_speaker == "DungeonMaster":
            next_speaker = random.choice([name for name in participants if name != "DungeonMaster"])
        else:
            # After a player's action, the DM responds
            next_speaker = "DungeonMaster"

        # Set the next message and speaker
        message = response
        current_speaker = next_speaker

        # Optional delay for readability
        sleep(1)

    # Cleanup
    for participant in participants.values():
        participant.close()

if __name__ == "__main__":
    main()
