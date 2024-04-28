import os
from os import path
class SaveReturn:
    def __init__(self, directory, filename):
        self.filename = os.path.join(directory, filename)
        self.data = []
    def save(self, episode, reward):
        self.data.append((episode, reward))
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        with open(self.filename, 'w') as f:
            for episode, reward in self.data:
                f.write(f"{episode}, {reward}\n")