from pathlib import Path
import random
import pygame


# minPathing is the folder where the songs are stored
# Separate the song between bass, drums, vocals and guitar
def findSongPathing(minPathing: str):
    songsDirectory = Path(minPathing)
    pathing = []

    for folders in songsDirectory.iterdir():
        pathing.append(folders)

    return str(random.choice(pathing))


class GameModule:
    def __init__(self, audioPathing):
        self.audioPath = audioPathing

        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

        self.sounds = [
            pygame.mixer.Sound(self.audioPath + "/other.wav"),
            pygame.mixer.Sound(self.audioPath + "/vocals.wav"),
            pygame.mixer.Sound(self.audioPath + "/drums.wav"),
            pygame.mixer.Sound(self.audioPath + "/bass.wav"),
        ]

        self.channels = [
            pygame.mixer.Channel(0),
            pygame.mixer.Channel(1),
            pygame.mixer.Channel(2),
            pygame.mixer.Channel(3),
        ]

        for i in range(4):
            self.channels[i].play(self.sounds[i], loops=-1)
            self.channels[i].set_volume(0.0)

    def changeVolume(self, volumes):
        for i, vol in enumerate(volumes):
            vol_norm = max(0.0, min(1.0, vol / 100.0))
            self.channels[i].set_volume(vol_norm)
