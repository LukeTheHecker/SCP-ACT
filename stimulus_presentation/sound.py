from psychopy.sound.backend_ptb import SoundPTB
import time

def play_audio(path):
    so = SoundPTB(value=path)
    so.play()
    while so.status != 'FINISHED':
        time.sleep(2)



path = r'C:\Users\Lukas\Documents\projects\act_scp\stimulus_presentation\DRMT.wav'
play_audio(path)