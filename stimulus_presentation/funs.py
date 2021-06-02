from psychopy.sound.backend_ptb import SoundPTB
import time

def seconds_to_hh_mm_ss(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    if seconds >= 3600:
        return f"{h:d}:{m:02d}:{s:02d}"
    else:
        return f"{m:d}:{s:02d}"

def play_audio(path):
    so = SoundPTB(value=path)
    so.play()
    while so.status != 'FINISHED':
        time.sleep(2)