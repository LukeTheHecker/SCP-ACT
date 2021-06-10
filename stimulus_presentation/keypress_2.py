import msvcrt
import playsound

path = r'C:\Users\Lukas\Documents\projects\act_scp\stimulus_presentation\DRMT.mp3'
while True:
    if msvcrt.kbhit():   #Checks if any key is pressed
         playsound.playsound(path, True)
    print('Press Key:')