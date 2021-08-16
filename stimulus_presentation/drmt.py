from datetime import datetime
from psychopy import visual, core, monitors, event
from funs import *
from texts import text_drmt as instruction
from texts import text_end
# Audio
path_drmt = r'C:\Users\Lukas\Documents\projects\SCP-ACT\stimulus_presentation\assets\DRMT_ohne_Stille.wav'
path_silence = r'C:\Users\Lukas\Documents\projects\SCP-ACT\stimulus_presentation\assets\Stille_und_Ton.wav'
# path = r'C:\Users\Lukas\Documents\projects\SCP-ACT-main\stimulus_presentation\assets\drmt.wav'

# Timing stuff
time_to_target = 60 * 7

# Trigger setting
ser = initialize_serial_port()
trig_expstart = TRIGGER_CODES['2']
trig_expend = TRIGGER_CODES['4']
trig_silence = TRIGGER_CODES['6']

#create a window
mywin = visual.Window([1920, 1080], monitor='testMonitor', units="deg", fullscr=True)
mywin.mouseVisible = False

# Instruction
show_instruction(mywin, instruction)

fixation = visual.TextStim(win=mywin, text='+')
timer = core.CountdownTimer(time_to_target)

# Trigger to show start of audio
trig(ser, trig_expstart)
# Play DRMT session
print(f'PLAY DRMT TALK: {datetime.now().strftime("%H:%M:%S")}')
play_audio(path_drmt)

# Trigger to show end of audio
trig(ser, trig_silence)

# Play silence and final sound
print(f'PLAY SILENCE AND BELL: {datetime.now().strftime("%H:%M:%S")}')
play_audio(path_silence)
trig(ser, trig_expend)
print(f'Done playing: {datetime.now().strftime("%H:%M:%S")}')

show_instruction(mywin, text_end)
