from psychopy import visual, core, event
from funs import *
from texts import text_rest as instruction
from texts import text_end

# Timing stuff
time_to_target = 60 * 7

# Trigger setting
ser = initialize_serial_port()




trig_expstart = TRIGGER_CODES['2']
trig_expend = TRIGGER_CODES['4']

#create a window
mywin = visual.Window([1920, 1080], monitor='testMonitor', units="deg", fullscr=True)
mywin.mouseVisible = False


show_instruction(mywin, instruction)


# Prepare the objects
# Countdown
fixation = visual.TextStim(win=mywin, text='+')
timer = core.CountdownTimer(time_to_target)



# Trigger to show start of experiment
trig(ser, trig_expstart)

RespEvent = event.getKeys()
while timer.getTime() > -1 and not 'q' in RespEvent:  # after 5s will become negative
    fixation.draw()
    mywin.flip()
    core.wait(0.01)
    RespEvent = event.getKeys()



trig(ser, trig_expend)

show_instruction(mywin, text_end)

