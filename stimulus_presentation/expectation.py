from psychopy import visual, core, event
from funs import *
from texts import text_expectation as instruction
from texts import text_end


# Circle parameters:
radius = 4
lineWidth = 1e10
edges = 2000

snitch_color = 'gold'
# Timing stuff
time_to_snitch = 7 * 60
dur_snitch = 1
dur_circle = 1

end_of_experiment_time = 2  # 20

# Trigger setting
ser = initialize_serial_port()

trig_expstart = TRIGGER_CODES['2']
trig_expend = TRIGGER_CODES['4']
trig_snitch = TRIGGER_CODES['6']

#create a window
mywin = visual.Window([1920, 1080], monitor='testMonitor', units="deg", fullscr=True)
mywin.mouseVisible = False
# Prepare the objects

# Snitch
snitch = visual.Circle(win=mywin, radius=radius, lineColor=snitch_color, lineWidth=lineWidth, edges=edges)

# Instruction
show_instruction(mywin, instruction)

# Countdown
fixation = visual.TextStim(win=mywin, text='+')
timer = core.CountdownTimer(time_to_snitch)

# Trigger to show start of experiment
trig(ser, trig_expstart)

RespEvent = event.getKeys()
while timer.getTime() > -1 and not 'q' in RespEvent:  # after 5s will become negative
    fixation.draw()
    mywin.flip()
    core.wait(0.01)
    RespEvent = event.getKeys()

mywin.flip()
# core.wait(0.5)


# Draw the target Circle
snitch.draw()
mywin.flip()
# Trigger to show snitch onset
trig(ser, trig_snitch)

core.wait(dur_circle)

mywin.flip()
core.wait(end_of_experiment_time)
# Trigger to show snitch onset
trig(ser, trig_expend)

show_instruction(mywin, text_end)
