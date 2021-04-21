from psychopy import visual, core, monitors
import numpy as np
import time
from funs import *

# Circle parameters:
radius = 4
lineWidth = 1e10
edges = 2000
main_color = 'white'
snitch_color = 'gold'
show_snitch = True
# Tiuming stuff
time_to_target = 5  # 60 * 6.5
time_to_snitch = 2
dur_snitch = 1
dur_circle = 1

end_of_experiment_time = 2  # 20


#create a window
mywin = visual.Window([1920, 1080], monitor='testMonitor', units="deg", fullscr=True)

# Prepare the objects

# Circle
# https://www.psychopy.org/api/visual/circle.html#psychopy.visual.circle.Circle
circle = visual.Circle(win=mywin, radius=radius, lineColor=main_color, lineWidth=lineWidth, edges=edges)

# Snitch
snitch = visual.Circle(win=mywin, radius=radius, lineColor=snitch_color, lineWidth=lineWidth, edges=edges)

# Countdown
countdown = visual.TextStim(win=mywin, text='')
timer = core.CountdownTimer(time_to_target)


while timer.getTime() > -1:  # after 5s will become negative
    # print(timer.getTime())
    countdown_time = np.clip(timer.getTime(), a_min=0, a_max=9999999)
    countdown.text = seconds_to_hh_mm_ss(int(round(countdown_time)))
    # countdown.draw()
    if show_snitch and timer.getTime() < time_to_target - time_to_snitch:
        show_snitch = False
        snitch.draw()
        mywin.flip()
        core.wait(dur_snitch)
    else:
        countdown.draw()
        mywin.flip()
        core.wait(0.01)

mywin.flip()
core.wait(2)


# Draw the target Circle
circle.draw()
mywin.flip()
core.wait(dur_circle)

mywin.flip()
core.wait(end_of_experiment_time)

