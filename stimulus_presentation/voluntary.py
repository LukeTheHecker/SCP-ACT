from psychopy import visual, core, monitors
import numpy as np
import time
from funs import *

# Tiuming stuff
time_to_target = 5  # 60 * 6.5
end_of_experiment_time = 2  # 20


#create a window
mywin = visual.Window([1920, 1080], monitor='testMonitor', units="deg", fullscr=True)

# Prepare the objects
# Countdown
countdown = visual.TextStim(win=mywin, text='')
timer = core.CountdownTimer(time_to_target)


while timer.getTime() > -1:  # after 5s will become negative
    # print(timer.getTime())
    countdown_time = np.clip(timer.getTime(), a_min=0, a_max=9999999)
    countdown.text = seconds_to_hh_mm_ss(int(round(countdown_time)))
 
    countdown.draw()
    mywin.flip()
    core.wait(0.01)

mywin.flip()
core.wait(1)

mywin.flip()
core.wait(end_of_experiment_time)

