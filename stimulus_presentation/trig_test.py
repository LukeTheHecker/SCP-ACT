from funs import *
import time

ser = initialize_serial_port()
delay = 0.2

trig(ser, TRIGGER_CODES["2"])

time.sleep(delay)
trig(ser, TRIGGER_CODES["4"])

time.sleep(delay)
trig(ser, TRIGGER_CODES["6"])

time.sleep(delay)
trig(ser, TRIGGER_CODES["8"])

time.sleep(delay)
trig(ser, TRIGGER_CODES["10"])

time.sleep(delay)
trig(ser, TRIGGER_CODES["12"])