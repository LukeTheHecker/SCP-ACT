import time
from psychopy import visual, core, data, event, logging, sound, gui, info
from psychopy.sound.backend_ptb import SoundPTB

import serial

PULSE_WIDTH = 0.01

TRIGGER_CODES = {
    "2": [0x02],
    "4": [0x04],
    "6": [0x06],
    "8": [0x08],
    "10": [0x0A],
    "12": [0x0C]
}

def show_instruction(win, instruction):
    text = visual.TextStim(win=win, text=instruction, height=0.7, wrapWidth=90)
    text.draw()
    win.flip()
    RespEvent = event.getKeys()
    while not 'space' in RespEvent:
        # win.flip()
        RespEvent = event.getKeys()
    text = visual.TextStim(win=win, text='')
    text.draw()
    win.flip()

def seconds_to_hh_mm_ss(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    if seconds >= 3600:
        return f"{h:d}:{m:02d}:{s:02d}"
    else:
        return f"{m:d}:{s:02d}"


def trig(ser, signal):
    ''' Send a signal through a serial port.
    
    Parameters
    ----------
    ser : serial.Serial
        The serial port object
    signal : bytecode/int
        The signal to send through the serial port
    
    Return
    ------
    '''
    ser.write([0x00])
    ser.write(signal)
    time.sleep(PULSE_WIDTH)
    ser.write([0x00])


def initialize_serial_port(port='COM3'):
    ''' Initialize the serial port connection
    Parameters
    ----------
    port : str
        Defines the COM port.
    '''
    ser = serial.Serial(port)
    ser.write([0x00])
    ser.write([0xFF])
    
    return ser


def play_audio(path):
    ''' Play audio located in path.
    
    Parameters
    ----------
    path : str
        Path to the audio file. Must be of type WAVE (.wav)
    
    Return
    ------


    '''
    so = SoundPTB(value=path)
    so.play()
    RespEvent = event.getKeys()
    while so.status != -1 and not 'q' in RespEvent:
        time.sleep(0.25)
        RespEvent = event.getKeys()
        # break
    so.stop()