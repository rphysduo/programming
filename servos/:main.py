from machine import Pin, PWM
import sys
import select

center = 150  
max_amp = 45

servo0 = PWM(Pin(11))
servo1 = PWM(Pin(12))

servo0.freq(50)
servo1.freq(50)
#50hz pwm

def set_servo(pwm, angle):
    angle = max(0, min(300, angle))
    pulse_us = 500 + (angle / 300) * 2000
    duty = int(pulse_us * 65535 / 20000)
    pwm.duty_u16(duty)

set_servo(servo0, center)
set_servo(servo1, center)
poll = select.poll() #this waits for serial input
poll.register(sys.stdin, select.POLLIN) #this wakes up the pico when stdin arrives

while True:
    if poll.poll(10): #checks for new data every 10ms, and if it exists, process it
        line = sys.stdin.readline().strip()

        try:
            offset = float(line)

            offset = max(-max_amp, min(max_amp, offset))


            set_servo(servo0, center + offset)
            set_servo(servo1, center - offset)

            print("okay")

        except:
            print("error")
