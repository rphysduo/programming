import serial, time, math

serial_connection = serial.Serial('/dev/ttyACM0', 115200)
time.sleep(2)


amp = 25
freq = 0.7 #hz


while True:
    current_time = time.time()
    offset = amp * math.sin(math.pi * 2 * freq * current_time)
    serial_connection.write(f"{offset}\n".encode()) #send over bits through serial to be able to write to servos later
    time.sleep(0.1)
