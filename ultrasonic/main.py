import serial

ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
#wait 1 second to collect data, time out if nothing
while True:
    line = ser.readline().decode('utf-8').strip()
    print(line)
    #create string out of bytes, strip usueless stuff
    #if line:
        #try:
            #d1, d2, d3 = map(int, line.split(','))
            #creates 3 ints from string of numbers separated by commas
            #print("Sensor distances:", d1, d2, d3)
        #except:
            #pass
