import serial
import time 

h7 = serial.Serial(port='/dev/ttyS0', baudrate=115200, timeout=.1) 
print("connected")

while h7.isOpen():
    h7.write((bytes("left",  'utf-8')))
    print("write")
    time.sleep(1)