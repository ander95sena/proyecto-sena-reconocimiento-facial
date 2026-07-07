import serial
import time

serialArduino = serial.Serial("COM3", 9600)
time.sleep(2)  # Espera a que Arduino inicialice


while True:
    # Enviar '1' para encender
    serialArduino.write(bytes([1]))
    time.sleep(1)

    # Enviar '0' para apagar
    serialArduino.write(bytes([0]))
    time.sleep(1)

serialArduino.close()
