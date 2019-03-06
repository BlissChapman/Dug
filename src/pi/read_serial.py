import argparse
import pygame
import serial

parser = argparse.ArgumentParser(description='Read given serial port.')
parser.add_argument('port', help='serial port (ex: /dev/ttyACM0)')
args = parser.parse_args()

# Open serial port
ser = serial.Serial(args.port, 9600)

# Initialize pygame
pygame.init()
pygame.display.set_caption('Alma Recorder')
screen = pygame.display.set_mode((400, 400))

# Read serial port
stimulus = 0
done = False

while not done:
    
    if ser.in_waiting == 0:
        # No data to read
        continue
    
    # Update stimulus flag
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True 
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            stimulus = 1
        elif event.type == pygame.KEYUP and event.key == pygame.K_SPACE:
            stimulus = 0
            
    # Read data from serial port
    line = ser.readline()

    # Record data and stimulus
    try:
       line_int = int(line)
       print("{0} {1}".format(line_int, stimulus))
    except ValueError:
       continue
