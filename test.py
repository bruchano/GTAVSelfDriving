import torch
import torchvision
import cv2
import numpy as np
import time
from PIL import Image, ImageGrab
import keyboard
import mouse
import pyautogui
import matplotlib.pyplot as plt
from random import shuffle
from model import *
from press import PressKey, ReleaseKey

W = 0x11
S = 0x1F
D = 0x20
A = 0x1E

X = 580
Y = 750
WIDTH = 135
HEIGHT = 80

X = 640
Y = 800
WIDTH = 20
HEIGHT = 10
COLOR = (168, 84, 243)
screen_position = (X, Y, X + WIDTH, Y + HEIGHT)


# screen_position = (X, Y, X + WIDTH, Y + HEIGHT)
s = ImageGrab.grab(screen_position)
s = np.array(s)
s = torch.from_numpy(s)
print(s)
t = (s != torch.tensor(COLOR)).all()
print(t)
s = torch.from_numpy(s).unsqueeze(0)
s = s.type(torch.float)

# cv2.imshow("s", s)
# cv2.waitKey()

# while 1:
#     if keyboard.is_pressed("q"):
#         print("break")
#         break
#     x, y = mouse.get_position()
#     print(pyautogui.pixel(x, y))
