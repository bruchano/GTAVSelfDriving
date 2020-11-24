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


X = 580
Y = 750
WIDTH = 135
HEIGHT = 80
screen_position = (X, Y, X + WIDTH, Y + HEIGHT)

W = 0x11
S = 0x1F
D = 0x20
A = 0x1E

ver = 1
lr = 0.0005
dataset_path = "self_driving_dataset_1.pt"
save_path = f"self_driving_lr_{lr}_ver_{ver}.pt"
load_ver = 1
load_path = f"self_driving_lr_{lr}_ver_{load_ver}.pt"

epoch = 1
loss_count = []


def grab_screen():
    screen = ImageGrab.grab(screen_position)
    screen = np.array(screen).transpose(2, 0, 1)
    screen = torch.from_numpy(screen).unsqueeze(0)
    return screen


def move(output, t=0.1):
    if output[0] == 1:
        ReleaseKey(S)
        PressKey(W)
        if output[2] == 1:
            PressKey(A)
            time.sleep(0.1)
            ReleaseKey(A)
        elif output[3] == 1:
            PressKey(D)
            time.sleep(0.1)
            ReleaseKey(D)
        else:
            time.sleep(t)

    elif output[1] == 1:
        ReleaseKey(W)
        PressKey(S)
        if output[2] == 1:
            PressKey(A)
            time.sleep(0.1)
            ReleaseKey(A)
        elif output[3] == 1:
            PressKey(D)
            time.sleep(0.1)
            ReleaseKey(D)
        else:
            time.sleep(t)

    elif output == [0, 0, 1, 0]:
        ReleaseKey(W)
        ReleaseKey(S)
        PressKey(A)
        time.sleep(t)
        ReleaseKey(A)

    elif output == [0, 0, 0, 1]:
        ReleaseKey(W)
        ReleaseKey(S)
        PressKey(D)
        time.sleep(t)
        ReleaseKey(D)


def check_finished():
    X = 640
    Y = 800
    WIDTH = 20
    HEIGHT = 10
    COLOR = (168, 84, 243)
    screen_position = (X, Y, X + WIDTH, Y + HEIGHT)

    screen = ImageGrab.grab(screen_position)
    screen = np.array(screen)
    if (screen != COLOR).all():
        return True

    return False


def plot_loss():
    plt.figure(1)
    plt.plot(loss_count)
    plt.title("Loss Graph")
    plt.xlabel("loop")
    plt.ylabel("loss")
    plt.show()


def train(dataset_path, load_path=None):
    global loss_count

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    driver = SelfDriving().to(device)
    if load_path:
        driver.load_state_dict(torch.load(load_path))
    driver.train()

    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(driver.parameters(), lr=lr)

    for e in range(epoch):
        print(f"--Epoch {e + 1}--")
        training_dataset = torch.load(dataset_path)
        shuffle(training_dataset)

        for data in training_dataset:
            optimizer.zero_grad()

            screen, target = data
            screen = screen.transpose(2, 0, 1)
            screen = torch.from_numpy(screen).unsqueeze(0).to(device)
            target = torch.tensor(target)

            output = driver(screen)
            loss = mse_loss(output, target)
            loss_count.append(loss.item())
            print("output:", output)
            print("target:", target)
            print("loss:", loss.item())
            print("\n")

            loss.backward()
            optimizer.step()

        torch.save(driver.state_dict(), save_path)

    print("Done")
    plot_loss()


def eval(model_path):
    for i in range(1, 4):
        print(i)
        time.sleep(1)
    print("--Evaluation start--")
    t = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    driver = SelfDriving().to(device)
    driver.load_state_dict(torch.load(model_path))
    driver.eval()

    while True:
        if keyboard.is_pressed("q"):
            print("--break--")
            break

        screen = grab_screen().to(device)

        if check_finished():
            print("arrived required destination")
            print("travel time: %.1fs" % (time.time() - t))
            break

        output = driver(screen)
        print("output:", output)
        output = [(x > 0.8).item() for x in output]
        output = [1 if i == True else 0 for i in output]

        move(output)



