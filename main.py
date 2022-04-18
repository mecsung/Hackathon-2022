import cv2
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from datetime import datetime
from tkinter import messagebox
import requests
from bs4 import BeautifulSoup

result = ""

# Object detector constant
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3

# Colors for BBOX
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
# Defining fonts
FONTS = cv2.FONT_HERSHEY_COMPLEX

# Load COCO Dataset
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')


# Load SSD Algorithm and trained weights
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def createwidgets():
    root.feedlabel = Label(root, text="Camera", font=('',20))
    root.feedlabel.grid(row=1, column=1, padx=10, pady=10, columnspan=2)

    root.cameraLabel = Label(root, borderwidth=3, relief="groove")
    root.cameraLabel.grid(row=2, column=1, padx=10, pady=10, columnspan=2)

    root.captureBTN = Button(root, text="Capture", command=Capture, font=('',15), width=20)
    root.captureBTN.grid(row=4, column=1, padx=10, pady=10)

    root.previewlabel = Label(root, text="Preview", font=('',20))
    root.previewlabel.grid(row=1, column=4, padx=10, pady=10, columnspan=2)

    root.imageLabel = Label(root, borderwidth=3, relief="groove")
    root.imageLabel.grid(row=2, column=4, padx=10, pady=10, columnspan=2)

    ShowFeed()


def ShowFeed():
    ret, frame = root.cap.read()

    data = object_detector(frame)
    for d in data:
        if d[0] == 'banana':
            result = 'banana'
            cal = fetch_calories(result)
            if cal:
                draw_label(frame, cal[0] + " and " + cal[1] + " sugar", (20, 50))
                draw_label(frame, cal[2] + " carbs and " + cal[3] + " fat", (20, 70))
                cv2.putText(frame, "Food: " + result, (20, 90),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0))
                print(cal)
        elif d[0] == 'orange':
            result = 'orange'
            cal = fetch_calories(result)
            if cal:
                draw_label(frame, cal[0] + " and " + cal[1] + " sugar", (20, 50))
                draw_label(frame, cal[2] + " carbs and " + cal[3] + " fat", (20, 70))
                cv2.putText(frame, "Food: " + result, (20, 90),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0))
                print(cal)

    cv2.putText(frame, datetime.now().strftime('Date: %d/%m/%Y Time: %H:%M:%S'), (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0))
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    videoImg = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image = videoImg)

    root.cameraLabel.configure(image=imgtk)
    root.cameraLabel.imgtk = imgtk
    root.cameraLabel.after(10, ShowFeed)


# Object Detection Function
def object_detector(img):
    classes, scores, boxes = net.detect(img, CONFIDENCE_THRESHOLD)
    # Creating empty list to add objects data
    data_list = []
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]

        if classid == 52:  # banana class id
            cv2.rectangle(img, box, color, 2)
            cv2.rectangle(img, (box[0] - 1, box[1] - 28), (box[0] + 150, box[1]), color, -1)
            cv2.putText(img, classNames[classid - 1], (box[0], box[1] - 10), FONTS, 0.5, (255, 255, 255), 1)
            data_list.append([classNames[classid - 1], box[2], (box[0], box[1] - 2)])

        elif classid == 55:  # orange class id
            cv2.rectangle(img, box, color, 2)
            cv2.rectangle(img, (box[0] - 1, box[1] - 28), (box[0] + 150, box[1]), color, -1)
            cv2.putText(img, classNames[classid - 1], (box[0], box[1] - 10), FONTS, 0.5, (255, 255, 255), 1)
            data_list.append([classNames[classid - 1], box[2], (box[0], box[1] - 2)])

    return data_list


# Search
def fetch_calories(prediction):
    url_cal = 'https://www.google.com/search?&q=calories in ' + prediction
    req = requests.get(url_cal).text
    scrap = BeautifulSoup(req, 'html.parser')
    calories = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text

    url_sug = 'https://www.google.com/search?&q=sugar in ' + prediction
    req = requests.get(url_sug).text
    scrap = BeautifulSoup(req, 'html.parser')
    sugar = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text

    url_carb = 'https://www.google.com/search?&q=carbs in ' + prediction
    req = requests.get(url_carb).text
    scrap = BeautifulSoup(req, 'html.parser')
    carb = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text

    url_fat = 'https://www.google.com/search?&q=fat in ' + prediction
    req = requests.get(url_fat).text
    scrap = BeautifulSoup(req, 'html.parser')
    fat = scrap.find("div", class_="BNeawe iBp4i AP7Wnd").text

    return calories, sugar, carb, fat


# Draw Labels
def draw_label(img, text, pos):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = (0, 0, 0)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)


def Capture():
    image_name = datetime.now().strftime('%d-%m-%Y %H-%M-%S')

    image_path = "./pictures_taken/"
    imgName = image_path + image_name + ".jpg"
    ret, frame = root.cap.read()

    # Date and time
    cv2.putText(frame, datetime.now().strftime('%d/%m/%Y %H:%M:%S'), (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0))
    # Bounding Box
    object_detector(frame)
    # Nutrients
    data = object_detector(frame)
    for d in data:
        if d[0] == 'banana':
            result = 'banana'
            cal = fetch_calories(result)
            if cal:
                draw_label(frame, cal[0] + " and " + cal[1] + " sugar", (20, 50))
                draw_label(frame, cal[2] + " carbs and " + cal[3] + " fat", (20, 70))
                cv2.putText(frame, "Food: " + result, (20, 90),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0))
                print(cal)
        elif d[0] == 'orange':
            result = 'orange'
            cal = fetch_calories(result)
            if cal:
                draw_label(frame, cal[0] + " and " + cal[1] + " sugar", (20, 50))
                draw_label(frame, cal[2] + " carbs and " + cal[3] + " fat", (20, 70))
                cv2.putText(frame, "Food: " + result, (20, 90),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0))
                print(cal)

    success = cv2.imwrite(imgName, frame)
    saved_image = Image.open(imgName)
    saved_image = ImageTk.PhotoImage(saved_image)
    root.imageLabel.config(image=saved_image)
    root.imageLabel.photo = saved_image
    if success :
        messagebox.showinfo("Success", "Image saved!")


root = tk.Tk()
root.cap = cv2.VideoCapture(0)
width, height = 640, 480
root.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
root.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

root.title("FoodPrint")
root.geometry("1340x700")
root.resizable(True, True)
root.configure(background = "white")

createwidgets()
root.mainloop()