from tkinter import Frame,Label,Tk,Button,PhotoImage,messagebox,CENTER,TOP,BOTTOM,X,FLAT,SOLID
from PIL import ImageTk,Image
from subprocess import Popen
from os import listdir,makedirs
from os.path import join,exists

import numpy as np
from tensorflow.keras.models import load_model
import cv2
from threading import Thread
from skimage.metrics import structural_similarity as ssim
from datetime import datetime
from math import sqrt
import pyttsx3

def mute_music():
    global muted
    
    if muted:
        btn_volume.configure(image=volume_photo)
        muted=False
    else:
        btn_volume.configure(image=mute_photo)
        muted=True

def play_warning(warning):
    if warning == 'incorrect_distancing':
        engine.say("Please wear you mask correctly and practicing social distancing.")
    elif warning == 'no_distancing':
        engine.say("Please wear you mask and practicing social distancing.")
    elif warning == 'incorrect':
        engine.say("Please wear you mask correctly.")
    elif warning == 'no':
        engine.say("Please wear you mask.")
    elif warning == 'distancing':
        engine.say("Please practicing social distancing.")

    engine.runAndWait()
    engine.stop()

def view_database():
    if len(listdir('Database')) == 0:
        messagebox.showerror("Error","There is no data stored in database yet.")
    else:
        Popen(r'explorer "Database\"')

def check_database(idx,current_face_rgb_resized):
    now = datetime.now()

    #today folder
    today_folder_path = join('Database',now.strftime("%d-%m-%Y"))
    if not exists(today_folder_path):
        makedirs(today_folder_path)

    #inside today: class folder
    for cat in category:
        folder_path = join(today_folder_path,cat)
        if not exists(folder_path):
            makedirs(folder_path)
    
    #check if there is a similar image
    class_folder = join(today_folder_path,category[idx])
    files = listdir(class_folder)
    repeat = False
    for file in files:
        img_path = join(class_folder, file)
        img = cv2.imread(img_path,1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if ssim(img,current_face_rgb_resized,multichannel=True) >= 0.3:
            repeat = True
            break

    # save img
    if not repeat:
        file_path = join(class_folder,now.strftime("%H%M%S")+'.jpg')
        current_face_bgr_resized = cv2.cvtColor(current_face_rgb_resized, cv2.COLOR_RGB2BGR)#cv2 works in bgr
        cv2.imwrite(file_path,current_face_bgr_resized)

def check_social_distancing(centroids):
    close = False

    for i,centroid in enumerate(centroids):
        if i == 0:
             continue
        else:
            x1,y1 = centroid
            x2,y2 = centroids[i-1]
            distance = sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance <= 250:
                close = True
    
    if close:
        alert_label.configure(text='Social Distancing: Breach', fg="#fc0303")
    else:
        alert_label.configure(text='Social Distancing: Good', fg="#37eb34")

    return close

def detect_face(frame):
    global th

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()

    centroids = []
    mask = []

    # to draw faces on image
    try:
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                cX = int((x2-x1)/2 + x1)
                cY = int((y2-y1)/2 + y1)
                centroids.append( (cX,cY) )

                # preprocess the face and predict with model
                current_face_rgb = frame[y1:y2,x1:x2].copy()
                current_face_rgb_resized = cv2.resize(current_face_rgb, (50,50))
                current_face_rgb_reshaped = np.array(current_face_rgb_resized).reshape(-1,50,50,3)
                current_face_rgb_reshaped = current_face_rgb_reshaped.astype('float32') / 255.0

                result = model.predict(current_face_rgb_reshaped)[0]
                idx = np.argmax(result)
                mask.append(category[idx])

                # draw bouding box and text
                cv2.rectangle(frame, (x1, y1-30), (x2, y1), (0,0,0), -1)
                if category[idx] == 'With Mask':
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, category[idx], (x1+10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
                    cv2.putText(frame, category[idx], (x1+10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

                # check whether need to save in database (based on similarity of the images)
                check_database(idx,current_face_rgb_resized)

        # check social distancing
        close = check_social_distancing(centroids)

        # whether to sound the alert
        if not th.is_alive() and not muted:
            if close and 'Incorrect Mask' in mask:
                th = Thread(target=play_warning, args=['incorrect_distancing'],daemon = True)
                th.start()
            elif close and 'No Mask' in mask:
                th = Thread(target=play_warning, args=['no_distancing'],daemon = True)
                th.start()
            elif 'Incorrect Mask' in mask:
                th = Thread(target=play_warning, args=['incorrect'],daemon = True)
                th.start()
            elif 'No Mask' in mask:
                th = Thread(target=play_warning, args=['no'],daemon = True)
                th.start()
            elif close:
                th = Thread(target=play_warning, args=['distancing'],daemon = True)
                th.start()
    except cv2.error as e:
        print('Invalid webcam frame!')
    
    return frame

def display_video():
    # Get the latest frame and convert into Image
    frame = cv2.cvtColor(video_capture.read()[1], cv2.COLOR_BGR2RGB)#BGR

    # Convert image to PhotoImage and apply on webcam_label 
    img = Image.fromarray(detect_face(frame))
    imgtk = ImageTk.PhotoImage(image = img)
    webcam_label.imgtk = imgtk
    webcam_label.configure(image=imgtk)

    # Repeat after an interval to capture continiously
    webcam_label.after(10, display_video)

# Create database if no such directory
if not exists('Database'):
    makedirs('Database')

# checking if webcam can be opened
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not video_capture.isOpened():
    messagebox.showerror("Error","Unable to open webcam/camera.")
elif not exists('Models'):
    messagebox.showerror("Error","Please include the Models folder.")
else:
    # declare variables
    font = ("bahnschrift",12)
    title_font = ("bahnschrift",20,"bold")
    muted = False
    category = ['Incorrect Mask','No Mask','With Mask']
    engine = pyttsx3.init()
    th = Thread(target=play_warning, args=[''],daemon = True)

    # read in models
    model = load_model(join('Models','Mask','rgbMaskDetector.h5'))
    net = cv2.dnn.readNetFromCaffe(join('Models','Face',"deploy.prototxt.txt"), join('Models','Face',"res10_300x300_ssd_iter_140000.caffemodel"))

    # prepare root window
    root = Tk()
    root.title('Face Mask Detector')
    root.geometry("800x690")
    root.config(bg="slategrey")
    root.resizable(False, False)

    # prepare frames
    top_frame = Frame(root,bg="slategrey",pady=10)
    top_frame.pack(anchor=CENTER,side=TOP)

    main_frame = Frame(root,bg="slategrey")
    main_frame.pack(fill=X)

    buttom_frame = Frame(root,bg="slategrey",pady=20)
    buttom_frame.pack(anchor=CENTER,side=BOTTOM)

    ##### top frame #####

    title_label = Label(top_frame, font=title_font, text="Face Mask Detector", bg="slategrey", fg="white")
    title_label.pack()

    ##### main frame #####

    webcam_label = Label(main_frame)
    webcam_label.pack()

    alert_label = Label(main_frame, font=title_font, text='Social Distancing: Good', bg="black", fg="#37eb34", pady=10)
    alert_label.pack(fill=X)

    ##### bottom frame #####

    # open database file
    btn_view_database = Button(buttom_frame, font=font, text="View Database", padx=50, pady=10, cursor="hand2", bg="white", relief=FLAT, overrelief=SOLID, command=view_database)#
    btn_view_database.grid(row=0,column=0,padx=20)

    # volumn button
    mute_photo = PhotoImage(file=join('Icons','mute.png'))
    volume_photo = PhotoImage(file=join('Icons','unmute.png'))
    btn_volume = Button(buttom_frame, image=volume_photo, relief=FLAT, overrelief=SOLID, cursor="hand2", command=mute_music)
    btn_volume.grid(row=0,column=1,padx=20)

    display_video()

    # place window at center when opening
    root.eval('tk::PlaceWindow . center')

    root.mainloop()
