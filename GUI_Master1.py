import tkinter as tk
from PIL import Image, ImageTk
import csv  # comma seperated value file that uses a comma to seperate values.
from datetime import date  # Used to get the accurate time and date
import time
import numpy as np  # use to perform mathematical operations
import cv2  # python library used to work with computer vision
# fuction return the file name that we selected
from tkinter.filedialog import askopenfilename
import os  # python provides fun for interacting with operation systems
from tkinter import messagebox as ms  # used to display the message box
import shutil  # copy content of source file to destination file
#from skimage import measure
import Train_FDD_cnn as TrainM
global fn  # golable function fn

# ==============================================================================
root = tk.Tk()
root.state('zoomed')

root.title("Fake Image Video Detection System")

current_path = str(os.path.dirname(os.path.realpath('__file__')))

basepath = current_path + "\\"

# ==============================================================================
# ==============================================================================
#Setting the background image

bg_img = ImageTk.PhotoImage(Image.open("wneural.jpg"))
bg_lbl = tk.Label(root, image=bg_img)
bg_lbl.place(x=0, y=0)

# Set the heading label
heading = tk.Label(root, text="Fake Image Video Detection System", width=53, height=2, font=("Arial", 35, "bold"),
                   bg="sky blue", fg="dark blue")
heading.place(x=0, y=0)

# Function to update the label
def update_label(text):
    result_label = tk.Label(root, text=text, width=50, font=("Arial", 25), bg="sky blue", fg="dark blue")
    result_label.place(x=400, y=400)

def train_model():

    update_label("Model Training Start...............")

    start = time.time()

    X = TrainM.main()

    end = time.time()

    ET = "Execution Time: {0:.4} seconds \n".format(end-start)

    msg = "Model Training Completed.."+'\n' + X + '\n' + ET

    update_label(msg)


def show_FDD_video(video_path):
    ''' Display FDD video with annotated bounding box and labels '''
    from keras.models import load_model

   
    img_cols, img_rows = 64, 64

   
    FALLModel = load_model(r'fake_event.h5')

    video = cv2.VideoCapture(video_path)

 

    if (not video.isOpened()):
        print("{} cannot be opened".format(video_path))
       
    font = cv2.FONT_HERSHEY_SIMPLEX
    green = (0, 255, 0)
    red = (0, 0, 255)
 
    line_type = cv2.LINE_AA
    i = 1
    f = 0
    n = 0
    while True:
        ret, frame = video.read()

        if not ret:
            break
        img = cv2.resize(frame, (img_cols, img_rows), fx=0,
                         fy=0, interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.array(img)

        X_img = img.reshape(-1, img_cols, img_rows, 1)
        X_img = X_img.astype('float32')

        X_img /= 255

        predicted = FALLModel.predict(X_img)

        if predicted[0][0] < 0.5:
            predicted[0][0] = 0
            predicted[0][1] = 1
            label = 1
        else:
            predicted[0][0] = 1
            predicted[0][1] = 0
            label = 0

        frame_num = int(i) 
        label_text = ""

        color = (255, 255, 255)

        if label == 1:
            label_text = "Fake Image Detected"
            color = red
            f += 1
        else:
            label_text = "Normal Image Detected"
            color = green
            n += 1

        frame = cv2.putText(
            frame, "Frame: {}".format(frame_num), (5, 30),
            fontFace=font, fontScale=1, color=color, lineType=line_type
        )
        frame = cv2.putText(
            frame, "Label: {}".format(label_text), (5, 60),
            fontFace=font, fontScale=1, color=color, lineType=line_type
        )
        frame = cv2.putText(
            frame, "Default Threshold Value: {}".format(0.5), (5, 90),
            fontFace=font, fontScale=1, color=color, lineType=line_type
        )
        frame = cv2.putText(
            frame, "Predicted Threshold Value : {}".format(
                predicted[0][0]), (5, 130),
            fontFace=font, fontScale=1, color=color, lineType=line_type
        )

        i = i+1
        cv2.imshow('FDD', frame)
        if cv2.waitKey(30) == 27:
            break

    video.release()
    cv2.destroyAllWindows()
    fk = (f/(f+n))*100
    format_fk = "{:.2f}".format(fk)
    ms.showinfo("Output", "Video is "+str(format_fk)+"% fake")


def Video_Verify():


    fileName = fn
    Sel_F = fileName.split('/').pop()
    Sel_F = Sel_F.split('.').pop(1)

    if Sel_F != 'mp4':
        print("Select Video File!!!!!!")
    else:

        

        

        show_FDD_video(fileName)
        # run_video(fn,520, 190,400,500)
# ============================================================================================================


def upload():

    global fn

    fileName = askopenfilename(initialdir='/dataset', title='Select image',
                               filetypes=[("all files", "*.*")])

    fn = fileName
    Sel_F = fileName.split('/').pop()
    Sel_F = Sel_F.split('.').pop(1)

    if Sel_F != 'mp4':
        print("Select Video File!!!!!!")
        ms.showerror('Oops!', 'Select Video File!!!!!!')
    else:
        ms.showinfo('Success!', 'Video Uploaded Successfully !')
        return fn


def convert():
   
    cam = cv2.VideoCapture(fn)
    try:

       
        if not os.path.exists('images'):
            os.makedirs('images')

    # if not created then raise error
    except OSError:
        print('Error: Creating directory of images')

    # frame
    currentframe = 0

    while(True):                

        # reading from frame
        ret, frame = cam.read()

        if ret:
            # if video is still left continue creating images
            name = './images/frame' + str(currentframe) + '.jpg'
            print('Creating...' + name)

            # writing the extracted images
            cv2.imwrite(name, frame)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break


# =============destroy all windows and detele all data and show the results
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()
    ms.showinfo('Success!', 'Video converted into frames Successfully !')


def CLOSE():
    root.destroy()


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# ==========buttons styling ==========
button5 = tk.Button(root, command=upload, text="Upload Video", width=20, font=("Arial", 25, "bold"),
                    bg="sky blue", fg="dark blue")
button5.place(x=500, y=200)

button1 = tk.Button(root, command=convert, text="Convert Video To Frames", width=20, font=("Arial", 25, "bold"),
                    bg="sky blue", fg="dark blue")
button1.place(x=500, y=300)

button2 = tk.Button(root, command=Video_Verify, text="Detect Fake Video", width=20, font=("Arial", 25, "bold"),
                    bg="sky blue", fg="dark blue")
button2.place(x=500, y=400)

close = tk.Button(root, command=CLOSE, text="Exit", width=20, font=("Arial", 25, "bold"), bg="red", fg="white")
close.place(x=500, y=500)


root.mainloop()
