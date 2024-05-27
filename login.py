import sqlite3
import tkinter as tk
from tkinter import *
import tkinter.messagebox
from PIL import Image, ImageTk

root = tk.Tk()
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Login Form")

Name = StringVar()
upass = StringVar()

# Connect to the SQLite database
my_conn = sqlite3.connect('evaluation.db')

root.configure(background="seashell2")

# For background Image
image2 = Image.open('wneural.jpg')
image2 = image2.resize((1530, 900), Image.ANTIALIAS)
background_image = ImageTk.PhotoImage(image2)
background_label = tk.Label(root, image=background_image)
background_label.image = background_image
background_label.place(x=0, y=0)


def login_now():
    username = Name.get()
    password = upass.get()

    # Verify the login credentials
    cursor = my_conn.cursor()
    cursor.execute("SELECT * FROM registration WHERE username=? AND password=?", (username, password))
    result = cursor.fetchone()

    if result:
        tkinter.messagebox.showinfo("Success", "Login successful")
        from subprocess import call
        call(["python", "GUI_Master1.py"])  
   
   
    else:
        tkinter.messagebox.showerror("Error", "Invalid username or password")


label_0 = Label(root, text="Login Here", width=20, font=("Poppins", 20, "bold"))
label_0.place(x=600, y=100)

label_1 = Label(root, text="User Name", width=20, font=("Poppins", 11, "bold"))
label_1.place(x=550, y=200)

entry_1 = Entry(root, textvar=Name, bg="lightgray", font=("Poppins", 11, "bold"))
entry_1.place(x=800, y=200)

label_2 = Label(root, text="Password", width=20, font=("Poppins", 11, "bold"))
label_2.place(x=550, y=300)

entry_2 = Entry(root, textvar=upass, bg="lightgray", show="*", font=("Poppins", 11, "bold"))
entry_2.place(x=800, y=300)

Button(root, text='Login Now', width=20, font=("Poppins", 11, "bold"), bg='red', fg='white', command=login_now).place(x=680, y=400)

root.mainloop()
