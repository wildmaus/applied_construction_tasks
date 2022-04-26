import tkinter as tk
from tkinter.font import BOLD
from PIL import ImageTk, Image
from tkinter import filedialog
from tkinter import ttk
import cv2

def uploadImageFile():
    global img
    global scale
    imPath = filedialog.askopenfilename(
        title = "Select image",
        initialdir = "images", 
        filetypes = (("JPG File","*.jpg"), ("All File Types","*.*")) 
    )

    if imPath:
        img = Image.open(imPath)
        scale = img.size[0] / 320
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)))
        img.save("interface/input.jpg")
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.pack()


def uploadTemplFile():
    global templ
    imPath = filedialog.askopenfilename(
        title = "Select template", 
        initialdir = "templates",
        filetypes = (("JPG File","*.jpg"), ("All File Types","*.*")) 
    )

    if imPath:
        templ = Image.open(imPath)
        print_scale = templ.size[0] / 200
        templ_save = templ.resize((int(templ.size[0] / scale), int(templ.size[1] / scale)))
        templ_save.save("interface/template.jpg")
        templ = ImageTk.PhotoImage(templ.resize((int(templ.size[0] / print_scale), int(templ.size[1] / print_scale))))
        templ_label.config(image=templ)
        templ_label.pack()


def calculateResult():
    global result
    input = cv2.imread("interface/input.jpg", 0)
    template = cv2.imread("interface/template.jpg", 0)

    w, h = template.shape[::-1]

    # Apply template Matching
    res = cv2.matchTemplate(input, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    input = cv2.imread("interface/input.jpg", 1)
    cv2.rectangle(input, top_left, bottom_right, (0,255, 0), 2)
    cv2.imwrite("interface/result.jpg", input)

    result = ImageTk.PhotoImage(Image.open("interface/result.jpg"))
    res_label.config(image=result)
    res_label.pack()


def methodChange(event):
    global method
    method = eval("cv2." + chooseMethod.get())
    calculateResult()



def App(appName:str, WIDTH:int, HEIGHT:int):
    global img_label
    global templ_label
    global res_label
    global method
    global chooseMethod
    
    window = tk.Tk()
    window.config(background="#f8e1fc")
    window.columnconfigure([0, 1, 2], minsize=320, weight=1)
    window.rowconfigure(0, minsize=50, weight=1)
    window.rowconfigure(1, minsize=426, weight=1)
    window.rowconfigure(2, minsize=5, weight=1)

    window.title(appName)
    
    window.geometry(str(WIDTH)+'x'+str(HEIGHT))
    
    # input frame/label/btn
    img_frame = tk.Frame(width=320, height=426, relief=tk.RAISED, borderwidth=5)
    img_frame.grid(row=1, column=0, padx=10)
    img_frame.config(background="#deecff")
    img_label = tk.Label(img_frame)
    btn = tk.Button(window, text=f"Upload image", command=uploadImageFile, background="#deecff")
    btn.grid(row=2, column=0, pady=5, sticky="n")

    # template frame/label/btn
    templ_frame = tk.Frame(width=200, height=234, relief=tk.RAISED, borderwidth=5)
    templ_frame.grid(row=1, column=1, padx=5)
    templ_frame.config(background="#deecff")
    templ_label = tk.Label(templ_frame)
    btn = tk.Button(window, text=f"Upload template", command=uploadTemplFile, background="#deecff")
    btn.grid(row=2, column=1, pady=5, sticky="n")

    # result frame/label/btn
    res_frame = tk.Frame(width=320, height=426, relief=tk.RAISED, borderwidth=5)
    res_frame.grid(row=1, column=2, padx=10)
    res_frame.config(background="#deecff")
    res_label = tk.Label(res_frame)    
    btn = tk.Button(window, text=f"Calculate result", command=calculateResult, background="#deecff")
    btn.grid(row=2, column=2, pady=5, sticky="n")
    
    # method
    method = cv2.TM_CCOEFF
    method_frame = tk.Frame(window)
    method_frame.config(background="#f8e1fc")
    method_frame.grid(row=0, column=2)
    chooseMethod = ttk.Combobox(method_frame,   
                            values=[
                                'TM_CCOEFF', 
                                'TM_CCOEFF_NORMED', 
                                'TM_CCORR',
                                'TM_CCORR_NORMED',
                                'TM_SQDIFF',
                                'TM_SQDIFF_NORMED'
                            ],
                            state="readonly",
                            font=BOLD,
                            background="#deecff")
    text = tk.Label(method_frame, text="Choose functional:", background="#f8e1fc", font=BOLD)
    text.pack()
    chooseMethod.pack()
    chooseMethod.current(0)
    chooseMethod.bind("<<ComboboxSelected>>", methodChange)

    window.mainloop()

App('Template matcing', 1000, 540)
