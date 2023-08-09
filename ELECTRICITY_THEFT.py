from tkinter import *
import tkinter as tk
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd
#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
import itertools
import statsmodels.api as sm
import warnings 
from pylab import rcParams
import requests
rcParams['figure.figsize'] = 20, 8

import os , io 
from google.cloud import vision
from google.cloud.vision import ImageAnnotatorClient
os.environ['GOOGLE_APPLICATION_CREDENTIALS']=r'C:\Users\riyasatam\Downloads\Image_Based_Electricity_Theft_Detection (3)\Image_Based_Electricity_Theft_Detection\sublime-lyceum-350608-4622864529a6.json'
client = vision.ImageAnnotatorClient()



root = tk.Tk()
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("DETECTION")

image2 =Image.open(r'C:\Users\riyasatam\Downloads\Image_Based_Electricity_Theft_Detection (3)\Image_Based_Electricity_Theft_Detection\bg1.jpeg')
image2 =image2.resize((w,h), Image.ANTIALIAS)
background_image=ImageTk.PhotoImage(image2)
background_label = tk.Label(root, image=background_image)
background_label.image = background_image
background_label.place(x=0, y=0)

lbl = tk.Label(root, text="Electricity Theft Detection", font=('Copperplate Gothic Bold', 25,'italic'), height=2, width=30,bg="#FF8040",fg="black")
lbl.place(x=320, y=0)



global result_text


def Choose():
    global file
    file = askopenfilename(initialdir=r'C:\Users\riyasatam\Downloads\Image_Based_Electricity_Theft_Detection (3)\Image_Based_Electricity_Theft_Detection\DATA_SET', title='Select Image',
                                       filetypes=[("all files", "*.jpg*")])
    
    image3 =Image.open(file)
    image3 =image3.resize((350,220), Image.ANTIALIAS)
    
    choosen_image=ImageTk.PhotoImage(image3)
    
    display = tk.Label(root, image=choosen_image)
    
    display.image= choosen_image
    
    display.place(x=250, y=90)
    result_label = tk.Label(root,width=30,height=15,bg='black',fg='white')
    result_label.place(x=650,y=90)
    
    f=file.split("/").pop()
    f=f.split(".").pop(0)
    print(file)
    print(f)   
    #result_label = tk.Label(root,width=10,height=5,bg='black',fg='white')
    #result_label.place(x=900,y=90)
    
    avg = tk.Label(root,width=60,height=18,bg='black',fg='white')
    avg.place(x=900,y=90)
    
    cust=tk.Label(root,text='Customer ID: MAH '+str(f),font=('Times New Roman',20,'italic'),width=20,height=2,bg='black',fg='white')
    cust.place(x=300,y=350)

def detect_text():
    global file
    """Detects text in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()
    #file = r'E:\OMKARS\OMKARS1\M.Tech\Data\china.jpg'
    with io.open(file, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    

    for text in texts:
        print('')
        #print('\n"{}"'.format(text.description))
        
        return text.description
        #print(type(text.description))
        #vertices = (['({},{})'.format(vertex.x, vertex.y)
                    #for vertex in text.bounding_poly.vertices])

        #print('bounds: {}'.format(','.join(vertices)))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))



def Text_Label():
    global result_text
    result_text = detect_text()
    result_label = tk.Label(root,text=str(result_text)+' Units',font=('Times New Roman',30,'italic'),width=10,height=5,bg='black',fg='white')
    result_label.place(x=650,y=90)







def Check():
    global file
    data = pd.read_csv(r"C:\Users\riyasatam\Downloads\Image_Based_Electricity_Theft_Detection (3)\Image_Based_Electricity_Theft_Detection\data1.csv")
    data.dropna()  
    
    
    f=file.split("/").pop()
    f=f.split(".").pop(0)
    print(file)
    print(f)   
    
    y = data[str(f)]
    
    print("average",sum(y)/len(y))
    average = sum(y)/len(y)
    avg = tk.Label(root,text='Average Units\n'+str(average),font=('Times New Roman',30,'italic'),width=19,height=5,bg='black',fg='white')
    avg.place(x=900,y=90)
      
    
def Prediction():
    global file
    global result_text
    print(result_text)
    data = pd.read_csv(r"C:\Users\riyasatam\Downloads\Image_Based_Electricity_Theft_Detection (3)\Image_Based_Electricity_Theft_Detection\data1.csv")
    data.dropna()  
    print(file)
    #file = r"E:\OMKARS\OMKARS1\Electricity Theft\DATA_SET\1.jpg"
    
    
    
    f=file.split("/").pop()
    f=f.split(".").pop(0)
    print(file)
    print(f)   
    
    y = data[str(f)]
    
    
    
    p = d = q = range(0, 2)
    
    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))
    
    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    
    
    warnings.filterwarnings("ignore") # specify to ignore warning messages
    
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
    
                results = mod.fit()
    
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                minAIC[round(results.aic,2)]=(param, param_seasonal)
    
            except:
                continue
            
            
            
    Fparam=((1,0,0),(0,0,0,12))
    print(Fparam)
    
    ny = np.array(y.values)
    mod = sm.tsa.statespace.SARIMAX(ny,
                                    order=Fparam[0],
                                    seasonal_order=Fparam[1],
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    
    results = mod.fit()
    
    print(results.summary().tables[1])
    
    
    # Get forecast 5 steps ahead in future
    pred_uc = results.get_forecast(steps=3)
    # Get confidence intervals of forecasts
    pred_ci = pred_uc.conf_int()
    print(pred_ci[0])
    
    c = pd.DataFrame(pred_ci)
    
    a,b = c[0] , c[1] 
    a,b = pd.DataFrame(a) , pd.DataFrame(b)
    a = list(a[0])
    b = list(b[1])
    print(a,b)
    print("Forecast 1 : ",(a[0]+b[0])/2)
    plot_values = list(y.values)
    plot_values.append((a[0]+b[0])/2)
    plot_values
    #a.extend(b)
    #result_text = 530
    predicted_value = int(plot_values[-1])
    print(predicted_value)
    ans = tk.Label(root,text="Predicted Unit"+str(predicted_value),font=('Times New Roman',25,'italic'),width=30,height=3,bg='black',fg='white')
    ans.place(x=800,y=400)
    print("result_text",result_text)
    if int(result_text) in range(predicted_value-60,predicted_value+60):
        print('Genuine Unit Consumption is Detected')
        gen='Genuine Unit Consumption is Detected'
        ans = tk.Label(root,text=str(gen),font=('Times New Roman',25,'italic'),width=30,height=3,bg='black',fg='white')
        ans.place(x=350,y=600)
    else:
        print('Theft Like Unit Consumption is Detected')
        thf='Theft Like Unit Consumption is Detected'
        def Send():
                import smtplib
                from email.mime.multipart import MIMEMultipart
                from email.mime.text import MIMEText
                from email.mime.base import MIMEBase
                from email import encoders
                
                fromaddr = 'dumymmail@gmail.com' #Mail Id Here
                toaddr = 'riyasatam1@gmail.com' #Mail Id Here
                print('Got mail id')
            
                msg = MIMEMultipart()
            
                msg['From'] = fromaddr
            
                msg['To'] = toaddr
            
                password = 'dumymmail@123' #Password Here
            
                msg['Subject'] = "!!! ALERT !!!"
            
                body = 'ALERT !!! THEFT LIKE UNIT CONSUMPTION IS DETECTED WITH CUSTOMER ID MAH_{}. KINDLY CHECK WHETHER TRUE OR NOT.!'.format(f)
            
                msg.attach(MIMEText(body, 'plain'))
                            
                s = smtplib.SMTP('smtp.gmail.com', 587)
            
                s.starttls()
            
                try:
                    s.login(fromaddr, password)
            
                    text = msg.as_string()
            
                    s.sendmail(fromaddr, toaddr, text)
            
                    print("Mail Sent Successfully !")
            
                except  (smtplib.SMTPException, ConnectionRefusedError, OSError):
            
                    print("Oops Mail Not Sent !")
        

        Send()
        ans = tk.Label(root,text=str(thf),font=('Times New Roman',25,'italic'),width=30,height=2,bg='black',fg='white')
        ans.place(x=350,y=600)


en1 = tk.Entry(root,width=40,font=('Times New Roman',17,'italic'))
en1.insert(0,"Please enter your Customer ID to generate Graph")
en1.place(x=250,y=505)



def Graph():
    import matplotlib.pyplot as plt
    data = pd.read_csv(r"C:\Users\riyasatam\Downloads\Image_Based_Electricity_Theft_Detection (3)\Image_Based_Electricity_Theft_Detection\data1.csv")
    get = en1.get()
    print(get)
    y = data[str(get)]
    
    x=[1,2,3,4,5,6,7,8,9,10,11,12]
    plt.plot(x,y , label="Y-axis=>Units",color='green', linewidth=2,marker='o', markerfacecolor='green', markersize=12)
    plt.xlabel('Months')
    # naming the y axis
    plt.ylabel('Unit Consumption')
    plt.legend()
    plt.show()

def Exit():
    root.destroy()
    
button1 = tk.Button(root,text='Choose Image',command=Choose,font=('Times New Roman',15,'italic'),width=15,bg='black',fg='linen')
button1.place(x=40,y=100)

button2 = tk.Button(root,text="Detect Text",command=Text_Label,font=('Times New Roman',15,'italic'),width=15,bg='black',fg='linen')
button2.place(x=40,y=200)

button3 = tk.Button(root,text="Check",command=Check,font=('Times New Roman',15,'italic'),width=15,bg='black',fg='linen')
button3.place(x=40,y=300)

button4 = tk.Button(root,text="Predict",command=Prediction,font=('Times New Roman',15,'italic'),width=15,bg='black',fg='linen')
button4.place(x=40,y=400)

exit = tk.Button(root,text="Exit",command=Exit,font=('Times New Roman',15,'italic'),width=15,bg='red',fg='linen')
exit.place(x=40,y=600)

graph = tk.Button(root,text="Graph",command = Graph,font=('Times New Roman',15,'italic'),width=15,bg='black',fg='linen')
graph.place(x=40,y=500)

root.mainloop()
