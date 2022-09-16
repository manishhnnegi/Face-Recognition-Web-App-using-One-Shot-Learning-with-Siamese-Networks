import cv2
import streamlit as st
import cv2
import os
import time

from predict import Classify

# Set paths of directories

INPUT_PATH = os.path.join('application_data', 'input_image')


def main():
    """Face Recognition App"""


    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Face Recognition Web Application</h2>
    </div>
    </body>
    <body style="background-color:red;">
    <div >
    <h3 style="color:white;text-align:center;"></h3>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)



    run = st.checkbox('**Start Webcam to Capture Face**')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    st.info('Please click the button to Capture Face') 
    brk= st.button('Click')
    while run:
        _, frame = camera.read()
        frame = cv2.resize(frame,(250,250))
        
        frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame1)
        path = os.path.join(INPUT_PATH,"input_image.jpg")
        cv2.imwrite(path,frame)
        if brk:
            st.success("image captured successfully!!! Please Wait for the Verification Result!!!")
            
            break
        
    #st.write('Verification Result')  
             
    try:
        filename = "application_data\input_image\input_image.jpg"
        classifier = Classify(filename)
        result = classifier.recognition()
        os.remove(filename)
        st.success("Verification Result!!!")
        st.text(result)
    except:
        st.write("************************************************")
        

    


if __name__ == '__main__':
    main()


