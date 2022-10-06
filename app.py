import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model_ed = load_model('my_model_ed_final_160')


def isLowLightImage(image):
    short_hist_bgr = [[], [], []]  # to store histogram of each channel
    for i, col in enumerate(['b', 'g', 'r']):  # iterate over bgr channels to find channel mean pixel value
        short_hist_bgr[i] = cv2.calcHist([image], [i], None, [256], [0, 256])
    # find mean pixel histogram from the 3 histograms obtained for each image
    short_hist_avg = [(a + b + c) / 3 for a, b, c in zip(short_hist_bgr[0], short_hist_bgr[1], short_hist_bgr[2])]
    cdf = np.cumsum(short_hist_avg)
    cdf /= max(cdf)
    pc = min(np.argwhere(cdf > 0.95))[0]
    if pc < 100:
        return True
    return False


def predict(image):
    if not isLowLightImage(image):
        return image
    image = image/255.
    output = model_ed.predict(np.array([image]))
    return output


def predict_traditional(image):
    if not isLowLightImage(image):
        return image
    clahe = cv2.createCLAHE(clipLimit=5)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output = clahe.apply(image) + 30
    output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
    return output


def main():
    st.title("Light-up images clicked in the Dark")
    uploaded_file = st.file_uploader("Upload image")
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.resize(opencv_image, (160, 160))
        col1, col2, col3 = st.columns(3)

        with col2:
            img = st.empty()
            img.image(opencv_image, channels="BGR")
#           st.image(opencv_image, channels="BGR", key='img')
            st.button("Light up- AI", key="predict")
            st.button("Light up- Traditional", key="traditional")
            st.button("Show Original", key="ori")
            if st.session_state.predict:
                output = predict(opencv_image)
                img.image(output, clamp=True, channels="BGR")
            if st.session_state.ori:
                img.image(opencv_image, channels="BGR")
            if st.session_state.traditional:
                output = predict_traditional(opencv_image)
                img.image(output, channels="RGB")

if __name__ == '__main__':
    main()