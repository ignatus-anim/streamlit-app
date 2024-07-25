import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from textinfo import homeinfo, aboutinfo

# Function to load TFLite model and solution JSON based on selected crop
def load_model_and_solution(crop):
    model_file = f"models/{crop}.tflite"
    solution_file = f"solutions/{crop}_solution.json"

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    # Load crop diseases solutions
    with open(solution_file, 'r') as solutions:
        crop_diseases = json.load(solutions)

    return interpreter, crop_diseases

# Function to predict disease based on selected crop
def model_prediction(image_path, interpreter):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0).astype(np.float32)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_arr)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    result_index = np.argmax(prediction)
    return result_index, prediction

# Function to display confidence
def check_confidence(prediction):
    confidence = round((float(np.max(prediction)) * 100), 2)
    return confidence

# Function to display warnings for low confidence
def display_warning(confidence):
    if confidence < 80:
        st.warning("Confidence is below 80%. Please upload a clear image.")
        return True
    return False

# Function to attend to the crop
def attender(test_image, interpreter, crop_diseases):
    if st.button("Show Image"):
        if test_image is not None:
            st.image(test_image, use_column_width=True)
        else:
            st.write("Please upload an image first")

    if st.button("Predict"):
        if test_image is not None:
            result_index, prediction = model_prediction(test_image, interpreter)
            confidence = check_confidence(prediction)
            if not display_warning(confidence):
                disease_name = crop_diseases[result_index]["name"]
                st.success(f"Model is Predicting, It is  {disease_name}")
                st.success(f"Confidence = {confidence}%")
        else:
            st.write("Please upload an image first")

    if st.button("Show Cause"):
        if test_image is not None:
            result_index, prediction = model_prediction(test_image, interpreter)
            confidence = check_confidence(prediction)
            if not display_warning(confidence):
                disease_cause = crop_diseases[result_index]["causes"]
                st.write("Causes include:")
                for cause in disease_cause:
                    st.success(cause)
        else:
            st.write("Please upload an image first")

    if st.button("Recommend Solution"):
        if test_image is not None:
            result_index, prediction = model_prediction(test_image, interpreter)
            confidence = check_confidence(prediction)
            if not display_warning(confidence):
                disease_solution = crop_diseases[result_index]["recommended_solutions"]
                st.write("Some Recommended solutions include:")
                for solution in disease_solution:
                    st.success(solution)

    if st.button("Recommend Pesticide"):
        if test_image is not None:
            result_index, prediction = model_prediction(test_image, interpreter)
            confidence = check_confidence(prediction)
            if not display_warning(confidence):
                disease_pesticide = crop_diseases[result_index]["recommended_pesticide"]
                st.write("Some Recommended solutions include:")
                for pesticide in disease_pesticide:
                    st.success(pesticide)

def main():
    # SideBar
    st.sidebar.title("Dashboard")
    app_mode = st.sidebar.selectbox("Select Mode", ["Home", "Disease_Prediction", "About"])

    # HomePage
    if app_mode == "Home":
        st.header("AgriDetect Crop Disease Detection")
        image_path = 'photos/crops.jpeg'
        st.image(image_path, use_column_width=True)
        st.markdown(homeinfo)

    # About Page
    elif app_mode == "About":
        st.header("About")
        st.markdown(aboutinfo)

    # Disease Prediction Page
    elif app_mode == 'Disease_Prediction':
        st.title('Disease Prediction')
        st.write('Select a crop to predict diseases.')

        # Crop selection
        crop = st.selectbox('Select Crop', ['Corn','Tomato','Pepper'])
        
        # Load model and solution JSON based on selected crop
        interpreter, crop_diseases = load_model_and_solution(crop.lower())

        if interpreter is None or crop_diseases is None:
            st.stop()

        st.write(f'You selected {crop}. Displaying {crop} disease prediction...')
        st.header(f"{crop} Disease Recognition")
        st.image(f'photos/{crop.lower()}.jpeg', use_column_width=True)
         
        # Choice between camera and file uploader
        choice = st.radio("Choose input method", ("Upload Image", "Use Camera"))

        if choice == "Upload Image":
            test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
            if test_image:
                attender(test_image=test_image, interpreter=interpreter, crop_diseases=crop_diseases)
        elif choice == "Use Camera":
            captured_image_path = st.camera_input("Take a picture")
            if captured_image_path:
                attender(test_image=captured_image_path, interpreter=interpreter, crop_diseases=crop_diseases)

# Entry point of the application
if __name__ == "__main__":
    main()
