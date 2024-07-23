# Crop Disease Recognition System

This project is a Crop Disease Recognition System that uses deep learning models to predict diseases in various crops from images. The application is designed to help farmers and agricultural experts identify diseases in crops and recommend solutions.

## Features

- Predict crop diseases from images using trained models.
- Display confidence levels of predictions.
- Provide information about the causes of diseases.
- Recommend solutions and pesticides for identified diseases.
- Use data augmentation techniques for better model performance.
- Support for multiple crops including Corn, Tomato, Pepper, and more.

## Technologies Used

- **Frontend**: Streamlit for a user-friendly web interface.
- **Machine Learning**: TensorFlow for training and running the deep learning models.
- **Image Processing**: PIL for image handling and augmentation.
- **Data Storage**: JSON files for storing disease information and solutions.

## Installation

### Prerequisites

- Python 3.7 or higher
- Virtual environment (optional but recommended)

### Clone the Repository

```sh
git clone https://github.com/yourusername/crop-disease-recognition.git
cd crop-disease-recognition
```

### Create and Activate Virtual Environment

```sh
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### Install Dependencies

```sh
pip install -r requirements.txt
```

### Directory Structure

Ensure your directory structure looks like this:

```
crop-disease-recognition/
│
├── models/
│   ├── corn.tflite
│   ├── tomato.tflite
│   └── ... (other crop models)
│
├── solutions/
│   ├── corn_solution.json
│   ├── tomato_solution.json
│   └── ... (other crop solutions)
│
├── photos/
│   ├── corn.jpeg
│   ├── tomato.jpeg
│   └── ... (other crop images)
│
├── app.py
├── textinfo.py
├── requirements.txt
└── README.md
```

### Run the Streamlit App

```sh
streamlit run app.py
```

## Usage

1. Open the Streamlit app in your browser.
2. Select the crop you want to predict.
3. Upload an image of the crop or use the camera input.
4. Click on "Predict" to see the disease prediction, confidence level, causes, recommended solutions, and recommended pesticides.

## Project Structure

- **app.py**: Main file to run the Streamlit application.
- **textinfo.py**: Contains information for the Home and About pages.
- **models/**: Directory containing the trained models in `.tflite` format.
- **solutions/**: Directory containing JSON files with disease information and solutions.
- **photos/**: Directory containing images for the crops.

## Contributing

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.


## Contact

For any inquiries, please contact [Ignatus Anim](mailto:ignatusa3@gmail.com).
