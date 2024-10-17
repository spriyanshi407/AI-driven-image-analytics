# AI-driven-image-analytics


This project utilizes *Azure Vision API* and *Azure Face API* to analyze images uploaded by users and detect objects, people, captions, dense captions, and tags. Additionally, it uses Azure Face API to detect and display age and gender information for detected faces.

## **Features**
Upload an image for analysis using Streamlit.
Analyze the image for:
Captions and dense captions.
Tags and objects.
Detected people with bounding boxes.
Detect faces in the image using Azure Face API and display:
Age and gender information.
Bounding boxes around detected faces.
Interactive user interface built with Streamlit for uploading and displaying analysis results.

## **Technologies Used**
Azure Vision API: For image analysis including captions, tags, objects, and people detection.
Azure Face API: For face detection and extracting face attributes such as age and gender.
Streamlit: For building the interactive web interface.
Python Libraries:
azure.ai.vision
azure.cognitiveservices.vision.face
Pillow (PIL)
dotenv for environment variable management.
io for handling image data streams.

## **Setup and Installation**
**Prerequisites**
Azure Account: You need to have an Azure account with a valid subscription and the following Azure services:

Azure Vision API.
Azure Face API.
API Keys & Endpoints: Set up Azure Cognitive Services for both the Vision and Face APIs. You will need:

Azure Vision API key and endpoint.
Azure Face API key and endpoint.

## **How It Works**
**Upload Image:** The user uploads an image, which is sent to Azure services for analysis.
**Image Analysis:**
1. Azure Vision API extracts features such as captions, dense captions, tags, objects, and people.
2. Detected objects and people are highlighted on the image using bounding boxes.
**Face Detection:***
1. Azure Face API detects faces and extracts attributes like age and gender.
2. The faces are highlighted with bounding boxes, and age and gender information is displayed alongside the image.
**Results Display:** The processed image and analysis results are displayed in the Streamlit web interface.

## **Error Handling**
The application gracefully handles errors such as HTTP errors or issues with image processing by displaying relevant error messages in the UI.
Python: Ensure that you have Python 3.7+ installed on your machine.

## **Future Enhancements**
Add support for more face attributes like emotion detection.
Provide more in-depth analysis of objects and people detected in images.
Enhance the UI for better user experience.
