import streamlit as st
from dotenv import load_dotenv
import os
from PIL import Image, ImageDraw
from azure.core.credentials import AzureKeyCredential
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.core.exceptions import HttpResponseError
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
import io

# Load environment variables for Azure AI credentials
load_dotenv()
ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
ai_key = os.getenv('AI_SERVICE_KEY')

face_endpoint = os.getenv('FACE_API_ENDPOINT')  # Use a specific Face API endpoint
face_key = os.getenv('FACE_API_KEY')

# Authenticate Azure Face Client
face_client = FaceClient(ai_endpoint, CognitiveServicesCredentials(ai_key))

# Authenticate Azure AI Vision client
cv_client = ImageAnalysisClient(endpoint=ai_endpoint, credential=AzureKeyCredential(ai_key))

# Streamlit page setup
st.title("Azure Vision API Image Analysis with Age and Gender Detection")

# Upload image using Streamlit
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Define a function to analyze the image (Visual Features + Face Attributes)
def AnalyzeImage(image_data):
    st.write("Analyzing image...")

    try:
        # Analyze image using Azure Vision API for general features
        result = cv_client.analyze(
            image_data=image_data,
            visual_features=[
                VisualFeatures.CAPTION,
                VisualFeatures.DENSE_CAPTIONS,
                VisualFeatures.TAGS,
                VisualFeatures.OBJECTS,
                VisualFeatures.PEOPLE],
        )

        # Display analysis results
        if result.caption:
            st.write(f"**Caption**: '{result.caption.text}' (Confidence: {result.caption.confidence * 100:.2f}%)")

        if result.dense_captions:
            st.write("**Dense Captions**:")
            for caption in result.dense_captions.list:
                st.write(f"Caption: '{caption.text}' (Confidence: {caption.confidence * 100:.2f}%)")

        if result.tags:
            st.write("**Tags**:")
            for tag in result.tags.list:
                st.write(f"Tag: '{tag.name}' (Confidence: {tag.confidence * 100:.2f}%)")

        if result.objects:
            st.write("**Objects in the image**:")
            image = Image.open(io.BytesIO(image_data))
            draw = ImageDraw.Draw(image)
            color = 'cyan'

            for detected_object in result.objects.list:
                st.write(f"Object: {detected_object.tags[0].name} (Confidence: {detected_object.tags[0].confidence * 100:.2f}%)")
                r = detected_object.bounding_box
                bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
                draw.rectangle(bounding_box, outline=color, width=3)

            # Display image with objects highlighted
            st.image(image, caption="Objects detected", use_column_width=True)

        if result.people:
            st.write("**People detected in the image**:")
            image = Image.open(io.BytesIO(image_data))
            draw = ImageDraw.Draw(image)
            color = 'cyan'

            for detected_people in result.people.list:
                r = detected_people.bounding_box
                bounding_box = ((r.x, r.y), (r.x + r.width, r.y + r.height))
                draw.rectangle(bounding_box, outline=color, width=3)

            # Display image with people highlighted
            st.image(image, caption="People detected", use_column_width=True)

        # Now use the Face API to detect age and gender of people
        # Now use the Face API to detect age and gender of people
        detected_faces = face_client.face.detect_with_stream(
        io.BytesIO(image_data),  # Corrected: Pass the image as an io.BytesIO object
        return_face_attributes=['age', 'gender']
        )


        if not detected_faces:
            st.write("No faces detected.")

        # Load image for drawing bounding boxes for faces
        image = Image.open(io.BytesIO(image_data))
        draw = ImageDraw.Draw(image)

        # Draw bounding boxes and show age, gender for each face
        for face in detected_faces:
            rect = face.face_rectangle
            bounding_box = ((rect.left, rect.top), (rect.left + rect.width, rect.top + rect.height))
            draw.rectangle(bounding_box, outline='yellow', width=3)

            # Display age and gender
            st.write(f"Detected Person - Age: {face.face_attributes.age}, Gender: {face.face_attributes.gender}")

        # Display image with bounding boxes for age and gender
        st.image(image, caption="Detected faces with Age and Gender", use_column_width=True)

    except HttpResponseError as e:
        st.error(f"Error: {e.reason} - {e.message}")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Run the analysis when the image is uploaded
if uploaded_image is not None:
    image_data = uploaded_image.read()
    AnalyzeImage(image_data)
