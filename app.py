import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

# Load the trained YOLO model
model = YOLO("model/best.pt")  # Update the path if necessary

# Define the detect_objects function
def detect_objects(image):
    """
    Perform object detection on the input image using the YOLO model.
    """
    results = model(image)
    predictions = []
    for result in results[0].boxes.data:  # Use .data to access detection details
        x1, y1, x2, y2, conf, cls = result.tolist()
        predictions.append({
            'bounding_box': [x1, y1, x2, y2],
            'confidence': round(conf, 2),
            'class': model.names[int(cls)]  # Map class index to class name
        })
    return predictions

# Streamlit app interface
st.title("Blood Cell Detection")
st.write("Upload an image to detect blood cells such as RBCs, WBCs, and Platelets.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform detection
    predictions = detect_objects(np.array(image))

    # Annotate image with bounding boxes
    annotated_image = Image.fromarray(np.array(image))
    draw = ImageDraw.Draw(annotated_image)
    for pred in predictions:
        x1, y1, x2, y2 = map(int, pred['bounding_box'])
        label = f"{pred['class']} ({pred['confidence']*100:.1f}%)"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), label, fill="red")

    # Display annotated image
    st.image(annotated_image, caption="Annotated Image", use_column_width=True)

    # Display predictions table
    st.write("Detected Objects:")
    for pred in predictions:
        st.write(f"- **Class:** {pred['class']}, **Confidence:** {pred['confidence']}, **Bounding Box:** {pred['bounding_box']}")
