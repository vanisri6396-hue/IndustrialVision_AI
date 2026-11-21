
import streamlit as st
import pandas as pd
import numpy as np

from PIL import Image
from modules.detection import load_model, detect_image


# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="IndustrialVision AI",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ----------------------------------
# CUSTOM STYLING
# ----------------------------------

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .dashboard-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
    }

    .dashboard-subtitle {
        color: #aab4c3;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------
# LOAD MODEL
# ----------------------------------

@st.cache_resource
def get_model():
    return load_model()


# ----------------------------------
# SIDEBAR
# ----------------------------------

st.sidebar.title("🏭 IndustrialVision AI")

st.sidebar.caption("Intelligent Industrial Monitoring")

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Image Detection",
        "Video Detection",
        "Webcam Monitoring",
        "Object Tracking",
        "Analytics",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.success("YOLO26 model-based detection")

st.sidebar.caption("Version 1.0 | Development")


# ----------------------------------
# DASHBOARD
# ----------------------------------

if page == "Dashboard":

    st.markdown(
        '<div class="dashboard-title">IndustrialVision AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-subtitle">'
        'Intelligent Industrial Safety & Monitoring Platform'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("Welcome to IndustrialVision AI")

    st.write(
        "A computer vision platform designed to detect, "
        "track and analyze objects in industrial environments."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("AI Model", "YOLO26 Small")

    with col2:
        st.metric("Detection Classes", "80")

    with col3:
        st.metric("Detection Engine", "Active")

    st.divider()

    st.subheader("Application Modules")

    col1, col2 = st.columns(2)

    with col1:
        st.info("📷 Image Detection")
        st.write("Upload images and identify objects.")

        st.info("🎥 Video Detection")
        st.write("Analyze recorded videos frame by frame.")

        st.info("🔍 Object Tracking")
        st.write("Track objects across video frames.")

    with col2:
        st.info("📹 Webcam Monitoring")
        st.write("Perform live object detection.")

        st.info("📊 Analytics")
        st.write("Explore detection metrics and summaries.")

        st.info("🚨 Safety Monitoring")
        st.write("Future module for industrial safety alerts.")


# ----------------------------------
# IMAGE DETECTION
# ----------------------------------

elif page == "Image Detection":

    st.title("📷 Image Object Detection")

    st.write(
        "Upload an image to detect and identify objects using YOLO26."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)

        st.subheader("Original Image")
        st.image(image_array, use_container_width=True)

        if st.button("Run Object Detection", type="primary"):

            with st.spinner("YOLO26 is detecting objects..."):

                model = get_model()

                annotated_image, detections = detect_image(
                    model,
                    image_array,
                    confidence=confidence
                )

            st.success("Detection completed!")

            st.subheader("Detection Results")

            st.image(
                annotated_image,
                caption="YOLO26 Detection Output",
                use_container_width=True
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Objects Detected",
                    len(detections)
                )

            with col2:
                unique_classes = len(
                    set(item["Object"] for item in detections)
                )

                st.metric(
                    "Unique Object Classes",
                    unique_classes
                )

            if detections:

                st.subheader("Detection Details")

                dataframe = pd.DataFrame(detections)

                st.dataframe(
                    dataframe,
                    use_container_width=True,
                    hide_index=True
                )

                csv_data = dataframe.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Detection Report",
                    data=csv_data,
                    file_name="industrialvision_detections.csv",
                    mime="text/csv"
                )

            else:
                st.warning("No objects detected in this image.")


# ----------------------------------
# OTHER MODULES - COMING NEXT
# ----------------------------------

elif page == "Video Detection":

    st.title("🎥 Video Detection")
    st.info(
        "Video upload and annotated video processing "
        "will be integrated in Phase 2."
    )

elif page == "Webcam Monitoring":

    st.title("📹 Live Webcam Monitoring")
    st.info(
        "Local webcam integration will be implemented "
        "in Phase 2."
    )

elif page == "Object Tracking":

    st.title("🔍 Object Tracking")
    st.info(
        "ByteTrack integration will be implemented "
        "in Phase 2."
    )

elif page == "Analytics":

    st.title("📊 Industrial Analytics")
    st.info(
        "Detection history, charts and database-backed "
        "analytics will be implemented in a later phase."
    )

elif page == "About Project":

    st.title("About IndustrialVision AI")

    st.write("""
    IndustrialVision AI is an intelligent computer vision
    platform for industrial monitoring and safety applications.

    Technologies:

    - Python
    - YOLO26
    - OpenCV
    - Streamlit
    - FastAPI (planned)
    - PostgreSQL (planned)
    - Plotly (planned)
    """)
