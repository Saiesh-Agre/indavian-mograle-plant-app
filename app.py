import streamlit as st
import pandas as pd
import boto3
import tempfile
import cv2
import os
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import logging
from datetime import datetime

# Constants
BUCKET_NAME = st.secrets["BUCKET_NAME"]
INPUT_PREFIX = "Input"
OUTPUT_PREFIX = "Output"

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# S3 client setup
try:
    s3 = boto3.client(
        "s3",
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name=st.secrets["AWS_REGION"],
    )
except Exception as e:
    st.error("Failed to initialize S3 client.")
    logger.exception("S3 client initialization error: %s", e)
    st.stop()

st.set_page_config(page_title="Indavian Mograle Plant Dashboard", layout="wide")
st.title("Indavian Mograle Plant Dashboard")

# Utility: List S3 files
def list_s3_files(prefix, suffixes=(".mp4", ".avi", ".mov")):
    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
        files = [os.path.basename(obj["Key"]) for obj in response.get("Contents", [])
                 if obj["Key"].lower().endswith(suffixes) and not obj["Key"].endswith("/")]
        return files
    except Exception as e:
        logger.error("Error listing files from S3: %s", e)
        return []

# Utility: Download file from S3 (still used for thumbnails)
def download_s3_file(key):
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        s3.download_file(BUCKET_NAME, key, tmp.name)
        return tmp.name
    except Exception as e:
        logger.error("Error downloading file from S3: %s", e)
        return None

# Utility: Generate pre-signed URL
def generate_presigned_url(key, expiration=3600):
    try:
        return s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': BUCKET_NAME, 'Key': key, 'ResponseContentType': 'video/mp4'},
            ExpiresIn=expiration
        )
    except Exception as e:
        logger.error("Error generating presigned URL: %s", e)
        return None

# Utility: Extract video thumbnail
def get_video_thumbnail(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
    except Exception as e:
        logger.warning("Error extracting thumbnail: %s", e)
    return None

# # UI - Input video selection
# st.sidebar.header("Input Video Selection")
# input_videos = list_s3_files(INPUT_PREFIX)
# selected_input = st.sidebar.selectbox("Choose an input video", input_videos)





# List and filter input videos by selected date
input_videos = list_s3_files(INPUT_PREFIX)

# Extract unique dates from filenames
video_date_map = {}
for filename in input_videos:
    try:
        date_str = filename.split("_")[0]  # '2025-07-21'
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        video_date_map.setdefault(date_obj, []).append(filename)
    except Exception as e:
        logger.warning(f"Skipping file (invalid date format): {filename} | Error: {e}")

available_dates = sorted(video_date_map.keys(), reverse=True)

# Sidebar: Date filter
st.sidebar.subheader("Filter by Date")
selected_date = st.sidebar.date_input("Select date", value=available_dates[0] if available_dates else datetime.today().date())

# Filtered videos for selected date
filtered_videos = video_date_map.get(selected_date, [])
if not filtered_videos:
    st.sidebar.warning("No videos available for selected date.")
selected_input = st.sidebar.selectbox("Choose an input video", filtered_videos) if filtered_videos else None




# UI - Show selected video
st.subheader("Selected Input Video")

if selected_input:
    input_key = f"{INPUT_PREFIX}/{selected_input}"
    input_url = generate_presigned_url(input_key)
    if input_url:
        st.video(input_url)
    else:
        st.error("Could not generate URL for input video.")

    # Detection log + clips
    selected_name = os.path.splitext(selected_input)[0]
    detection_prefix = f"{OUTPUT_PREFIX}/{selected_name}"
    #log_key = f"{detection_prefix}/detection_csv/detection_log.csv"
    log_key = f"{detection_prefix}/detection_csv/detection_log.csv"
    print("LOG KEY : ", log_key)
    clips_prefix = f"{detection_prefix}/video_clips"
    print("CLIPS PREFIX : ", clips_prefix)

    # Detection log loading
    try:
        log_obj = s3.get_object(Bucket=BUCKET_NAME, Key=log_key)
        df_log = pd.read_csv(log_obj["Body"])
    except Exception as e:
        logger.warning("Detection log not found: %s", e)
        st.warning("Detection log not found in S3. Run the processing script first.")
        st.stop()

    # Saved clips grid
    st.subheader("Saved Clips")
    clip_files = list_s3_files(clips_prefix)

    if clip_files:
        num_cols = 3
        for i in range(0, len(clip_files), num_cols):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                if i + j < len(clip_files):
                    clip_file = clip_files[i + j]
                    clip_key = f"{clips_prefix}/{clip_file}"
                    presigned_url = generate_presigned_url(clip_key)
                    with cols[j]:
                        if presigned_url:
                            st.video(presigned_url)
    else:
        st.info("No clips saved for the selected target class.")

    # Pie chart
    st.subheader("Detection Proportions by Class")
    try:
        class_counts = df_log['class'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
        st.write("**Total Detections:**", len(df_log))
    except Exception as e:
        logger.warning("Failed to generate pie chart: %s", e)

    # Detection table
    st.subheader("Detection Log Table")
    try:
        if 'latitude' not in df_log.columns:
            df_log['latitude'] = 12.9716
        if 'longitude' not in df_log.columns:
            df_log['longitude'] = 77.5946
        if 'video_link' not in df_log.columns:
            df_log['video_link'] = f"s3://{BUCKET_NAME}/{clips_prefix}/{clip_files[0]}" if clip_files else "N/A"
        st.dataframe(df_log[['class', 'video_link', 'timestamp', 'latitude', 'longitude']])
    except Exception as e:
        logger.error("Error displaying detection table: %s", e)
        st.error("Could not load detection log table.")
else:
    st.info("Please select a video with available detections.")


st.markdown("---")
st.write("*Dashboard generated by Streamlit.*")
