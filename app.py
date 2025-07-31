import streamlit as st
import pandas as pd
import boto3
import pymysql
import tempfile
import cv2
import os
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit setup
st.set_page_config(page_title="Indavian Mograle Plant Dashboard", layout="wide")
st.title("Indavian Mograle Plant Dashboard")

# S3 Setup
try:
    s3 = boto3.client(
        "s3",
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
        region_name=st.secrets["AWS_REGION"],
    )
    BUCKET_NAME = st.secrets["BUCKET_NAME"]
except Exception as e:
    st.error("Failed to initialize S3 client.")
    logger.exception("S3 client initialization error: %s", e)
    st.stop()

# DB Connection
def get_db_connection():
    return pymysql.connect(
        host=st.secrets["DB_HOST"],
        user=st.secrets["DB_USER"],
        password=st.secrets["DB_PASSWORD"],
        database=st.secrets["DB_NAME"],
        port=int(st.secrets.get("DB_PORT", 3306)),
        cursorclass=pymysql.cursors.DictCursor
    )

# Fetch successful jobs
@st.cache_data(show_spinner=False)
def get_successful_jobs():
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT job_id, file_name, upload_timestamp, s3_video_key, s3_output_key
                FROM video_processing_jobs
                WHERE process_status = 'SUCCESS'
                ORDER BY upload_timestamp DESC
            """)
            return pd.DataFrame(cur.fetchall())
    except Exception as e:
        logger.error("Failed to fetch jobs from RDS: %s", e)
        return pd.DataFrame()
    finally:
        conn.close()

# Generate pre-signed S3 URL
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

# Extract thumbnail (optional)
def get_video_thumbnail(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
    except Exception as e:
        logger.warning("Thumbnail error: %s", e)
    return None

# Load jobs
df_jobs = get_successful_jobs()

if df_jobs.empty:
    st.warning("No processed videos found in the database.")
    st.stop()

# Sidebar: Date Filter
df_jobs['upload_date'] = pd.to_datetime(df_jobs['upload_timestamp']).dt.date
available_dates = sorted(df_jobs['upload_date'].unique(), reverse=True)
selected_date = st.sidebar.date_input("Filter by Date", value=available_dates[0] if available_dates else datetime.today().date())
filtered_df = df_jobs[df_jobs['upload_date'] == selected_date]

# Sidebar: Video Selection
if filtered_df.empty:
    st.sidebar.warning("No videos found for selected date.")
    st.stop()

selected_file = st.sidebar.selectbox("Select Video", filtered_df['file_name'].tolist())
selected_row = filtered_df[filtered_df['file_name'] == selected_file].iloc[0]

# Show selected video
st.subheader("Selected Input Video")
input_url = generate_presigned_url(selected_row['s3_video_key'])
if input_url:
    st.video(input_url)
else:
    st.error("Could not generate URL for input video.")

# Parse output path for detections
output_prefix = selected_row['s3_output_key'].rstrip("/")
log_key = f"{output_prefix}/detection_csv/detection_log.csv"
clips_prefix = f"{output_prefix}/video_clips"

# Load detection log
try:
    log_obj = s3.get_object(Bucket=BUCKET_NAME, Key=log_key)
    df_log = pd.read_csv(log_obj['Body'])
except Exception as e:
    logger.warning("Detection log not found: %s", e)
    st.warning("Detection log not found in S3. Run processing first.")
    st.stop()

# Show clips
st.subheader("Saved Clips")
def list_clip_files(prefix):
    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
        return [obj['Key'] for obj in response.get("Contents", []) if obj["Key"].endswith(".mp4")]
    except Exception as e:
        logger.error("Failed to list clips: %s", e)
        return []

clip_keys = list_clip_files(clips_prefix)
if clip_keys:
    num_cols = 3
    for i in range(0, len(clip_keys), num_cols):
        cols = st.columns(num_cols)
        for j in range(num_cols):
            if i + j < len(clip_keys):
                key = clip_keys[i + j]
                url = generate_presigned_url(key)
                if url:
                    with cols[j]:
                        st.video(url)
else:
    st.info("No clips available.")

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
        df_log['video_link'] = f"s3://{BUCKET_NAME}/{clip_keys[0]}" if clip_keys else "N/A"
    st.dataframe(df_log[['class', 'video_link', 'timestamp', 'latitude', 'longitude']])
except Exception as e:
    logger.error("Error displaying detection table: %s", e)
    st.error("Could not load detection log table.")

st.markdown("---")
st.write("*Dashboard powered by RDS + Streamlit.*")
