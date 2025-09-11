import streamlit as st
import pandas as pd
import boto3
import pymysql
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
import logging
import streamlit.components.v1 as components

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit setup with dark theme CSS
st.set_page_config(page_title="Indavian Mograle Plant Dashboard", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1e1e2f;
        color: #ffffff;
    }

    .tile {
        background-color: #252639;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 0 10px rgba(0,0,0,0.3);
    }

    .tile h2, .tile h3 {
        color: #ffffff !important;
        margin-top: 0;
    }

    label, .stSelectbox label, .stDateInput label {
        color: white !important;
    }

    div[data-baseweb="select"], .stDateInput, input, textarea {
        background-color: #2c2f4a !important;
        color: white !important;
        border: 1px solid #444 !important;
    }

    .stDataFrameContainer, .stDataFrame {
        background-color: #1e1e2f !important;
        color: white !important;
    }

    .matplotlib-figure {
        background-color: transparent !important;
    }

    .section-divider {
        margin: 30px 0;
        border-top: 2px solid #444;
    }
    </style>
    """, unsafe_allow_html=True
)

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

# Load jobs
df_jobs = get_successful_jobs()
if df_jobs.empty:
    st.warning("No processed videos found in the database.")
    st.stop()

# --- Top Filters ---
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
df_jobs['upload_date'] = pd.to_datetime(df_jobs['upload_timestamp']).dt.date
available_dates = sorted(df_jobs['upload_date'].unique(), reverse=True)

col1, col2 = st.columns(2)

with col1:
    selected_date = st.date_input(
        label="Filter by Date",
        value=available_dates[0] if available_dates else datetime.today().date(),
        key="date_picker"
    )

filtered_df = df_jobs[df_jobs['upload_date'] == selected_date]
if filtered_df.empty:
    st.warning("No videos found for selected date.")
    st.stop()

with col2:
    selected_file = st.selectbox("Select Video", filtered_df['file_name'].tolist())

selected_row = filtered_df[filtered_df['file_name'] == selected_file].iloc[0]

# Parse S3 keys and URLs
input_url = generate_presigned_url(selected_row['s3_video_key'])
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

# List clips
def list_clip_files(prefix):
    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
        return [obj['Key'] for obj in response.get("Contents", []) if obj["Key"].endswith(".mp4")]
    except Exception as e:
        logger.error("Failed to list clips: %s", e)
        return []
clip_keys = list_clip_files(clips_prefix)

# --- Dashboard Layout ---
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    st.subheader("Original Video")
    if input_url:
        st.video(input_url)
    else:
        st.error("Could not generate URL for input video.")

with row1_col2:
    st.subheader("Detection Proportions by Class")
    try:
        class_counts = df_log['class'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 6), facecolor='#252639')
        wedges, texts, autotexts = ax.pie(
            class_counts,
            labels=class_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'color': "white"}
        )
        ax.axis('equal')

        legend_labels = [f"{cls} = {count}" for cls, count in class_counts.items()]
        ax.legend(
            wedges,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            labelcolor='white',
            frameon=False
        )

        st.pyplot(fig, transparent=True)
        st.markdown(
            f'<div style="text-align:center;">Total Detections: {len(df_log)}</div>',
            unsafe_allow_html=True
        )
    except Exception as e:
        logger.warning("Failed to generate pie chart: %s", e)

# Divider
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("Saved Clips")

    if clip_keys:
        video_html = '<div style="max-height:600px; overflow-y:auto; padding:10px; border:1px solid #444; border-radius:10px; background-color:#252639; display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 10px;">'

        for i, key in enumerate(clip_keys):
            url = generate_presigned_url(key)
            if url:
                video_html += f'''
                <div style="">
                    <video width="100%" height="160" controls>
                        <source src="{url}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>
                </div>
                '''

        video_html += '</div>'
        components.html(video_html, height=600)
    else:
        st.info("No clips available.")

with row2_col2:
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

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.write("*Dashboard powered by RDS + Streamlit.*")
