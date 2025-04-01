import streamlit as st
import cv2
import sqlite3
import math
import numpy as np
from datetime import timedelta
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

# --- Database name ---
DB_NAME = "drone_security.db"

# --- Database Initialization ---
def init_db(db_name=DB_NAME):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            yolo_label TEXT,
            color TEXT,
            confidence REAL,
            blip_caption TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            alert_message TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS telemetry (
            telemetry_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            latitude REAL,
            longitude REAL,
            altitude REAL
        )
    ''')
    conn.commit()
    conn.close()

# Initialize DB on app startup
init_db()

# --- Load Models (cached so they load only once) ---
@st.cache_resource(show_spinner=False)
def load_models():
    yolo_model = YOLO("yolov8n.pt")  # Change to 'yolov8s.pt' if desired
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return yolo_model, blip_processor, blip_model

yolo_model, blip_processor, blip_model = load_models()

# --- Helper Functions ---
def approximate_color_name(r, g, b):
    color_map = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "white": (255, 255, 255),
        "black": (0, 0, 0),
        "gray": (128, 128, 128),
        "silver": (192, 192, 192)
    }
    
    min_dist = float("inf")
    chosen_color = "unknown"
    for cname, (cr, cg, cb) in color_map.items():
        dist = math.sqrt((r - cr)**2 + (g - cg)**2 + (b - cb)**2)
        if dist < min_dist:
            min_dist = dist
            chosen_color = cname
    return chosen_color

def get_box_color(frame, box):
    x1, y1, x2, y2 = map(int, box)
    h, w, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    
    if x2 <= x1 or y2 <= y1:
        return "unknown"
    
    crop = frame[y1:y2, x1:x2]
    mean_bgr = crop.reshape(-1, 3).mean(axis=0)
    b, g, r = mean_bgr
    return approximate_color_name(r, g, b)

def caption_object_with_blip(frame, box):
    x1, y1, x2, y2 = map(int, box)
    h, w, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    
    if x2 <= x1 or y2 <= y1:
        return "N/A"
    
    crop = frame[y1:y2, x1:x2]
    pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    
    inputs = blip_processor(images=pil_image, return_tensors="pt")
    output_ids = blip_model.generate(**inputs, max_new_tokens=30)
    caption = blip_processor.decode(output_ids[0], skip_special_tokens=True)
    return caption

def simulate_telemetry_data(frame_seconds):
    base_lat = 37.7749
    base_lon = -122.4194
    base_alt = 100.0
    lat = base_lat + (frame_seconds * 0.0001)
    lon = base_lon + (frame_seconds * 0.0001)
    alt = base_alt + (frame_seconds * 0.01)
    return lat, lon, alt

# --- Video Analysis & Logging ---
def analyze_video(video_path, db_name=DB_NAME, skip_frames=30):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error: Could not open video {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_index = 0
    relevant_labels = {"car", "truck", "bus", "motorcycle"}
    
    # Clear previous detections (optional) – comment out if you want to accumulate results
    # c.execute("DELETE FROM detections")
    # c.execute("DELETE FROM alerts")
    # c.execute("DELETE FROM telemetry")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_time_seconds = int(frame_index / fps)
        video_timestamp = str(timedelta(seconds=frame_time_seconds))
        
        lat, lon, alt = simulate_telemetry_data(frame_time_seconds)
        c.execute('''
            INSERT INTO telemetry (timestamp, latitude, longitude, altitude)
            VALUES (?, ?, ?, ?)
        ''', (video_timestamp, lat, lon, alt))
        
        if frame_index % skip_frames == 0:
            results = yolo_model.predict(frame, conf=0.3)
            if len(results) > 0:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    label = results[0].names[cls_id]
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0]
                    
                    if label in relevant_labels:
                        color_name = get_box_color(frame, xyxy)
                        obj_caption = caption_object_with_blip(frame, xyxy)
                        
                        c.execute('''
                            INSERT INTO detections (timestamp, yolo_label, color, confidence, blip_caption)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (video_timestamp, label, color_name, conf, obj_caption))
                        
                        st.write(f"Logs: {color_name} {label} (conf={conf:.2f}) | BLIP: {obj_caption} | Time: {video_timestamp} | Telemetry: lat={lat:.5f}, lon={lon:.5f}, alt={alt:.2f}")
                        
                        if label == "car" and color_name == "silver":
                            alert_message = (f"ALERT: Silver car detected at {video_timestamp} (lat={lat:.5f}, lon={lon:.5f}, alt={alt:.2f})")
                            c.execute('''
                                INSERT INTO alerts (timestamp, alert_message)
                                VALUES (?, ?)
                            ''', (video_timestamp, alert_message))
                            st.error(alert_message)
        
        frame_index += 1
    
    conn.commit()
    conn.close()
    cap.release()
    st.success("Video analysis complete.")

# --- Query Functions ---
def query_detections_by_label(label, db_name=DB_NAME):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    query = """
        SELECT detection_id, timestamp, yolo_label, color, confidence, blip_caption 
        FROM detections 
        WHERE yolo_label = ?
    """
    c.execute(query, (label,))
    rows = c.fetchall()
    conn.close()
    return rows

def query_detections_by_color(color, db_name=DB_NAME):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    query = """
        SELECT detection_id, timestamp, yolo_label, color, confidence, blip_caption 
        FROM detections 
        WHERE color = ?
    """
    c.execute(query, (color,))
    rows = c.fetchall()
    conn.close()
    return rows

def query_alerts(db_name=DB_NAME):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute("SELECT alert_id, timestamp, alert_message FROM alerts")
    rows = c.fetchall()
    conn.close()
    return rows

def query_telemetry(db_name=DB_NAME):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute("SELECT telemetry_id, timestamp, latitude, longitude, altitude FROM telemetry")
    rows = c.fetchall()
    conn.close()
    return rows

# --- LangChain (Gemini) Functions ---
# Set your API key as needed
os.environ["GOOGLE_API_KEY"] = "AIzaSyBDS-dAWUkBr8Figh-Jro_sgY6q8nKSb60" 

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.4)

def summarize_video_with_gemini(frame_descriptions: list):
    joined_frames = "\n".join(frame_descriptions)
    prompt = (
        f"Here is a video described frame by frame:\n{joined_frames}\n\n"
        "Generate a 1-sentence summary of the video."
    )
    response = llm.invoke(prompt)
    return response.content.strip()

def ask_question_on_video(frames: list, user_question: str):
    context = "\n".join(frames)
    prompt = (
        f"Here is the frame-by-frame description of a video:\n{context}\n\n"
        f"Now answer the following question based on the video: {user_question}"
    )
    response = llm.invoke(prompt)
    return response.content.strip()

# --- Streamlit App Layout ---
st.title("Drone Security Analyst Agent")

st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Choose the app mode:", 
                            ["Video Analysis", "View Logs & Queries", "Video Summary & Q&A"])

if app_mode == "Video Analysis":
    st.header("Video Analysis")
    st.write("Upload a video file to run analysis.")
    
    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    skip_frames = st.number_input("Skip Frames (for processing speed)", min_value=1, value=30)
    
    if st.button("Run Video Analysis"):
        if video_file is not None:
            # Save the uploaded video temporarily
            video_path = "temp_video.mp4"
            with open(video_path, "wb") as f:
                f.write(video_file.read())
            analyze_video(video_path, skip_frames=skip_frames)
            os.remove(video_path)
        else:
            st.error("Please upload a video file.")
            
elif app_mode == "View Logs & Queries":
    st.header("Database Logs & Queries")
    query_option = st.selectbox("Select Query Option", 
                                ["Detections by Label", "Detections by Color", "Alerts", "Telemetry"])
    
    if query_option == "Detections by Label":
        label = st.text_input("Enter label (e.g., car, truck)", value="car")
        if st.button("Query Detections by Label"):
            rows = query_detections_by_label(label)
            st.write(rows)
    elif query_option == "Detections by Color":
        color = st.text_input("Enter color (e.g., silver)", value="silver")
        if st.button("Query Detections by Color"):
            rows = query_detections_by_color(color)
            st.write(rows)
    elif query_option == "Alerts":
        if st.button("View Alerts"):
            rows = query_alerts()
            st.write(rows)
    elif query_option == "Telemetry":
        if st.button("View Telemetry"):
            rows = query_telemetry()
            st.write(rows)
            
elif app_mode == "Video Summary & Q&A":
    st.header("Video Summary & Q&A")
    
    # Fetch frame descriptions from the DB
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT blip_caption FROM detections ORDER BY detection_id ASC")
    rows = cursor.fetchall()
    conn.close()
    frames = [row[0] for row in rows if row[0]]
    
    if st.button("Generate Video Summary"):
        if frames:
            summary = summarize_video_with_gemini(frames)
            st.write("📽️ **Video Summary:**", summary)
        else:
            st.error("No frame descriptions found. Run video analysis first.")
    
    st.subheader("Ask a Question")
    question = st.text_input("Enter your question about the video", value="What objects were seen in the video?")
    if st.button("Get Answer"):
        if frames:
            answer = ask_question_on_video(frames, question)
            st.write("🤖 **Response:**", answer)
        else:
            st.error("No frame descriptions found. Run video analysis first.")
