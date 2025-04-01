# Drone Security Analyst Agent

## Overview

The **Drone Security Analyst Agent** is an AI-driven prototype designed to enhance property security by processing live drone video feeds and telemetry data. This project leverages state-of-the-art models—**YOLOv8** for object detection and **BLIP** for image captioning—to analyze video frames in real time, generate context-rich logs, and trigger alerts based on predefined rules. Notably, the system is configured to generate a security alert **only when a silver vehicle (specifically, a silver car) is detected**.

### Project Recording - https://drive.google.com/file/d/1QGURKPl1u0-whxwUYNCd7Le8mDg5SA_y/view?usp=sharing

This repository contains two key components:
- **`app.py`**: A complete Streamlit application that integrates all functionality for video analysis, database logging, querying, and AI-powered summarization & Q&A.
- **`Project_Build.ipynb`**: A Jupyter Notebook that documents the step-by-step development process, providing recruiters with insights into the underlying logic, design decisions, and iterative testing that went into the final implementation.

## Feature Specification

The Drone Security Analyst Agent offers the following value to property owners:
- **Automated Monitoring**: Continuously processes drone video feeds to detect security events.
- **Real-Time Alerts**: Generates immediate alerts for specific events (e.g., detection of a silver car).
- **Comprehensive Logging & Indexing**: Simulates and logs drone telemetry data alongside video frame detections for future analysis.
- **AI-Powered Summarization & Q&A**: Provides concise video summaries and the ability to answer follow-up questions about detected events using LangChain’s Gemini model.

## System Architecture & Design

### Architecture Overview
1. **Data Pipeline**:
   - **Input**: A video file is provided as input.
   - **Processing**: The video is read frame by frame. Every _nth_ frame (configurable via `skip_frames`) is processed for object detection.
   - **Detection & Captioning**: 
     - **YOLOv8** detects relevant objects (cars, trucks, buses, motorcycles).
     - **BLIP** generates detailed captions for each detected object.
   - **Telemetry Simulation**: Each frame is accompanied by simulated drone telemetry data (latitude, longitude, altitude).
   - **Database Logging**: Detections, telemetry data, and alerts are logged into an SQLite database.
   - **Alert Generation**: An alert is generated exclusively when a silver car is detected.

2. **Interactive Interface**:
   - **Streamlit App**: Provides a user-friendly interface to:
     - Upload a video file.
     - View real-time logs and alerts.
     - Query detections by object label or color.
     - Generate a one-sentence video summary.
     - Ask follow-up questions about the video content.

3. **AI Integration**:
   - **YOLOv8**: For efficient object detection in video frames.
   - **BLIP**: For generating natural language descriptions of detected objects.
   - **LangChain (Gemini 1.5 Pro)**: For summarizing video content and providing Q&A capabilities.

### Database Schema
- **`detections`**: Logs each detection with timestamp, object label, approximate color, confidence, and BLIP caption.
- **`alerts`**: Stores alerts triggered when a silver car is detected.
- **`telemetry`**: Logs simulated drone telemetry data for each processed frame.

## Environment Setup & Installation

### Prerequisites
- **Python 3.7+**
- Required packages:
  - `streamlit`
  - `opencv-python`
  - `numpy`
  - `ultralytics`
  - `transformers`
  - `Pillow`
  - `langchain_google_genai`
- An active Google API Key (for LangChain integration)

### Installation Steps

1. **Clone the Repository**
   
   git clone <your-private-repo-url>
   cd <repository-directory>

2. Install Dependencies Use pip to install the required packages:

  pip install streamlit opencv-python ultralytics transformers Pillow langchain_google_genai

3. Set Up Environment Variables Export your Google API key:

  export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"

### Running the Project

Running the Streamlit Application
Launch the App In your terminal, execute:

  streamlit run app.py

Using the App

### Video Analysis:

  Navigate to the Video Analysis section.

  Upload a video file (supported formats: mp4, avi, mov).

  Set the desired number of frames to skip (for faster processing) and click Run Video Analysis.

### View Logs & Queries:

  Use the sidebar options to query the database:

  Query detections by label (e.g., “car”, “truck”).

  Query detections by color (e.g., “silver”).

  View generated alerts and telemetry logs.

### Video Summary & Q&A:

  Generate a one-sentence summary of the video.

  Ask follow-up questions (e.g., “What objects were seen in the video?”) to receive AI-generated responses.

### Running the Jupyter Notebook

The Project_Build.ipynb notebook details the complete development process from database initialization, model loading, detection, and alert logic, to the final integration of all components. Open the notebook using:
  jupyter notebook Project_Build.ipynb

### Testing & Quality Assurance

  Detection Logging Test: Verifies that detections (e.g., cars, trucks) are correctly logged with their corresponding timestamps, confidence levels, and BLIP captions.

  Alert Generation Test: Confirms that an alert is generated only when a silver car is detected.

  Telemetry Data Test: Ensures simulated telemetry data (latitude, longitude, altitude) is accurately recorded per frame.

  QA Functions: Includes queries for detections, alerts, and telemetry that have been manually verified and documented.

### AI Tools & Workflow

  YOLOv8: Chosen for its efficiency and accuracy in object detection.

  BLIP: Selected for generating concise and informative captions for detected objects.

  LangChain (Gemini 1.5 Pro): Integrated to provide natural language summarization and Q&A capabilities, enabling interactive insights about the video content.

  AI-Assisted Development: Tools such as Claude Code and Cursor AI expedited the coding process, especially for integrating LangChain and designing the context management framework.

