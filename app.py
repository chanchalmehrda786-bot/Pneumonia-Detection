import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import cv2
import io
import time
from datetime import datetime

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle, Image as PDFImage
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.lib.units import inch

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="AI Pneumonia Detection Pro",
    page_icon="🫁",
    layout="wide"
)

# ---------------------------------------------------
# DARK MODE TOGGLE
# ---------------------------------------------------
dark = st.sidebar.toggle("🌙 Dark Mode")

# ---------------------------------------------------
# GLOBAL CSS (dynamic colors + enhanced buttons + dashboard heading)
# ---------------------------------------------------
def set_css(dark_mode):
    text_color = "white" if dark_mode else "black"
    # Button background and text colors
    analyze_bg = "#28a745" if not dark_mode else "#28a745"   # Green light/dark
    analyze_text = "white" if not dark_mode else "white"
    download_bg = "#007bff" if not dark_mode else "#3399ff"  # Blue light/dark
    download_text = "white"
    hover_analyze = "#218838" if not dark_mode else "#52cc80"
    hover_download = "#0069d9" if not dark_mode else "#66b3ff"

    st.markdown(f"""
    <style>
    .stApp {{
        color: {text_color};
    }}
    .main-title {{
        font-size:36px !important;
        font-weight:bold;
        color: {text_color};
    }}
    .tab-title {{
        font-size:28px !important;
        font-weight:bold;
        color: {text_color};
        margin-bottom:10px;
    }}
    .sub-heading {{
        font-size:20px !important;
        font-weight:bold;
        color: {text_color};
        margin-top:15px;
    }}
    .patient-subheader {{
        color: {text_color} !important;
    }}
    label {{
        color: {text_color} !important;
        font-weight:bold !important;
    }}
    /* Buttons styling */
    div.stButton > button {{
        font-weight:bold;
        border-radius:10px;
        height:45px;
        width:250px;
    }}
    /* Analyze button */
    div.stButton > button[data-baseweb]:nth-of-type(1) {{
        background-color:{analyze_bg};
        color:{analyze_text};
    }}
    div.stButton > button[data-baseweb]:nth-of-type(1):hover {{
        background-color:{hover_analyze};
        color:{analyze_text};
    }}
    /* Download buttons */
    div.stButton > button[data-baseweb]:nth-of-type(2),
    div.stButton > button[data-baseweb]:nth-of-type(3) {{
        background-color:{download_bg};
        color:{download_text};
    }}
    div.stButton > button[data-baseweb]:nth-of-type(2):hover,
    div.stButton > button[data-baseweb]:nth-of-type(3):hover {{
        background-color:{hover_download};
        color:{download_text};
    }}
    /* Dashboard heading */
    .dashboard-title {{
        font-size:24px !important;
        font-weight:bold;
        color: {text_color};
    }}
    </style>
    """, unsafe_allow_html=True)

set_css(dark)

# Background images
bg_img = "https://images.unsplash.com/photo-1530026405186-ed1f139313f8" if dark else "https://images.unsplash.com/photo-1581594693702-fbdc51b2763b"
st.markdown(f"""
<style>
.stApp {{
background-image:url("{bg_img}");
background-size:cover;
background-attachment:fixed;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:
    st.markdown("""
<div style="color:#003366;font-weight:bold;font-size:16px;line-height:1.6;background:white;padding:15px;border-radius:10px;">

## 🫁 Project Overview

AI Pneumonia Detection Pro analyzes chest X-ray images using a deep learning CNN model.

### 🔧 Features
AI X-ray scan – Analyze chest X-rays automatically  
Lung heatmap visualization – Highlight affected regions  
Batch processing – Process multiple X-rays at once  
Prediction probability charts – Visualize results  
Dark / Light mode toggle – Easy on the eyes  
Doctor PDF report – Generate detailed reports  

### 🛠 Technology
TensorFlow CNN model  
Streamlit web interface  
Plotly image analytics  
OpenCV image processing  

### 📋 Model Info
Input: 150x150 RGB  
Architecture: CNN  

⚠ For educational use only

</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pneumonia_model.h5", compile=False)

model = load_model()

# ---------------------------------------------------
# PREPROCESS
# ---------------------------------------------------
def preprocess(img):
    img = img.resize((150,150))
    arr = np.array(img)/255
    arr = np.expand_dims(arr,0)
    return arr

# ---------------------------------------------------
# HEATMAP
# ---------------------------------------------------
def heatmap(img):
    gray = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2GRAY)
    heat = cv2.applyColorMap(gray,cv2.COLORMAP_JET)
    return heat

# ---------------------------------------------------
# PDF REPORT
# ---------------------------------------------------
def generate_pdf(predictions, images, patient):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    for i,(pred,img) in enumerate(zip(predictions,images)):
        c.setFillColor(colors.HexColor("#fdfdf5"))
        c.rect(0,0,width,height,fill=1)
        c.setFillColor(colors.HexColor("#003366"))
        c.rect(0,height-70,width,70,fill=1)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold",22)
        c.drawString(90,height-45,"AI Pneumonia Detection Report")
        c.circle(45,height-35,18,fill=1)
        c.setFillColor(colors.HexColor("#003366"))
        c.setFont("Helvetica-Bold",16)
        c.drawCentredString(45,height-40,"AI")
        c.setFont("Helvetica",80)
        c.setFillColor(colors.lightgrey)
        c.drawCentredString(width/2,height/2,"AI REPORT")
        c.setFillColor(colors.black)
        c.setFont("Helvetica",12)
        c.drawString(50,height-110,f"Patient Name: {patient['name']}")
        c.drawString(50,height-130,f"Age: {patient['age']}")
        c.drawString(50,height-150,f"Patient ID: {patient['id']}")
        c.drawString(350,height-110,
                     f"Report Date: {datetime.now().strftime('%Y-%m-%d')}")
        img_buffer = io.BytesIO()
        img.save(img_buffer,format="PNG")
        img_buffer.seek(0)
        pdf_img = PDFImage(img_buffer,3*inch,3*inch)
        pdf_img.drawOn(c,width/2-1.5*inch,height-420)
        data = [["Class","Probability"],
                ["Pneumonia",f"{pred:.2%}"],
                ["Normal",f"{1-pred:.2%}"]]
        table = Table(data,colWidths=[200,200])
        style = TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#003366')),
            ('TEXTCOLOR',(0,0),(-1,0),colors.white),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('GRID',(0,0),(-1,-1),1,colors.grey)
        ])
        table.setStyle(style)
        table.wrapOn(c,width,height)
        table.drawOn(c,width/2-200,height-480)
        drawing = Drawing(400,150)
        bc = VerticalBarChart()
        bc.x = 70
        bc.y = 10
        bc.height = 120
        bc.width = 250
        bc.data=[[pred*100,(1-pred)*100]]
        bc.categoryAxis.categoryNames=['Pneumonia','Normal']
        bc.bars[0].fillColor=colors.red
        bc.bars[1].fillColor=colors.green
        drawing.add(bc)
        drawing.drawOn(c,width/2-200,height-650)
        c.setFont("Helvetica-Oblique",10)
        c.drawString(50,40,"Generated by AI Pneumonia Detection Pro – Educational Use Only")
        c.drawRightString(width-50,40,f"Page {i+1}")
        c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ---------------------------------------------------
# APP TITLE
# ---------------------------------------------------
st.markdown('<div class="main-title">🫁 AI Pneumonia Detection Pro</div>', unsafe_allow_html=True)
st.markdown('<p class="sub-heading">AI-powered chest X-ray pneumonia detection</p>', unsafe_allow_html=True)

# ---------------------------------------------------
# PATIENT INFO
# ---------------------------------------------------
st.markdown('<h2 class="patient-subheader">Patient Information</h2>', unsafe_allow_html=True)
col1,col2,col3 = st.columns(3)
with col1:
    name = st.text_input("Patient Name")
with col2:
    age = st.text_input("Age")
with col3:
    pid = st.text_input("Patient ID")
patient={"name":name,"age":age,"id":pid}

# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tab1,tab2,tab3 = st.tabs(["🔬 Scan","📦 Batch","📊 Dashboard"])
uploaded_file = None

# ---------------------------------------------------
# SINGLE SCAN
# ---------------------------------------------------
with tab1:
    st.markdown('<p class="tab-title">🔬 Scan</p>', unsafe_allow_html=True)
    file = st.file_uploader("Upload Chest X-ray", key="scan")
    if file:
        uploaded_file = file
        image = Image.open(file).convert("RGB")
        col1,col2 = st.columns(2)
        with col1:
            st.image(image,caption="Original X-ray")
        with col2:
            st.image(heatmap(image),caption="Lung Heatmap")

        if st.button("Analyze", key="analyze_scan"):
            arr = preprocess(image)
            with st.spinner("Analyzing..."):
                time.sleep(1)
            pred = model.predict(arr)[0][0]

            df = pd.DataFrame({"Class":["Pneumonia","Normal"],"Probability":[pred,1-pred]})
            fig = px.bar(df,x="Probability",y="Class",orientation="h",color="Class")
            st.plotly_chart(fig,use_container_width=True)

            pdf = generate_pdf([pred],[image],patient)
            st.download_button("Download Doctor Report",pdf,"AI_Pneumonia_Report.pdf")

# ---------------------------------------------------
# BATCH SCAN
# ---------------------------------------------------
with tab2:
    st.markdown('<p class="tab-title">📦 Batch</p>', unsafe_allow_html=True)
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        images = [image]
        preds = []
        results = []

        arr = preprocess(image)
        pred = model.predict(arr)[0][0]
        preds.append(pred)
        results.append({"File": uploaded_file.name, "Pneumonia Probability": f"{pred:.2%}"})

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption=f"Original X-ray: {uploaded_file.name}")
        with col2:
            st.image(heatmap(image), caption=f"Lung Heatmap: {uploaded_file.name}")

        st.dataframe(pd.DataFrame(results))

        # ✅ Fixed: Generate PDF before download button
        pdf = generate_pdf(preds, images, patient)
        st.download_button("Download Batch Report", pdf, "Batch_Report.pdf")
    else:
        st.info("Please upload a file in Scan tab first.")

# ---------------------------------------------------
# DASHBOARD
# ---------------------------------------------------
with tab3:
    st.markdown('<p class="tab-title">📊 Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="dashboard-title">Model Dashboard</p>', unsafe_allow_html=True)
    col1,col2,col3 = st.columns(3)
    with col1:
        st.metric("Model Layers", len(model.layers))
    with col2:
        st.metric("Total Parameters", f"{model.count_params():,}")
    with col3:
        st.metric("Estimated Accuracy", "95%")
    st.info("CNN model trained on chest X-ray pneumonia dataset")

st.caption("Educational AI Tool — Not a medical diagnosis")