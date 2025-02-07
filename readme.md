<div align="center">
	<h1>🫁ParuAI: Your AI Ally in the Fight Against Lung Cancer🚀</h1>
</div>
<div align="center">
    <img src="https://img.shields.io/badge/docker-blue?style=for-the-badge&logo=docker&logoColor=white">
    <img src="https://img.shields.io/badge/python-yellow.svg?style=for-the-badge&logo=python&logoColor=white">
    <img src="https://img.shields.io/badge/tensorflow-red.svg?style=for-the-badge&logo=tensorflow&logoColor=white">
    <img src="https://img.shields.io/badge/keras-gray.svg?style=for-the-badge&logo=keras&logoColor=white">
</div>
<br>

Welcome to **ParuAI** – an innovative AI-powered web application that combines state-of-the-art deep learning with an interactive chatbot assistant to revolutionize lung cancer diagnosis. ParuAI analyzes lung CT scan images and provides insightful, real-time information about lung health, making early detection smarter and more accessible.

---

## 🚀 What is ParuAI?

ParuAI is more than just a diagnostic tool; it’s your AI ally in the battle against lung cancer. Using a custom-built Convolutional Neural Network (CNN) and an engaging chatbot powered by the LLaMA3 language model, ParuAI:
- **Classifies CT scans** into three categories: **Normal**, **Benign**, and **Malignant**.
- **Educates and assists** users by answering queries about lung cancer through a user-friendly chatbot.
- **Delivers high accuracy** with a validation accuracy of **99.27%** and a low loss of **3.32%** using a balanced approach with class weights.

**Try it now !**
https://belgis-lung-cancer-prediction.hf.space

---

## 🌟 Key Features

- **Accurate Image Analysis:**  
  This CNN model, honed with a diverse dataset of 1190 CT scan images from 110 cases, efficiently processes and classifies lung images resized to 256x256 pixels.

- **Interactive Chatbot Assistant:**  
  The integrated LLaMA3-based chatbot is designed to offer clear, compassionate, and knowledgeable responses, making it a valuable resource for both patients and medical professionals.

- **Sleek & Responsive UI:**  
  Built with HTML, JavaScript, and Tailwind CSS, the web interface is designed for simplicity and responsiveness, ensuring a smooth user experience.

- **Seamless Deployment:**  
  The complete solution is containerized using Docker and deployed on Hugging Face, ensuring that ParuAI is easily accessible and scalable.

---

## 🔍 How It Works

1. **Data Preprocessing:**  
   - CT scan images are normalized and resized for uniformity.
   - Techniques like grayscale conversion and blur addition enhance image details.
   - Advanced methods, such as class weighting, balance the data for robust model training.

2. **Model Training & Evaluation:**  
   - The CNN model is rigorously trained and validated, achieving outstanding performance.
   - Two approaches were explored: SMOTE oversampling and class weighting. The latter was chosen for its stability and consistency.

3. **Real-Time Deployment:**  
   - ParuAI is hosted on Hugging Face and runs within a Docker container, making it ready for real-world application with just a few commands.

---

## 💻 Tech Stack

- **Backend & ML:** Python, TensorFlow/Keras, Flask
- **Frontend:** HTML, JavaScript, Tailwind CSS
- **Chatbot:** LLaMA3 (Large Language Model)
- **Deployment:** Docker & Hugging Face

---

## 🚀 Get Started

### Prerequisites

- [Docker](https://www.docker.com/get-started) must be installed on your system.
- Git (optional, for cloning the repository).

### Installation & Running

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/ParuAI.git
   cd ParuAI
   ```

2. **Build Docker Image**
   ```bash
   docker build -t paruai-img:latest .
   ```

3. **Run the Docker Container**
   ```bash
   docker run -d --name paruai -p 5000:5000 paruai-img:latest
   ```

4. **Access the Application**
   Open your browser and go to http://localhost:5000 to start exploring ParuAI.


# 🎉 Why ParuAI?
ParuAI is designed with both innovation and compassion in mind. By fusing cutting-edge AI technology with a friendly, informative chatbot, ParuAI aims to empower healthcare professionals and patients alike, making early lung cancer detection more efficient and accessible. Whether you're a medical practitioner looking for a reliable diagnostic assistant or someone seeking trustworthy health information, ParuAI is here to help.
