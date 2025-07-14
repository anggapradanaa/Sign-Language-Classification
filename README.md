<h1 align="center">Sign-Language-Classification</h1>

# Sign-Language-Classification
Indonesian Sign Language Classification System using Deep Learning to recognize SIBI hand gestures (A–Y, excluding J and Z).

### 🎯 Overview
This project utilizes Transfer Learning with the MobileNetV2 architecture to classify 24 letters in the Indonesian Sign Language System (SIBI). It features a user-friendly web interface for real-time prediction.

### 🛠️ Features
✅ High-accuracy classification of 24 SIBI letters
✅ Web interface powered by Gradio
✅ Instant image upload and analysis
✅ Top-3 predictions with confidence scores
✅ Lightweight and fast model based on MobileNetV2

### 📁 Project Structure
sibi-classifier/
├── training.py              # Script training model
├── web_interface.py         # Web interface with Gradio
├── best_sibi_model.keras    # Best Models
├── requirements.txt         # Dependencies
├── data/
│   └── SIBI/               # Dataset SIBI (A-Y folders)
└── README.md               # Documentation

### 🧠 Model Architecture
**Transfer Learning with MobileNetV2:**

* **Base Model:** MobileNetV2 (pre-trained on ImageNet)
* **Input:** 224x224x3 RGB images
* **Output:** 24 classes (SIBI letters A–Y, excluding J and Z)
* **Optimization:** Adam optimizer, Early Stopping, Data Augmentation

**Performance:**

* **Test Accuracy:** \~92%
* **Lightweight and mobile-friendly**
* **Fast inference time**

### 🎨 Web Interface
**Features**

* **Upload Area:** Drag & drop or click to upload
* **Real-time Preview:** View the uploaded image instantly
* **Instant Analysis:** Automatic prediction upon upload
* **Top-3 Results:** Ranked predictions with confidence scores
* **Modern UI:** Responsive design with gradient styling

**Usage Tips**

✅ Use good lighting
✅ Position your hand in the center of the frame
✅ Use a contrasting background
✅ Follow standard SIBI gestures
❌ Avoid blurry or cropped images

### 📊 Dataset
SIBI Letters Supported:
A B C D E F G H I K L M N O P Q R S T U V W X Y
**Note:** Letters *J* and *Z* are not included as they involve dynamic movements in SIBI.

###🙏 Acknowledgments

TensorFlow team untuk framework
Gradio team untuk web interface tools
SIBI community untuk dataset dan guidelines


**Built with ❤️ using Deep Learning**
