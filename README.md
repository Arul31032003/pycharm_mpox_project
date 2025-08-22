Deep Learning Framework for Monkeypox Detection & Chatbot
=========================================================

Overview
--------
This project, “A Deep Learning-Based Framework for Accurate Detection 
and Classification of Monkeypox Disease with Chatbot”, is designed 
to support early diagnosis, awareness creation, and public health 
management of Monkeypox (Mpox). 

The framework has two main components:
1. Image Classification Module – A VGG16 Convolutional Neural Network 
   (CNN) trained on medical image datasets of skin lesions to classify 
   Monkeypox vs Non-Monkeypox with high accuracy. The model is evaluated 
   using accuracy, precision, recall, and F1-score.
2. Chatbot Module – An AI-powered chatbot built using PyTorch and NLTK 
   to interact with users. It answers questions related to Monkeypox 
   symptoms, diagnostic steps, prevention, and treatment awareness. 

Both modules are integrated and deployed with the Flask framework, 
providing a simple and user-friendly web interface. 

Features
--------
- Automated Monkeypox detection using VGG16 CNN model
- Interactive chatbot for user queries (symptoms, prevention, treatment)
- Flask-based web application for deployment
- Diagnostic reports summarizing classification results
- Awareness module to educate users about Monkeypox
- Evaluation metrics: Accuracy, Precision, Recall, F1-score

Tech Stack
----------
- Python 3.6+
- Flask (Web framework)
- TensorFlow/Keras (VGG16 CNN for classification)
- PyTorch (Neural network for chatbot)
- NLTK (Natural Language Processing for chatbot)
- Pandas, NumPy (Data handling)
- Matplotlib, Seaborn (Visualization and performance graphs)

Project Structure
-----------------
pycharm_mpox_project/
│── data/                # Dataset (Monkeypox & Non-Monkeypox images)
│── src/                 # Source code
│   ├── preprocessing.py # Data preprocessing & augmentation
│   ├── model.py         # VGG16 CNN implementation
│   ├── chatbot.py       # Neural network chatbot (PyTorch + NLTK)
│   ├── app.py           # Flask application integration
│── static/              # CSS, JS, and image assets for frontend
│── templates/           # HTML templates for Flask UI
│── requirements.txt     # Dependencies
│── README.txt           # Documentation

Installation
------------
1. Clone the repository:
   git clone https://github.com/Arul31032003/pycharm_mpox_project.git
   cd pycharm_mpox_project

2. Install dependencies:
   pip install -r requirements.txt

3. Run the Flask app:
   python src/app.py

4. Open in your browser:
   http://127.0.0.1:5000/

Usage
-----
- Upload a lesion image → Model predicts Monkeypox / Non-Monkeypox 
  with confidence score.
- Interact with chatbot → Ask questions like:
  "What are Monkeypox symptoms?" or "How does it spread?"
- View diagnostic results → Classification + chatbot response.

Example Outputs
---------------
Case 1: Image Classification
- Input: Skin lesion image
- Output: Monkeypox Detected (Confidence: 95%)

Case 2: Chatbot Interaction
- User: "What are the symptoms of Monkeypox?"
- Chatbot: "Common symptoms include fever, rash, swollen lymph nodes, 
  and body aches."

Future Scope
------------
- Expand dataset with more diverse cases for higher accuracy.
- Improve chatbot using advanced NLP/NLU for smarter responses.
- Develop a mobile application for better accessibility.
- Integrate with telemedicine platforms for remote consultation.
- Extend system to detect other diseases (e.g., Chickenpox, Measles).
- Add real-time outbreak analytics and visualization dashboard.


License
-------
This project is licensed under the MIT License. You are free to use, 
modify, and distribute this project with attribution.


Developed as part of the B.E. Computer Science & Engineering Project 
at SRM Valliammai Engineering College (2024).
