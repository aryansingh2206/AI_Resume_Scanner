
### 🚀 AI-Powered Resume Scanner

A smart AI-based tool that analyzes resumes and matches them with job descriptions using **NLP, Word2Vec, and Machine Learning**. This project helps recruiters and job seekers by providing a **match score** based on skill relevance.

## 🔥 Features
- 📄 **Resume Parsing**: Extracts text from PDFs using `pdfplumber`
- 🧠 **AI-Powered Matching**: Uses `Word2Vec` similarity and **Logistic Regression** to determine job fit
- 📊 **Match Score Calculation**: Assigns a percentage score for relevance
- 🖥 **Web-Based UI**: Frontend in **React** (deployed on Vercel), Backend in **Flask** (deployed on Railway)
- 📡 **Database Integration**: Stores analyzed resumes in **PostgreSQL**

## 🛠 Tech Stack
- **Backend**: Flask, NLP (NLTK, Gensim), PostgreSQL, Word2Vec
- **Frontend**: React.js, JavaScript, HTML/CSS
- **Machine Learning**: Scikit-learn (Logistic Regression), TF-IDF Vectorization
- **Deployment**: Railway (Backend), Vercel (Frontend)
- **API Handling**: Flask-CORS, Fetch API

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/aryansingh2206/AI_Resume_Scanner.git
cd AI_Resume_Scanner
```

### 2️⃣ Backend Setup (Flask)
```sh
pip install -r requirements.txt
python app.py
```
**Note:** Ensure PostgreSQL is running with the correct credentials.

### 3️⃣ Frontend Setup (React)
```sh
cd frontend
npm install
npm run dev
```

## 📡 Deployment
- **Frontend:** Vercel (https://ai-resume-scanner.vercel.app)
- **Backend:** Railway (https://airesumescanner-production.up.railway.app/)

## 📜 API Endpoints
- `POST /analyze`: Uploads a resume & job description, returns a match score.

## 🤝 Contributing
Feel free to fork this repo and submit PRs! 🚀
