
# 📈 TimeSeries Anomaly Transformer

TimeSeries Anomaly Transformer is a full-stack AI system designed to detect **anomalies in time-series data** using a custom-built **Transformer architecture**.
Users can upload CSV datasets, visualize anomalies, and explore detected abnormal points through an interactive UI.

---

## 🚀 Features

* 🤖 **Transformer-based Anomaly Detection**
* 🧠 **Association Discrepancy + Anomaly Attention Mechanism**
* 📊 **Real-time Graph Visualization**
* 📁 **CSV Dataset Upload Support**
* 🔍 **Highlights Exact Anomaly Locations**
* ⚡ Backend-powered anomaly scoring
* 🌐 React-based modern UI

---

## 🧩 Tech Stack

| Layer      | Technology                                     |
| ---------- | ---------------------------------------------- |
| Frontend   | React, Chart.js                                |
| Backend    | Express.js                                     |
| Model      | Python (NumPy, Matplotlib, PyTorch)            |
| Core Logic | Transformer, Gaussian Kernel, Minimax Strategy |
| Dataset    | Bitcoin (2018–2024) CSV                        |

---

## 📁 Folder Structure

```plaintext
TimeSeries-Transformer/
├── backend/                # Node + Express API
│   ├── server.js
│   ├── routes/
│   └── controllers/
│
├── model/                  # Python Transformer Models
│   ├── simple_transformer.py
│   ├── pytorch_transformer.py
│   ├── transformer_no_libs.py
│   ├── anomaly_attention.py
│   └── preprocess.py
│
├── frontend/               # React UI
│   ├── src/
│   │   ├── components/
│   │   │   ├── FileUploader.js
│   │   │   ├── GraphView.js
│   │   │   └── AnomalyList.js
│   │   ├── App.js
│   │   └── index.js
│   ├── public/
│   └── package.json
│
├── datasets/
│   ├── BTC_1D.csv
│   ├── BTC_4H.csv
│   ├── BTC_15M.csv
│   └── BTC_1H.csv
│
└── README.md
```

---

## 🔧 Setup Instructions

### 🐍 Python Model Setup

```bash
cd model
python -m venv venv
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

Run model:

```bash
python simple_transformer.py
```

---

### 🌐 Frontend (React)

```bash
cd frontend
npm install
npm start
```

---

### 🔌 Backend (Express)

```bash
cd backend
npm install
npm start
```

---

## 👥 Contributors

@koustub1412
Team Members — NGIT

