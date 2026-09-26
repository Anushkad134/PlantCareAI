# PlantCareAI 🌿


### AI-Powered Plant Health Analysis & Prediction

**PlantHealthAI** is an intelligent plant-health analysis application that combines **multimodal AI** and **machine learning** to assess plant health through two complementary approaches:

* 📷 **AI Vision Analysis** — analyzes plant/leaf images using **GPT-4.1-mini**
* 📊 **Machine Learning Prediction** — predicts plant health conditions from environmental and plant-related parameters using a **Random Forest Classifier**

The project demonstrates the integration of **Generative AI, Computer Vision, classical Machine Learning, data preprocessing, feature engineering, and interactive application development** into a single plant-health analysis system.

---

## ✨ Key Features

### 📷 AI-Based Leaf Image Analysis

Upload a plant or leaf image and let a multimodal AI model analyze visible characteristics.

The system can provide:

* 🌱 Plant/crop identification when recognizable
* 🩺 Healthy or diseased assessment
* 🦠 Possible disease identification
* 🔎 Visible symptoms
* 🌡️ Possible contributing causes
* 🛡️ General preventive measures
* ⚠️ Uncertainty indication when the image is unclear

Powered by **OpenAI GPT-4.1-mini** through the OpenAI API.

---

### 📊 Machine Learning Plant Health Prediction

The second analysis mode uses structured plant and environmental information to predict a plant-health class.

**Input features include:**

* Plant type
* Habitat
* Season
* Air humidity
* Soil humidity
* Ethylene concentration
* CO₂ concentration
* O₂ concentration
* Light level
* Leaf color index
* Temperature

The trained **Random Forest Classifier** predicts:

`Healthy` • `Moderate Stress` • `High Stress`

---

## 🧠 AI / ML Architecture

```text
                    ┌──────────────────────┐
                    │   PlantHealthAI      │
                    └──────────┬───────────┘
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
     📷 IMAGE ANALYSIS                  📊 ML PREDICTION
              │                                 │
       Leaf / Plant Image              Plant + Environment Data
              │                                 │
              ▼                                 ▼
       Base64 Encoding                  Pandas DataFrame
              │                                 │
              ▼                                 ▼
      OpenAI GPT-4.1-mini              Preprocessing Pipeline
              │                                 │
              ▼                                 ▼
   Visual Health Analysis              One-Hot Encoding
                                                │
                                                ▼
                                      Random Forest Classifier
                                                │
                                                ▼
                                      Plant Health Classification
```

---

# 🔬 Machine Learning Pipeline

The ML component follows a complete supervised-learning workflow:

### 1. Dataset Loading

The project uses the Hugging Face dataset:

**`soumikmahato/plant-health-points-synthetic`**

The dataset contains:

* **2,000 records**
* **14 columns**
* Categorical + numerical plant/environment features
* Target: `health_class`

The dataset is **synthetic**, rather than a direct collection of real-world sensor measurements.

---

### 2. Data Preparation

The `record_id` column is removed because it is only an identifier.

The prediction target is:

```text
health_class
```

The `health_score` field is excluded from the input features.

The remaining attributes are divided into:

**Categorical Features**

```text
plant_type
habitat
season
```

**Numerical Features**

```text
air_humidity_pct
soil_humidity_pct
ethylene_ppm
co2_ppm
o2_ppm
ldr_on_plant
leaf_color_index
temperature_c
```

---

### 3. Categorical Feature Encoding

Categorical variables cannot be directly processed as ordinary numerical values.

Therefore, the project uses:

**One-Hot Encoding**

Example:

```text
plant_type = tomato
```

can be transformed into binary feature columns.

The implementation uses:

```python
OneHotEncoder(handle_unknown="ignore")
```

This also makes the model more robust when an unseen category appears during prediction.

---

### 4. Preprocessing Pipeline

A `ColumnTransformer` is used to apply the appropriate preprocessing to different feature types.

```python
ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
], remainder="passthrough")
```

Categorical features → One-Hot Encoding
Numerical features → Passed through unchanged

---

### 5. Random Forest Classification

The prediction model is:

```text
Random Forest Classifier
```

Configuration:

```text
Number of Trees: 100
Random State: 42
```

Random Forest combines predictions from multiple decision trees instead of relying on a single tree.

This makes it suitable for classification problems involving multiple interacting features.

---

### 6. Train-Test Split

The dataset is divided into:

```text
80% → Training
20% → Testing
```

With 2,000 records, this corresponds approximately to:

```text
1,600 training records
400 testing records
```

---

### 7. Model Evaluation

The trained model is evaluated using:

```python
accuracy_score(y_test, y_pred)
```

The actual measured accuracy in the training notebook was:

### **71%**

> Note: A later notebook cell displayed `accuracy + 0.2`, producing `0.91`. That was only a display manipulation and does **not** represent the actual model accuracy.

---

# 💾 Model Serialization

After training, the complete ML pipeline is saved using **Joblib**:

```python
joblib.dump(model, "plant_health_model.pkl")
```

This creates:

```text
plant_health_model.pkl
```

The Streamlit application loads this trained pipeline directly:

```python
model = joblib.load("plant_health_model.pkl")
```

This means the model does not need to be retrained every time the application starts.

---

# 🤖 Generative AI Integration

The image-analysis module uses:

### OpenAI GPT-4.1-mini

The uploaded image is:

```text
Image
   ↓
Read as bytes
   ↓
Base64 Encoding
   ↓
OpenAI API
   ↓
GPT-4.1-mini
   ↓
Plant Health Analysis
```

The model is instructed to analyze:

1. Plant/crop identity when recognizable
2. Health condition
3. Possible disease
4. Visible symptoms
5. Possible causes
6. General preventive measures

If the image is unclear, the application asks the model to explicitly indicate that a reliable diagnosis cannot be determined.

**Important:** GPT-4.1-mini is a pretrained external AI model accessed through the OpenAI API; it was not trained from scratch as part of this project.

---

# 🛠️ Technology Stack

### 🧠 Artificial Intelligence

* Generative AI
* Multimodal AI
* OpenAI API
* GPT-4.1-mini
* Prompt Engineering

### 🤖 Machine Learning

* Scikit-learn
* Random Forest Classification
* Supervised Learning
* One-Hot Encoding
* Train/Test Splitting
* Model Evaluation
* ML Pipelines

### 📊 Data Science

* Python
* Pandas
* Synthetic Dataset Analysis
* Categorical Feature Processing
* Numerical Feature Handling
* Data Preprocessing

### 🖼️ Image Processing

* Image Upload Handling
* Binary Image Processing
* Base64 Encoding
* Vision-based AI Analysis

### 🌐 Application Development

* Streamlit
* Interactive UI Components
* Custom CSS
* User Input Validation
* Real-time Prediction

### 💾 Model & File Handling

* Joblib
* Pickle-based Model Serialization
* Environment Variables
* `.env` / API Key Management

### ☁️ Development Environment

* Google Colab
* Hugging Face Datasets
* Python 3.x

---

# 📁 Project Structure

```text
PlantHealthAI/
│
├── app.py
├── plant_health_model.pkl
├── requirements.txt
├── README.md
│
└── .env
```

### File Responsibilities

| File                     | Purpose                        |
| ------------------------ | ------------------------------ |
| `app.py`                 | Streamlit application          |
| `plant_health_model.pkl` | Trained Random Forest pipeline |
| `requirements.txt`       | Python dependencies            |
| `.env`                   | API key configuration          |
| `README.md`              | Project documentation          |

---

# 🔄 End-to-End Workflow

```text
                    USER
                     │
             ┌───────┴───────┐
             │               │
             ▼               ▼
        Leaf Image      Plant Parameters
             │               │
             ▼               ▼
       Base64 Encode     Pandas DataFrame
             │               │
             ▼               ▼
      OpenAI GPT-4.1-mini  ML Pipeline
             │               │
             ▼               ▼
       AI Analysis       One-Hot Encoding
                             │
                             ▼
                       Random Forest
                             │
                             ▼
                     Health Classification
             │               │
             └───────┬───────┘
                     ▼
              Plant Health Result
```

---

# 📌 Skills Demonstrated

This project demonstrates practical experience with:

`Python` • `Machine Learning` • `Generative AI` • `Multimodal AI` • `Computer Vision` • `Random Forest` • `Scikit-learn` • `Pandas` • `Feature Preprocessing` • `One-Hot Encoding` • `ML Pipelines` • `Model Serialization` • `OpenAI API` • `Prompt Engineering` • `Streamlit` • `Hugging Face Datasets` • `Base64 Encoding` • `Data Analysis` • `Supervised Learning`

---

# 🎯 Project Highlights

* 🔀 **Dual-analysis architecture** combining Generative AI and classical ML
* 👁️ **Multimodal AI** for plant/leaf image interpretation
* 🌳 **100-tree Random Forest** classification pipeline
* 🧹 Automated categorical preprocessing using One-Hot Encoding
* 🔗 End-to-end Scikit-learn Pipeline
* 💾 Serialized ML model for deployment
* 📊 Trained on **2,000 synthetic plant-health records**
* 🌐 Interactive Streamlit interface
* 🔐 Environment-based API key configuration
* 🧩 Combines structured environmental data with visual AI analysis

---

# ⚠️ Current Limitations

* The ML training dataset is synthetic.
* The measured ML accuracy is approximately **71%**.
* The training dataset contains **8 plant types**, while the application UI provides a larger plant-selection list.
* Image-based analysis depends on image quality and the capabilities of the external vision model.
* The application is intended as an analytical/educational tool and should not replace professional agricultural or plant-disease diagnosis.

---

# 🚀 Future Improvements

Potential improvements include:

* Expand the dataset with real-world plant sensor data
* Add more plant species and disease classes
* Collect and evaluate real leaf-image datasets
* Improve ML performance through hyperparameter tuning
* Add confusion matrix and additional evaluation metrics
* Add model confidence/probability visualization
* Integrate real-time IoT sensor data
* Deploy the application as a cloud service
* Add historical plant-health tracking
* Combine image predictions and environmental predictions into a unified assessment

---

## 🌱 Built With

**Python · Scikit-learn · Pandas · OpenAI · GPT-4.1-mini · Streamlit · Hugging Face Datasets · Joblib · Google Colab**

> **PlantHealthAI — combining visual intelligence and machine learning for smarter plant-health analysis.**
