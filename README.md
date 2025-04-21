# 🌿 AgroLens: AI-Powered Plant Health & Irrigation Risk Analyzer

AgroLens is a multimodal Generative AI-powered assistant that analyzes plant leaf images to detect diseases, interprets irrigation water quality data, and generates detailed PDF reports with treatment guidance and fertilizer recommendations. It integrates vision models, retrieval-augmented generation (RAG), and intelligent agents using LangChain and Gemini for real-time, context-aware insights.

---

## 📌 Features

- 🧠 **Image Classification:** Uses a ResNet/ViT model to detect plant diseases from real-world leaf images (PlantVillage dataset).
- 📄 **Document RAG:** Embeds and retrieves snippets from a domain-specific plant pathology handbook using Gemini embeddings + FAISS vector store.
- 💧 **Water Quality Risk Analysis:** Evaluates nitrate, TDS, chloride, fluoride, pH, SAR levels to identify hidden irrigation issues.
- 🌱 **Fertilizer Guidance:** Suggests NPK ratios tailored to the diagnosed disease and plant health.
- 🤖 **LangChain Agents:** Combines tools into an interactive agent capable of answering queries like “How to treat this disease with current water?”
- 📄 **PDF Generator:** Outputs beautifully formatted, one-click downloadable health reports per plant image.

---

## 🚀 Demo Example

### 🎯 Input
- Leaf Image (e.g., Tomato Bacterial Spot)
- District (e.g., VIKARABAD)

### 📋 Output
- **Disease Detected:** Tomato Bacterial Spot
- **Water Risk Flags:**
  - ⚠️ Nitrate High
  - ⚠️ Chloride Elevated
- **Recommended Treatment:**
  - Avoid nitrogen-heavy fertilizers
  - Improve drainage and apply copper-based fungicide
- **PDF report generated** with diagnosis, water profile, and fertilizer plan

---

## 🧠 Tech Stack

| Component        | Description                                   |
|------------------|-----------------------------------------------|
| `torchvision.models` | Pretrained ResNet for image classification |
| `langchain`      | Framework for chaining GenAI components       |
| `Gemini API`     | Google Generative AI for LLM + embeddings     |
| `FAISS`          | Vector store for efficient document retrieval |
| `FPDF`           | Generate reports in PDF format                |
| `PyPDFLoader`    | Load and chunk PDF handbooks                  |

---

## 📁 Folder Structure

```
📦 AgroLens
 ┣ 📂 data/
 ┃ ┣ 📜 PlantVillage/
 ┃ ┣ 📜 WaterQuality_2018_2019_2020.csv
 ┃ ┣ 📜 Disease_Handbook.pdf
 ┣ 📜 leaf_disease_assistant.ipynb
 ┣ 📜 requirements.txt
 ┗ 📜 README.md
```

---

## 📦 Setup Instructions

1. Clone the repo or open in **Kaggle Notebook** / **Google Colab**.
2. Install dependencies:

```bash
pip install -q fpdf faiss-cpu langchain langchain-community google-genai==1.7.0 langchain-google-genai PyMuPDF
```

3. Load your **Gemini API key** via environment or secrets.
4. Run the notebook and generate reports!

---

## ✅ GenAI Capabilities Demonstrated

- ✅ **Image Understanding**
- ✅ **Document Understanding**
- ✅ **Agents (LangChain)**
- ✅ **Embeddings + Vector Store (FAISS)**
- ✅ **Retrieval-Augmented Generation (RAG)**
- ✅ **Function Calling via LangChain Tools**
- ✅ **Structured Output / Report Generation**

---

## 📈 Future Enhancements

- Add real-time chat UI for farmers
- Support multiple PDF sources + languages
- Integrate weather data for smarter irrigation tips
- Deploy as a Streamlit or FastAPI app

---

## 📄 License

MIT License

---

## 👨‍🌾 Inspired by:
- [PlantVillage Dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)
- [Telangana Groundwater Quality Data](https://www.kaggle.com/datasets/sivapriyagarladinne/telangana-post-monsoon-ground-water-quality-data)
- Google DeepMind + LangChain AI stack
```
