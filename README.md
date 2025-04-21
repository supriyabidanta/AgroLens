# 🌿 AgroLens: AI-Powered Plant Health & Irrigation Risk Analyzer

> **Capstone Project – Generative AI with Google**  
> Kaggle Notebook | Blogpost | GitHub Repo

---

## 🌾 Problem Statement: Agricultural Diagnostics at the Edge

Farmers and gardeners frequently face challenges in identifying plant diseases and understanding the environmental factors that exacerbate them. Many rely on visual cues and anecdotal knowledge, leading to incorrect treatment, overuse of fertilizers, or poor irrigation practices. Moreover, water quality, often overlooked, plays a silent but vital role in plant health.

**AgroLens** addresses this dual challenge:  
- 📸 **Detecting plant diseases** from leaf images  
- 💧 **Analyzing irrigation water quality** to assess its role in plant stress  
- 🧾 **Generating AI-driven treatment plans** and visual reports

This solution empowers growers with personalized, data-backed, AI-generated insights — without needing deep technical knowledge.

---

## 🚀 Project Overview

AgroLens is an end-to-end assistant that takes:
- A **leaf image** and **district selection**
- Retrieves **plant disease diagnosis**
- Analyzes **regional groundwater data**
- Uses RAG + LangChain agents to explain risks
- Outputs a beautifully formatted **PDF health report**

---

## 🤖 GenAI Capabilities Used

| Capability                  | How It's Used                                                                 |
|----------------------------|-------------------------------------------------------------------------------|
| ✅ **Image Understanding** | Classifies plant leaf diseases using ResNet/Vision Transformers               |
| ✅ **RAG**                 | Uses LangChain + Gemini embeddings to retrieve disease info from handbook     |
| ✅ **LangChain Agents**     | Integrates multiple tools: classifier, retriever, and risk analyzer           |

---

## 🔍 Example Use Case

A farmer from Bhupalpally uploads a tomato leaf image and selects their district. AgroLens:

1. Detects *Early Blight*
2. Finds water nitrate levels elevated at **42.5 mg/L**
3. Uses Gemini to explain the interplay between water stress and disease vulnerability
4. Suggests best-fit fertilizer blends and watering routines
5. Generates a detailed **PDF health report** (with image, diagnosis, water flags, treatment, and tips)

---

## 🧬 Core Architecture

### 1. 🧠 Leaf Disease Classifier
```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.eval()

def classify_leaf(image_path):
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
    predicted = class_names[torch.argmax(outputs)]
    return predicted
```

---

### 2. 📄 RAG using LangChain + Gemini Embeddings
```python
loader = PyPDFLoader("disease_info.pdf")
chunks = CharacterTextSplitter(chunk_size=500).split_documents(loader.load())

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(chunks, embedding)

def retrieve_info(query):
    docs = vectorstore.similarity_search(query)
    return "\n\n".join([doc.page_content for doc in docs])
```

---

### 3. 💧 Water Risk Analysis
```python
def get_water_risk_flags(row):
    def level(val, low, med):
        return "Safe" if val <= low else "Moderate" if val <= med else "High"

    return f"""
    - Nitrate: {level(row['NO3'], 30, 45)} ({row['NO3']} mg/L)
    - TDS: {level(row['TDS'], 500, 1000)} ({row['TDS']} mg/L)
    - Fluoride: {'Elevated' if row['F'] > 0.5 else 'Safe'} ({row['F']} mg/L)
    """
```

---

### 4. 🧰 LangChain Agent
```python
tools = [
    Tool(name="LeafClassifier", func=classify_leaf, description="Classifies a plant leaf image"),
    Tool(name="DiseaseRetriever", func=retrieve_info, description="Retrieves info about a plant disease"),
    Tool(name="WaterAnalyzer", func=lambda x: get_water_risk_flags(current_row), description="Flags water risks")
]

agent = initialize_agent(tools, llm=ChatGoogleGenerativeAI(model="gemini-pro"))
agent.run("Analyze this leaf image and provide water-based recommendations.")
```

---

### 5. 📄 PDF Report Generator
- Embeds image, water table, risk highlights
- Uses clean formatting and date-based naming
- Auto-generated titles like `plant_report_Bhupalpally_Tomato_EarlyBlight_2025-04-21.pdf`

---

## ⚙️ Challenges & Solutions

| Challenge                                      | Solution                                                             |
|-----------------------------------------------|----------------------------------------------------------------------|
| Handling inconsistent water data columns       | Normalized & unified across 3 years                                  |
| Token limits for large PDFs                    | Split and indexed chunks for RAG                                     |
| Rendering large leaf images inside PDF         | Used scaling + aligned sidebar water table                           |
| Gemini API errors (GCP metadata limitations)   | Switched to direct API key config for stable execution               |

---

## 🚫 Limitations

- Currently only trained on **PlantVillage dataset** — no generalization to unseen diseases
- Water data is **district-level**, not GPS-specific
- Gemini RAG can fail in restricted environments (e.g., Kaggle GPU)

---

## 🔮 Future Directions

- 🎯 Fine-tune ViT model on **real-world images**
- 🌍 Integrate **weather + soil** APIs for context-aware prescriptions
- 📱 Build a **streamlit or Android UI** for farmer-facing front-end
- 📦 Add export-to-excel and multi-leaf report batch mode

---

## 📝 Conclusion

AgroLens is a practical demonstration of how **Generative AI**, **computer vision**, and **domain data** can work together to empower agriculture. It combines disease diagnostics with real environmental analysis and produces human-friendly reports with treatment recommendations — all in one flow.

> “It’s like a doctor, a chemist, and an agronomist inside your camera.”  
> — A very happy beta user 😄

---

## 👨‍🌾 Inspired by:
- [PlantVillage Dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)
- [Telangana Groundwater Quality Data](https://www.kaggle.com/datasets/sivapriyagarladinne/telangana-post-monsoon-ground-water-quality-data)
- Google DeepMind + LangChain AI stack
```
