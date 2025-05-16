# 🩺 Medical Chatbot using RAG + LLaMA2

A modular, multi-agent chatbot built to handle medical queries with factual grounding and conversational fluency. This project leverages **LLaMA2**, **RAG (Retrieval-Augmented Generation)**, and a **multi-agent architecture** to simulate domain-specific medical experts collaborating to provide reliable answers.

---

## 📁 Project Structure

├── model.py # Loads LLaMA2 with RAG for medical QA

├── modelagent.py # Defines specialized agents (e.g., symptom checker)

├── multiagent.py # Multi-agent orchestration and conversation flow

├── requirements.txt # Dependencies

└── README.md


---

## 🧠 Core Technologies

- **LLaMA2** – Large language model for natural language generation
  
- **RAG** – Combines retrieval with generation for grounded responses

- **FAISS / VectorDB** – Embeds and indexes medical documents

- **LangChain Agents** – Modular tools simulating domain experts

- **Multi-Agent Collaboration** – Escalation and cross-referencing between agents

---

## 🔧 Setup

1. **Clone the repo**

  ```bash
  git clone https://github.com/yourusername/MultiAgent_Medi-Bot.git
  cd MultiAgent_Medi-Bot
  ```

2. **Clone the repo**

```bash
pip install -r requirements.txt
```

3. **Prepare vectorstore (FAISS)**

  Could you ensure you have a prebuilt FAISS index (medical_knowledge_index) based on your corpus of medical articles or open-source datasets like PubMed summaries or WHO guidelines?

4. **Run the chatbot**

```bash
python multiagent.py
```

📄 File Descriptions

**model.py**

Loads and initializes:
   
  -LLaMA2 with HuggingFace Transformers
  -FAISS vector store using LangChain
  -RAG pipeline via RetrievalQA

**modelagent.py**

Defines multiple domain-specific agents:

  -Symptom Checker
  -Medication Expert
  -Each tool routes the query through the RAG pipeline with prompt engineering

multiagent.py

Implements a controller agent that:
   
  -Routes medical questions to appropriate agents
  -Escalates to multiple agents if needed
  -Streams final responses back to the user
  
🧪 Example Usage

  User: "What could be the cause of chronic headaches?"

  Bot: "Chronic headaches could be caused by migraines, tension-type headaches, medication overuse, or other neurological conditions. Please consult a doctor for diagnosis."


  User: "What are the side effects of ibuprofen?"

  Bot: "Common side effects include nausea, stomach pain, and dizziness. Serious side effects may include gastrointestinal bleeding or kidney issues."

📌 Notes

  ⚠️ This is a research prototype and not intended for clinical use.

  ⚠️ Embedding quality and document coverage directly impact output accuracy.

  ⚠️ Prompt engineering can enhance factual grounding and specificity.

📬 Contact
  Feel free to reach out or collaborate!

  LinkedIn: Burada Praneeth

  GitHub: @Praneeth-19

🛡️ Disclaimer

  This chatbot does not provide medical advice and should not be used for diagnosis or treatment. Always consult a licensed medical professional.

⭐️ Future Enhancements
  ✅ Better document retrieval (e.g., semantic search via Cohere or OpenSearch)

  ✅ Plugin for patient symptom upload (PDFs, JSON, etc.)

  ✅ Integration with FHIR/EHR systems

  ✅ UI/Chat frontend (Streamlit, React, or chatbot SDK)
