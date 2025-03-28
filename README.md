# Agentic Multi-Agent Research Assistant

This project implements an integrated research assistant that leverages three specialized agents to create comprehensive research reports. The system combines structured financial data from Snowflake, unstructured NVIDIA quarterly reports, and real-time web data to deliver historical performance insights, valuation metrics, and up-to-date industry trends.

---

## **📌 Project Resources**
- **Streamlit:** [Application Link](http://34.85.173.233:8501/)
- **Airflow Dashboard:** [Airflow Link](http://34.21.56.116:8080)
- **Backend:** [Backend Link](http://34.85.173.233:8000/)
- **Demo Video:** [YouTube Demo](https://youtu.be/7x4iwCADyJA)
- **Documentation:** [Codelab/Documentation Link](https://codelabs-preview.appspot.com/?file_id=1xFumshJM3UlPdMnpQ0lPdu22o8shux50UXjRI8qIng4#1)

---

## **📌 Technologies Used**
<p align="center">
  <img src="https://img.shields.io/badge/-Apache_Airflow-017CEE?style=for-the-badge&logo=apache-airflow&logoColor=white" alt="Apache Airflow">
  <img src="https://img.shields.io/badge/-Snowflake-007FFF?style=for-the-badge" alt="Snowflake">
  <img src="https://img.shields.io/badge/-Pinecone-734BD4?style=for-the-badge" alt="Pinecone">
  <img src="https://img.shields.io/badge/-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/-Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/-Web_Search-FFA500?style=for-the-badge" alt="Web Search API">
</p>

---

## **📌 Architecture Diagram**
<p align="center">
  <img src="https://github.com/Damg7245-BigDataIntelligence/Agentic_Research_Assistant/blob/main/Diagram/MultiAgent_AI_Nvidia_LangGraph.png" alt="Architecture Diagram" width="600">
</p>

---

## **📌 Project Flow**

### **Step 1: Data Requirements**
- **Unstructured Data:**  
  - NVIDIA Quarterly Reports.
- **Structured Data:**  
  - NVIDIA Valuation Measures obtained from Yahoo Finance, populated into a Snowflake database.

### **Step 2: Reworking the RAG System**
- **Vector Database:**  
  - Exclusively use Pinecone for the RAG pipeline.
- **Structured Metadata:**  
  - Implement metadata columns in Pinecone by including *Year* and *Quarter* for each data chunk.
- **Hybrid Search:**  
  - Enable filtering of results by Year, Quarter, or both. (Refer to the Pinecone Metadata Filtering Example)

### **Step 3: Creating an Agentic Multi-Agent System with LangGraph**

#### **A. Snowflake Agent**
- **Functionality:**  
  - Connect to Snowflake to query structured valuation measures.
  - Generate textual summaries and visualizations (charts) from the valuation data.

#### **B. RAG Agent (Pinecone-powered)**
- **Functionality:**  
  - Perform metadata-filtered retrieval from quarterly NVIDIA report data.
  - Generate context-aware responses based on the retrieved report chunks.

#### **C. Web Search Agent**
- **Functionality:**  
  - Leverage a real-time search API (e.g., SerpAPI, Tavily, or Bing API) to fetch current and relevant web-based data.
  - Supplement research reports with the latest news, trends, or articles related to NVIDIA.

### **Step 4: Research Report Generation**
- **Output:**  
  - **Historical Performance:** Derived using the RAG agent.
  - **Financial Valuation Metrics and Visuals:** Provided by the Snowflake agent.
  - **Real-Time Industry Insights:** Sourced from the Web Search agent.

### **Step 5: User Interface (Streamlit + FastAPI)**
- **Features:**  
  - Allow users to pose research questions.
  - Enable filtering by Year/Quarter via Pinecone metadata.
  - Provide options to trigger responses from individual agents or a combined agent output.
  - Display structured research reports that include:
    - Summaries
    - Data-driven visuals
    - Real-time insights from web queries
- **Backend:**  
  - Use FastAPI to interface between the UI and the multi-agent LangGraph backend.

### **Step 6: Deployment (Dockerized)**
- **Deployment Setup:**  
  - Create a streamlined Docker-based deployment that integrates:
    - LangGraph orchestration.
    - Snowflake connectivity.
    - Pinecone-enabled RAG.
    - Web search API integration.
- **Note:**  
  - Airflow orchestration is no longer required for this assignment.

---

## **📌 Contributions**

| **Member**   | **Contribution**                                                                                                                                         |
|--------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Janvi** | **33%** – Developed and integrated the Snowflake Agent. This involved connecting to Snowflake to query structured valuation measures and generating textual summaries and visualizations from the data. |
| **Ketki** | **33%** – Implemented the RAG Agent using Pinecone. Focused on performing metadata-filtered retrieval from NVIDIA quarterly reports and generating context-aware responses from the retrieved chunks. |
| **Sahil** | **33%** – Designed and integrated the Web Search Agent. Responsible for leveraging a real-time search API to fetch current web data and developing the Streamlit UI along with FastAPI backend integration. |
---

## **📌 Attestation**
**WE CERTIFY THAT WE HAVE NOT USED ANY OTHER STUDENTS' WORK IN OUR ASSIGNMENT AND COMPLY WITH THE POLICIES OUTLINED IN THE STUDENT HANDBOOK.**

---
