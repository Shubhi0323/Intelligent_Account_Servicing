# IASW System Architecture

This document contains a high-level system architecture diagram for the Intelligent Account Servicing Workflow (IASW). It outlines the relationship between the Streamlit Frontend, the LangGraph Backend, External APIs, and the SQLite Database.

If your code editor supports Markdown Preview (like VS Code), you can open the preview to view the rendered diagram below.

```mermaid
graph TD
    %% Define Styles
    classDef ui fill:#1E3A8A,stroke:#3B82F6,stroke-width:2px,color:#fff;
    classDef core fill:#065F46,stroke:#10B981,stroke-width:2px,color:#fff;
    classDef db fill:#4C1D95,stroke:#8B5CF6,stroke-width:2px,color:#fff;
    classDef ext fill:#7F1D1D,stroke:#EF4444,stroke-width:2px,color:#fff;
    
    %% Actors
    User((Bank Customer))
    Admin((Human Checker))

    %% UI Layer (Streamlit)
    subgraph UI ["🖥️ Streamlit Frontend (main.py)"]
        UI_Intake[Customer Intake Form]
        UI_Dashboard[Admin Checker Dashboard]
        UI_Profile[User Profiles View]
    end

    %% Core Application Layer (LangGraph Pipeline)
    subgraph Core ["⚙️ Backend Processing (core/)"]
        Graph[LangGraph Orchestrator<br/>core/graph.py]
        
        subgraph Agents ["🤖 AI Node Agents"]
            Node1[OCR Processing]
            Node2[Authenticity Engine]
            Node3[Semantic Similarity<br/>ChromaDB]
            Node4[Validation Rule Engine]
            Node5[Fraud Detection<br/>OpenCV]
            Node6[Confidence Scorer]
            Node7[Summary Generator]
        end
    end

    %% Database Layer
    subgraph Security ["🔒 Security Layer"]
        Crypto[crypto_utils.py<br/>Fernet Encryption Gateway]
    end

    subgraph Database ["💾 Data Persistence (iasw.db)"]
        DB_Req[(Requests Table<br/>Encrypted)]
        DB_User[(Users Table<br/>Encrypted)]
    end

    %% External APIs
    subgraph External ["🌐 External APIs"]
        API_OCR[OCR.space REST API]
        API_Geo[OpenStreetMap Nominatim]
        API_LLM[Google Gemini API]
    end
    
    %% User Flow
    User -->|Submits Image & Data| UI_Intake
    UI_Intake -->|Triggers Pipeline| Graph
    
    %% Graph Internal Flow
    Graph --> Node1
    Node1 -.->|Parallel Fan-out| Node2
    Node1 -.->|Parallel Fan-out| Node3
    Node1 -.->|Parallel Fan-out| Node4
    Node1 -.->|Parallel Fan-out| Node5
    Node2 -.->|Fan-in| Node6
    Node3 -.->|Fan-in| Node6
    Node4 -.->|Fan-in| Node6
    Node5 -.->|Fan-in| Node6
    Node6 --> Node7
    
    %% External API Connections
    Node1 <-->|POST Image Base64| API_OCR
    Node4 <-->|Geocode Address| API_Geo
    Node7 <-->|Generate NLP Summary| API_LLM
    
    %% DB Connections
    Graph -->|Saves Verification Result| Crypto
    UI_Dashboard <-->|Fetches Pending Queue| Crypto
    UI_Profile <-->|Fetches Live Users| Crypto
    UI_Dashboard -->|Approves & Applies Change| Crypto
    
    Crypto <-->|Encrypts/Decrypts on the fly| DB_Req
    Crypto <-->|Encrypts/Decrypts on the fly| DB_User
    
    %% Admin Flow
    Admin -->|Reviews Pending AI Outputs| UI_Dashboard
    Admin -->|Monitors Records| UI_Profile

    %% Apply Classes
    class UI_Intake,UI_Dashboard,UI_Profile ui;
    class Graph,Node1,Node2,Node3,Node4,Node5,Node6,Node7 core;
    class DB_Req,DB_User db;
    class API_OCR,API_Geo,API_LLM ext;
```

## Component Breakdown

1. **Frontend (`main.py`)**: A Streamlit application handling Role-Based Access Control (RBAC). It separates the "Maker" (Customer submitting the form) and the "Checker" (Admin reviewing the AI output).
2. **Backend Orchestrator (`core/graph.py`)**: Uses `langgraph` to asynchronously manage the flow of data through specialized AI agents.
3. **AI Agents (`core/*.py`)**: Modular Python scripts handling specific verification tasks (OCR extraction, rule validation, semantic vector matching via ChromaDB, fraud detection using OpenCV heuristics, and generative summarization via Gemini).
4. **External Integrations**: Relies on `requests` to reach out to OCR.space, OpenStreetMap (for physical address validation), and the Google Gemini LLM API.
5. **Database (`core/database.py`)**: An SQLAlchemy-managed SQLite file (`iasw.db`) tracking all historical requests, confidence breakdowns, and acting as the source of truth for current user profile data.
