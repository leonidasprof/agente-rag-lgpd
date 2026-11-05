# 🛡️ Auditor Inteligente de Conformidade LGPD

Sistema automatizado de governança e auditoria regulatória baseado em **Inteligência Artificial Generativa e RAG (Retrieval-Augmented Generation)**, desenvolvido para a disciplina de **Gestão de Segurança da Informação** da CESAR School.

---

## 🎯 Objetivos do Sistema

1. **💬 Chat de Consultoria Regulatória (Q&A):**
   - Respostas fundamentadas estritamente no texto oficial da **LGPD (Lei nº 13.709/2018)**.
   - Citação precisa de artigos, incisos, parágrafos e páginas da legislação.
   - Diretrizes severas de engenharia de prompt com temperatura zero para mitigação total de alucinações.

2. **📁 Auditoria Automatizada de Documentos (Google Drive):**
   - Conexão e varredura da pasta corporativa do Google Drive (`15beEeK4PEmmkf_Z72ff7Ee21LqsA0WOY`).
   - Leitura e processamento de contratos, termos de uso e políticas nos formatos `.pdf`, `.docx`, `.txt`.
   - Comparação semântica cláusula a cláusula contra a base vetorial da LGPD.
   - Emissão de parecer visual:
     - 🟢 **Conforme / Aprovado**: Documento cumpre os princípios e bases legais.
     - 🔴 **Não Conforme / Reprovado**: Documento viola regras da lei, apontando as infrações, os artigos infringidos e recomendações corretivas.

---

## 🏗️ Arquitetura Tecnológica

- **Front-End Interativo:** Streamlit com navegação por abas e estilização temática.
- **Leitura do PDF Oficial:** PyMuPDF (`fitz`) com extração página a página.
- **Segmentação (Chunking):** `RecursiveCharacterTextSplitter` com sobreposição semântica.
- **Embeddings:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` via Hugging Face.
- **Banco Vetorial:** FAISS com cache de recursos (`st.cache_resource`) e persistência em disco.
- **Modelo de Linguagem (LLM):** `ChatNVIDIA` com `stepfun-ai/step-3.5-flash` (temperatura 0.0).
- **Integração Google Drive:** Suporte a download de pastas públicas via `gdown` e autenticação corporativa via API v3 com Service Account (`credentials.json`).

---

## 🚀 Como Executar

### 1. Criar o Ambiente Virtual e Ativar
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Instalar as Dependências
```bash
pip install -r requirements.txt
```

### 3. Configurar a Chave da NVIDIA (Opcional via .env ou direto na Interface)
Copie o arquivo `.env.example` para `.env` e adicione sua chave:
```env
NVIDIA_API_KEY=nvapi-...
```
*(Você também pode digitar a chave diretamente na barra lateral da aplicação web)*

### 4. Iniciar o Streamlit
```bash
streamlit run app.py
```
A aplicação abrirá automaticamente no navegador em `http://localhost:8501`.

