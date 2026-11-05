import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Carregar variáveis de ambiente do .env se existir
load_dotenv()

from rag_engine import (
    criar_ou_carregar_vectorstore,
    obter_llm,
    consultar_lgpd,
    auditar_documento,
    DEFAULT_NVIDIA_MODEL
)
from drive_service import (
    baixar_arquivos_drive_publico,
    baixar_arquivos_drive_api,
    extrair_texto_arquivo,
    extrair_id_pasta
)

# =====================================================================
# CONFIGURAÇÃO GERAL DA PÁGINA
# =====================================================================
st.set_page_config(
    page_title="Auditor Inteligente LGPD - CESAR",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização visual complementar
st.markdown("""
<style>
    .reportview-container {
        background: #f8f9fa;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .card-conforme {
        background-color: #f0fdf4;
        border-left: 6px solid #22c55e;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .card-nao-conforme {
        background-color: #fef2f2;
        border-left: 6px solid #ef4444;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .badge-artigo {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85em;
        margin-right: 4px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# INICIALIZAÇÃO E CACHE DA BASE RAG (LGPD)
# =====================================================================
CAMINHO_PDF_LGPD = os.path.join("LGPD", "Lei_geral_protecao_dados_pessoais_1ed.pdf")

@st.cache_resource(show_spinner="Indexando o texto oficial da LGPD no FAISS com Hugging Face Embeddings...")
def inicializar_motor_rag(pdf_path: str):
    """Inicializa a base vetorial FAISS com cache de recursos do Streamlit."""
    return criar_ou_carregar_vectorstore(pdf_path)


# =====================================================================
# BARRA LATERAL (SIDEBAR) - CONFIGURAÇÕES E CREDENCIAIS
# =====================================================================
with st.sidebar:
    st.image("logo.png", use_column_width=True)
    st.title("Painel de Governança")
    st.markdown("**CESAR School** • Gestão de Segurança da Informação")
    st.divider()

    st.subheader("⚙️ Configurações da LLM")
    nvidia_key_env = os.getenv("NVIDIA_API_KEY", "")
    api_key_input = st.text_input(
        "NVIDIA API Key",
        value=st.session_state.get("nvidia_api_key", nvidia_key_env),
        type="password",
        help="Insira sua chave da NVIDIA AI Endpoints para ativar o modelo stepfun-ai/step-3.5-flash."
    )
    if api_key_input:
        st.session_state["nvidia_api_key"] = api_key_input
        os.environ["NVIDIA_API_KEY"] = api_key_input
        st.success("✅ Chave ativa! LLM habilitado.")
    else:
        st.warning("⚠️ Sem chave: modo demonstração.")

    modelo_selecionado = st.selectbox(
        "Modelo NVIDIA",
        options=[DEFAULT_NVIDIA_MODEL, "meta/llama-3.1-70b-instruct", "mistralai/mixtral-8x22b-instruct"],
        index=0
    )

    st.divider()
    st.subheader("📁 Google Drive")
    drive_url_padrao = "https://drive.google.com/drive/folders/15beEeK4PEmmkf_Z72ff7Ee21LqsA0WOY?usp=sharing"
    url_pasta_drive = st.text_input(
        "URL ou ID da Pasta do Drive",
        value=drive_url_padrao,
        help="Endereço da pasta do Google Drive com os documentos corporativos para auditoria."
    )

    modo_conexao = st.radio(
        "Método de Leitura do Drive",
        options=["Download Público Direto (Link Compartilhado)", "Google Drive API v3 (Service Account JSON)"],
        index=0
    )

    caminho_credenciais_sa = None
    if modo_conexao == "Google Drive API v3 (Service Account JSON)":
        uploaded_cred = st.file_uploader("Arquivo credentials.json (Service Account)", type=["json"])
        if uploaded_cred:
            temp_cred_path = os.path.join(tempfile.gettempdir(), "credentials_sa.json")
            with open(temp_cred_path, "wb") as f:
                f.write(uploaded_cred.getbuffer())
            caminho_credenciais_sa = temp_cred_path
        elif os.path.exists("credentials.json"):
            caminho_credenciais_sa = "credentials.json"
            st.caption("Usando `credentials.json` local detectado.")

    st.divider()
    # Status da base vetorial
    if os.path.exists(CAMINHO_PDF_LGPD):
        st.success("✅ Base Legal: Lei nº 13.709/2018 Carregada")
    else:
        st.error("❌ PDF da LGPD não encontrado na pasta LGPD/")

    if st.button("🔄 Recriar Índice FAISS"):
        st.cache_resource.clear()
        with st.spinner("Reconstruindo índice vetorial..."):
            criar_ou_carregar_vectorstore(CAMINHO_PDF_LGPD, forcar_recriar=True)
        st.success("Índice FAISS reconstruído com sucesso!")


# Carregamento do Vector Store e LLM
try:
    vectorstore = inicializar_motor_rag(CAMINHO_PDF_LGPD)
except Exception as e:
    st.error(f"Erro ao inicializar base vetorial FAISS: {e}")
    vectorstore = None

llm = None
chave_ativa = st.session_state.get("nvidia_api_key", os.getenv("NVIDIA_API_KEY"))
if chave_ativa:
    try:
        llm = obter_llm(chave_ativa, modelo_selecionado)
    except Exception as e:
        st.sidebar.warning(f"Aviso ao instanciar ChatNVIDIA: {e}")


# =====================================================================
# CABEÇALHO PRINCIPAL
# =====================================================================
st.title("🛡️ Sistema Inteligente de Auditoria e Governança LGPD")
st.markdown("""
Plataforma orientada a IA para conformidade regulatória com a **Lei Geral de Proteção de Dados (Lei nº 13.709/2018)**.
Utiliza **RAG com FAISS**, **Hugging Face Multilingual Embeddings**, **LangChain** e orquestração com **NVIDIA AI Endpoints**.
""")

# Abas da Interface
aba1, aba2 = st.tabs(["💬 Chat de Consultoria LGPD", "📁 Auditor de Documentos do Drive"])


# ---------------------------------------------------------------------
# ABA 1: CHAT RAG DE CONSULTORIA
# ---------------------------------------------------------------------
with aba1:
    st.subheader("📖 Consultoria Regulatória Especializada")
    st.markdown("Esclareça dúvidas legais com respostas fundamentadas diretamente no texto oficial da lei.")

    # Inicializar histórico do chat na sessão
    if "mensagens_chat" not in st.session_state:
        st.session_state.mensagens_chat = [
            {
                "role": "assistant",
                "content": "Olá! Sou seu Consultor Regulatório de LGPD. Como posso auxiliar na conformidade da sua organização hoje? Você pode me perguntar sobre bases legais, direitos dos titulares, dados sensíveis ou incidentes de segurança."
            }
        ]

    # Sugestões rápidas de perguntas
    st.markdown("**Perguntas Frequentes:**")
    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    pergunta_rapida = None
    if col_q1.button("📋 Bases Legais – Art. 7º", use_container_width=True):
        pergunta_rapida = "Quais são as 10 bases legais previstas no Artigo 7º para o tratamento de dados pessoais?"
    if col_q2.button("🧬 Dados Sensíveis – Art. 11", use_container_width=True):
        pergunta_rapida = "Quais são as hipóteses legais para tratamento de dados sensíveis segundo o Art. 11?"
    if col_q3.button("👤 Direitos dos Titulares – Art. 18", use_container_width=True):
        pergunta_rapida = "Quais direitos o titular pode exercer conforme o Artigo 18 da LGPD?"
    if col_q4.button("🚨 Incidentes de Segurança – Art. 48", use_container_width=True):
        pergunta_rapida = "Quais medidas o controlador deve adotar em caso de vazamento ou incidente de segurança (Art. 48)?"

    # Exibição do histórico de mensagens
    for msg in st.session_state.mensagens_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "fontes" in msg and msg["fontes"]:
                with st.expander("📚 Ver Artigos e Páginas Citadas (Chunks Recuperados)"):
                    for doc in msg["fontes"]:
                        st.markdown(f"**Página {doc.metadata.get('page', '?')}** | *{doc.metadata.get('source', 'LGPD')}*")
                        st.caption(doc.page_content)

    # Input do chat
    input_usuario = st.chat_input("Digite sua dúvida regulatória sobre a LGPD...")
    pergunta_final = pergunta_rapida or input_usuario

    if pergunta_final:
        # Registrar pergunta do usuário
        st.session_state.mensagens_chat.append({"role": "user", "content": pergunta_final})
        with st.chat_message("user"):
            st.markdown(pergunta_final)

        # Processar resposta via RAG
        with st.chat_message("assistant"):
            if vectorstore is None:
                st.error("Base vetorial da LGPD não está carregada. Verifique o arquivo PDF.")
            else:
                with st.spinner("Buscando artigos pertinentes e gerando fundamentação jurídica..."):
                    resultado = consultar_lgpd(pergunta_final, vectorstore, llm)
                    st.markdown(resultado["resposta"])
                    
                    if resultado["documentos_fonte"]:
                        with st.expander("📚 Ver Artigos e Páginas Citadas (Chunks Recuperados)"):
                            for doc in resultado["documentos_fonte"]:
                                st.markdown(f"**Página {doc.metadata.get('page', '?')}** | *{doc.metadata.get('source', 'LGPD')}*")
                                st.caption(doc.page_content)

                    # Salvar na sessão
                    st.session_state.mensagens_chat.append({
                        "role": "assistant",
                        "content": resultado["resposta"],
                        "fontes": resultado["documentos_fonte"]
                    })


# ---------------------------------------------------------------------
# ABA 2: AUDITOR DE PASTAS DO GOOGLE DRIVE
# ---------------------------------------------------------------------
with aba2:
    st.subheader("📁 Auditoria Automatizada de Documentos Corporativos")
    st.markdown("""
    O agente varre a pasta corporativa no Google Drive, lê contratos, termos e políticas internas,
    compara cada cláusula com a base vetorial da LGPD e emite um relatório detalhado de conformidade.
    """)

    # Painel de controle da varredura
    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        st.info(f"📍 **Pasta Alvo:** `{url_pasta_drive}` (ID: `{extrair_id_pasta(url_pasta_drive)}`)")
    with col_ctrl2:
        fonte_dados = st.selectbox(
            "Origem da Auditoria:",
            options=["Pasta do Google Drive", "Amostras Pré-Carregadas (Testes)", "Upload Manual de Arquivos"]
        )

    arquivos_para_auditar = []
    diretorio_temporario = None

    if fonte_dados == "Upload Manual de Arquivos":
        arquivos_up = st.file_uploader(
            "Envie arquivos para auditoria (.pdf, .docx, .txt)",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt"]
        )
        if arquivos_up:
            diretorio_temporario = tempfile.mkdtemp()
            for arq in arquivos_up:
                caminho_salvo = os.path.join(diretorio_temporario, arq.name)
                with open(caminho_salvo, "wb") as f:
                    f.write(arq.getbuffer())
                arquivos_para_auditar.append(caminho_salvo)

    elif fonte_dados == "Amostras Pré-Carregadas (Testes)":
        pasta_amostras = "amostras_documentos"
        if os.path.exists(pasta_amostras):
            arquivos_para_auditar = [
                os.path.join(pasta_amostras, f)
                for f in os.listdir(pasta_amostras)
                if f.endswith((".pdf", ".docx", ".txt"))
            ]
            st.caption(f"Encontrados {len(arquivos_para_auditar)} documentos de teste (Conformes e Não Conformes).")

    # Botão de Execução
    if st.button("🚀 Iniciar Auditoria dos Arquivos", type="primary", use_container_width=True):
        if vectorstore is None:
            st.error("Erro: A base vetorial da LGPD não foi carregada.")
            st.stop()

        with st.spinner("Obtendo documentos e executando auditoria de conformidade..."):
            # Se a fonte for o Google Drive
            if fonte_dados == "Pasta do Google Drive":
                diretorio_temporario = tempfile.mkdtemp(prefix="auditoria_drive_")
                try:
                    if modo_conexao == "Google Drive API v3 (Service Account JSON)":
                        if not caminho_credenciais_sa:
                            st.warning("Credenciais de Service Account não fornecidas. Tentando download público via link...")
                            arquivos_para_auditar = baixar_arquivos_drive_publico(url_pasta_drive, diretorio_temporario)
                        else:
                            arquivos_para_auditar = baixar_arquivos_drive_api(
                                url_pasta_drive, caminho_credenciais_sa, diretorio_temporario
                            )
                    else:
                        arquivos_para_auditar = baixar_arquivos_drive_publico(url_pasta_drive, diretorio_temporario)
                except Exception as e:
                    st.error(f"Erro ao acessar Google Drive: {e}")
                    st.info("💡 Você também pode testar usando a opção 'Amostras Pré-Carregadas (Testes)' ou 'Upload Manual de Arquivos'.")
                    arquivos_para_auditar = []

            # Verificar se encontrou arquivos
            if not arquivos_para_auditar:
                st.warning("Nenhum documento encontrado para auditoria. Verifique as permissões da pasta compartilhada ou utilize os arquivos de amostra.")
            else:
                # Processamento da auditoria
                resultados_auditoria = []
                barra_progresso = st.progress(0, text="Auditando documentos...")

                for idx, caminho in enumerate(arquivos_para_auditar):
                    nome_arq = Path(caminho).name
                    barra_progresso.progress(
                        (idx + 1) / len(arquivos_para_auditar),
                        text=f"Auditando documento ({idx+1}/{len(arquivos_para_auditar)}): {nome_arq}"
                    )

                    try:
                        texto_doc = extrair_texto_arquivo(caminho)
                        if not texto_doc.strip():
                            st.warning(f"Documento '{nome_arq}' está vazio ou sem texto extraível.")
                            continue

                        relatorio = auditar_documento(nome_arq, texto_doc, vectorstore, llm)
                        relatorio["_caminho"] = caminho
                        relatorio["_nome"] = nome_arq
                        relatorio["_texto"] = texto_doc
                        resultados_auditoria.append(relatorio)
                    except Exception as err_doc:
                        st.error(f"Erro ao processar {nome_arq}: {err_doc}")

                barra_progresso.empty()

                # -------------------------------------------------------------
                # EXIBIÇÃO DO PAINEL DE RESULTADOS
                # -------------------------------------------------------------
                total = len(resultados_auditoria)
                conformes = sum(1 for r in resultados_auditoria if r.get("conforme", False))
                reprovados = total - conformes
                taxa = (conformes / total * 100) if total > 0 else 0

                st.markdown("### 📊 Sumário da Auditoria")
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("Total de Documentos", total)
                with col_m2:
                    st.metric("🟢 Em Conformidade", conformes)
                with col_m3:
                    st.metric("🔴 Não Conformes", reprovados)
                with col_m4:
                    st.metric("Taxa de Conformidade", f"{taxa:.1f}%")

                st.divider()

                # Lista de cards para cada arquivo auditado
                st.markdown("### 📑 Relatório Detalhado por Documento")

                for res in resultados_auditoria:
                    is_ok = res.get("conforme", False)
                    nome = res.get("_nome", "Documento")
                    resumo = res.get("resumo_executivo", "")
                    detalhes = res.get("detalhes_analise", "")
                    artigos = res.get("artigos_citados", [])
                    infracoes = res.get("infracoes_ou_riscos", [])
                    pontos_ok = res.get("pontos_conformes", [])
                    recomendacoes = res.get("recomendacoes", [])

                    if is_ok:
                        # Card Verde de Conformidade
                        with st.container():
                            col_icon, col_content = st.columns([1, 11])
                            with col_icon:
                                st.markdown("## 🟢")
                            with col_content:
                                st.markdown(f"#### **{nome}**")
                                st.success(f"**STATUS: EM CONFORMIDADE**\n\n{resumo}")
                                
                                if pontos_ok:
                                    st.markdown("**Pontos de Conformidade Atendidos:**")
                                    for p in pontos_ok:
                                        st.markdown(f"- ✅ {p}")

                                if artigos:
                                    st.markdown("**Artigos da LGPD em consonância:**")
                                    st.markdown(" ".join([f"<span class='badge-artigo'>{art}</span>" for art in artigos]), unsafe_allow_html=True)

                            with st.expander(f"📄 Ver Texto Extraído ({nome})"):
                                st.text(res.get("_texto", "")[:1500] + "...")
                        st.divider()

                    else:
                        # Card Vermelho de Não-Conformidade
                        with st.container():
                            col_icon, col_content = st.columns([1, 11])
                            with col_icon:
                                st.markdown("## 🔴")
                            with col_content:
                                st.markdown(f"#### **{nome}**")
                                st.error(f"**STATUS: NÃO CONFORME (REPROVADO)**\n\n{resumo}")
                                
                                if infracoes:
                                    st.markdown("**🚨 Infrações / Riscos Identificados:**")
                                    for inf in infracoes:
                                        st.markdown(f"- ❌ **{inf}**")

                                if recomendacoes:
                                    st.markdown("**🔧 Ações Corretivas Recomendadas:**")
                                    for rec in recomendacoes:
                                        st.markdown(f"- 💡 {rec}")

                                if artigos:
                                    st.markdown("**Artigos / Princípios da Lei Violados:**")
                                    st.markdown(" ".join([f"<span class='badge-artigo'>{art}</span>" for art in artigos]), unsafe_allow_html=True)

                            with st.expander(f"📄 Ver Texto Extraído ({nome})"):
                                st.text(res.get("_texto", "")[:1500] + "...")
                        st.divider()

    # Rodapé da aplicação
    st.markdown("---")
    st.caption("🛡️ **Auditor Inteligente LGPD** • Projeto Acadêmico CESAR School • Desenvolvido com Streamlit, LangChain, FAISS e NVIDIA AI.")

