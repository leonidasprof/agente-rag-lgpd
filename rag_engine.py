import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
except ImportError:
    ChatNVIDIA = None

# Configurações padrão
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_NVIDIA_MODEL = "stepfun-ai/step-3.5-flash"
FAISS_CACHE_DIR = "faiss_index_lgpd"


def extrair_documentos_lgpd(caminho_pdf: str) -> List[Document]:
    """
    Lê o PDF oficial da LGPD página por página usando PyMuPDF (fitz)
    e retorna uma lista de Documentos com metadados de página e fonte.
    """
    if not os.path.exists(caminho_pdf):
        raise FileNotFoundError(f"Arquivo PDF da LGPD não encontrado em: {caminho_pdf}")

    doc = fitz.open(caminho_pdf)
    documentos = []

    for num_pagina, pagina in enumerate(doc):
        texto = pagina.get_text()
        if texto.strip():
            doc_obj = Document(
                page_content=texto,
                metadata={
                    "source": "Lei nº 13.709/2018 (LGPD)",
                    "page": num_pagina + 1,
                    "arquivo": Path(caminho_pdf).name
                }
            )
            documentos.append(doc_obj)

    return documentos


def criar_ou_carregar_vectorstore(
    caminho_pdf: str,
    diretorio_cache: str = FAISS_CACHE_DIR,
    forcar_recriar: bool = False
) -> FAISS:
    """
    Inicializa os embeddings HuggingFace e cria/carrega a base vetorial FAISS.
    Usa persistência local para carregamento instantâneo.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=DEFAULT_EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Se já existir o índice persistido e não forçar recriação, carrega diretamente
    index_path = Path(diretorio_cache)
    if not forcar_recriar and index_path.exists() and (index_path / "index.faiss").exists():
        try:
            return FAISS.load_local(
                folder_path=diretorio_cache,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception:
            pass  # Se falhar ao carregar, reconstrói

    # Extrai o texto do PDF oficial
    docs = extrair_documentos_lgpd(caminho_pdf)

    # Fatiamento em chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=140,
        separators=["\n\nArt. ", "\nArt. ", "\n\n§ ", "\n§ ", "\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    # Indexação no FAISS
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Salva no disco para acessos futuros
    os.makedirs(diretorio_cache, exist_ok=True)
    vectorstore.save_local(diretorio_cache)

    return vectorstore


def obter_llm(api_key: Optional[str] = None, model_name: str = DEFAULT_NVIDIA_MODEL):
    """
    Inicializa o modelo ChatNVIDIA configurado com temperatura zero
    e proteção rigorosa contra alucinações.
    """
    key = api_key or os.getenv("NVIDIA_API_KEY")
    if not key:
        return None

    if ChatNVIDIA is None:
        raise ImportError("O pacote 'langchain-nvidia-ai-endpoints' não está disponível.")

    return ChatNVIDIA(
        model=model_name,
        api_key=key,
        temperature=0.0,
        max_tokens=1500
    )


# ---------------------------------------------------------------------
# PROMPTS DE CONSULTORIA E AUDITORIA
# ---------------------------------------------------------------------

PROMPT_CHAT_CONSULTORIA = """Você é um Consultor Jurídico Sênior e Auditor de Conformidade especializado estritamente na Lei Geral de Proteção de Dados Pessoais (LGPD - Lei nº 13.709/2018).

DIRETRIZES E RESTRIÇÕES SEVERAS:
1. Baseie sua resposta EXCLUSIVAMENTE nas informações contidas no CONTEXTO LEGAL fornecido abaixo.
2. Não invente, não presuma e não utilize conhecimentos externos conflitantes. Se o contexto não contiver a resposta, afirme explicitamente: "A legislação consultada nos trechos recuperados não aborda diretamente este ponto específico."
3. Sempre cite os artigos, parágrafos, incisos e páginas específicas da lei que fundamentam seu posicionamento.
4. Mantenha um tom profissional, técnico, preciso e estruturado em formato markdown.

CONTEXTO LEGAL DA LGPD:
{contexto}

DÚVIDA DO USUÁRIO:
{pergunta}

RESPOSTA JURÍDICO-REGULATÓRIA FUNDAMENTADA:"""


PROMPT_AUDITORIA_DOCUMENTO = """Você é um Auditor Especialista Sênior em Segurança da Informação e Conformidade LGPD (Lei nº 13.709/2018).
Sua missão é auditar o documento corporativo fornecido, confrontando cláusula por cláusula contra os preceitos da LGPD fornecidos no contexto legal.

PONTOS CRÍTICOS DE AUDITORIA:
- Existência de base legal clara para tratamento de dados pessoais (Art. 7º) e dados sensíveis (Art. 11).
- Cumprimento dos princípios do Art. 6º (finalidade, adequação, necessidade/minimização, transparência, segurança, prevenção, não discriminação).
- Direitos dos titulares garantidos de forma clara e acessível (Art. 18).
- Medidas técnicas de segurança e plano de resposta a incidentes (Art. 46, 48).
- Cláusulas abusivas (ex: autorização genérica, cessão irrestrita de dados a terceiros sem finalidade ou consentimento).

CONTEXTO LEGAL DA LGPD:
{contexto_legal}

DADOS DO DOCUMENTO A SER AUDITADO:
Nome do Arquivo: {nome_arquivo}
Conteúdo do Documento:
\"\"\"
{conteudo_documento}
\"\"\"

Você DEVE responder ESTRITAMENTE em formato JSON puro, sem crases de código (```json ou ```) e sem texto antes ou depois. Use exatamente este formato:
{{
  "conforme": true/false,
  "status": "CONFORME" ou "NÃO CONFORME",
  "resumo_executivo": "Resumo de 2 a 3 frases com o parecer geral",
  "detalhes_analise": "Análise crítica do documento",
  "pontos_conformes": ["Ponto conforme 1", "Ponto conforme 2"],
  "infracoes_ou_riscos": ["Infração/Risco 1 com citação do artigo violado", "Infração/Risco 2"],
  "artigos_citados": ["Art. X", "Art. Y"],
  "recomendacoes": ["Recomendação prática de adequação 1", "Recomendação 2"]
}}"""


def consultar_lgpd(pergunta: str, vectorstore: FAISS, llm: Any) -> Dict[str, Any]:
    """
    Executa busca semântica na LGPD e gera resposta fundamentada com o LLM.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs_relevantes = retriever.invoke(pergunta)

    # Formatar contexto com fontes e páginas
    trechos = []
    for i, doc in enumerate(docs_relevantes):
        pagina = doc.metadata.get("page", "?")
        trechos.append(f"[Trecho {i+1} - Página {pagina}]:\n{doc.page_content.strip()}")
    contexto_str = "\n\n---\n\n".join(trechos)

    if llm is None:
        return {
            "resposta": "⚠️ **Chave da API NVIDIA não configurada.**\n\nPor favor, insira sua `NVIDIA_API_KEY` na barra lateral para obter pareceres gerados pelo modelo `stepfun-ai/step-3.5-flash`.\n\n### Trechos mais relevantes encontrados na LGPD:\n\n" + contexto_str,
            "documentos_fonte": docs_relevantes
        }

    prompt = ChatPromptTemplate.from_template(PROMPT_CHAT_CONSULTORIA)
    chain = prompt | llm | StrOutputParser()

    resposta = chain.invoke({
        "contexto": contexto_str,
        "pergunta": pergunta
    })

    return {
        "resposta": resposta,
        "documentos_fonte": docs_relevantes
    }


def auditar_documento(
    nome_arquivo: str,
    conteudo_texto: str,
    vectorstore: FAISS,
    llm: Any
) -> Dict[str, Any]:
    """
    Audita um documento contra a base da LGPD e retorna o relatório estruturado.
    """
    # Limita tamanho de texto analisado se for excessivamente longo
    texto_resumido = conteudo_texto[:7000]

    # Busca trechos relevantes na LGPD baseando-se no texto do documento
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    query_busca = f"conformidade bases legais consentimento tratamento dados segurança {texto_resumido[:500]}"
    docs_relevantes = retriever.invoke(query_busca)

    contexto_legal = "\n\n".join([
        f"[Página {d.metadata.get('page', '?')}]: {d.page_content.strip()}"
        for d in docs_relevantes
    ])

    if llm is None:
        # Modo de contingência quando a API Key ainda não foi inserida
        texto_lower = conteudo_texto.lower()
        nao_conforme_triggers = [
            "sem consentimento", "sem aviso", "indeterminada", "compartilhar livremente",
            "todos os dados", "qualquer finalidade", "renuncia a direitos", "dados sensíveis"
        ]
        infracoes = [t for t in nao_conforme_triggers if t in texto_lower]
        
        is_conforme = len(infracoes) == 0
        return {
            "conforme": is_conforme,
            "status": "CONFORME" if is_conforme else "NÃO CONFORME",
            "resumo_executivo": f"Análise preliminar em modo de demonstração (API Key não informada). Documento {'aprovado nos critérios básicos' if is_conforme else 'apresenta termos de alto risco'}.",
            "detalhes_analise": f"Varredura por termos de risco detectou: {', '.join(infracoes) if infracoes else 'Nenhum padrão crítico básico identificado'}.",
            "pontos_conformes": ["Identificação de partes", "Estrutura documental padrão"] if is_conforme else [],
            "infracoes_ou_riscos": [f"Termo suspeito identificado: '{inf}' (possível violação de princípios dos Arts. 6º e 7º)" for inf in infracoes],
            "artigos_citados": ["Art. 6º", "Art. 7º", "Art. 18"],
            "recomendacoes": ["Insira sua NVIDIA_API_KEY na barra lateral para auditoria completa e aprofundada com IA."]
        }

    prompt = ChatPromptTemplate.from_template(PROMPT_AUDITORIA_DOCUMENTO)
    chain = prompt | llm | StrOutputParser()

    resposta_raw = chain.invoke({
        "contexto_legal": contexto_legal,
        "nome_arquivo": nome_arquivo,
        "conteudo_documento": texto_resumido
    })

    # Tratamento de parsing JSON robusto
    try:
        # Remover blocos markdown se existirem
        limpo = re.sub(r"^```(?:json)?\s*", "", resposta_raw.strip(), flags=re.MULTILINE)
        limpo = re.sub(r"\s*```$", "", limpo, flags=re.MULTILINE).strip()
        dados = json.loads(limpo)
        return dados
    except Exception:
        # Fallback de parsing se o modelo incluir texto adicional
        match = re.search(r'\{.*\}', resposta_raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

        # Fallback estruturado manual
        return {
            "conforme": "NÃO CONFORME" not in resposta_raw.upper(),
            "status": "CONFORME" if "NÃO CONFORME" not in resposta_raw.upper() else "NÃO CONFORME",
            "resumo_executivo": "Parecer emitido pelo auditor de IA.",
            "detalhes_analise": resposta_raw,
            "pontos_conformes": [],
            "infracoes_ou_riscos": ["Necessária revisão manual detalhada do parecer."],
            "artigos_citados": ["Art. 6º", "Art. 7º"],
            "recomendacoes": ["Ajustar cláusulas conforme os artigos citados."]
        }

