import os
import io
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
import docx

try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
except ImportError:
    build = None
    service_account = None
    MediaIoBaseDownload = None

try:
    import gdown
except ImportError:
    gdown = None


def extrair_id_pasta(url_ou_id: str) -> str:
    """Extrai o ID da pasta a partir de uma URL do Google Drive ou retorna o próprio ID."""
    padrao = r'folders/([a-zA-Z0-9_-]+)'
    match = re.search(padrao, url_ou_id)
    if match:
        return match.group(1)
    # Se for uma URL com id=
    padrao_id = r'id=([a-zA-Z0-9_-]+)'
    match_id = re.search(padrao_id, url_ou_id)
    if match_id:
        return match_id.group(1)
    return url_ou_id.strip()


def extrair_texto_arquivo(caminho_arquivo: str) -> str:
    """
    Extrai o texto de arquivos suportados (.pdf, .docx, .txt, .md).
    """
    path = Path(caminho_arquivo)
    ext = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_arquivo}")

    if ext == ".pdf":
        doc = fitz.open(caminho_arquivo)
        texto = []
        for i, pagina in enumerate(doc):
            t = pagina.get_text()
            if t.strip():
                texto.append(f"[Página {i + 1}]\n{t}")
        return "\n\n".join(texto)

    elif ext == ".docx":
        doc = docx.Document(caminho_arquivo)
        paragrafos = [p.text for p in doc.paragraphs if p.text.strip()]
        # Também extrair tabelas se existirem
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    paragrafos.append(row_text)
        return "\n\n".join(paragrafos)

    elif ext in [".txt", ".md", ".rtf", ".csv", ".json"]:
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                with open(caminho_arquivo, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        return path.read_text(errors="ignore")

    else:
        raise ValueError(f"Extensão de arquivo não suportada: {ext}")


def baixar_arquivos_drive_publico(url_ou_id: str, destino_dir: str) -> List[str]:
    """
    Tenta baixar os arquivos de uma pasta compartilhada pública usando gdown.
    """
    folder_id = extrair_id_pasta(url_ou_id)
    os.makedirs(destino_dir, exist_ok=True)
    
    if gdown is None:
        raise ImportError("O pacote 'gdown' não está disponível.")

    url = f"https://drive.google.com/drive/folders/{folder_id}?usp=sharing"
    
    # Download da pasta via gdown
    try:
        baixados = gdown.download_folder(
            url=url,
            output=destino_dir,
            quiet=True,
            use_cookies=False,
            remaining_ok=True
        )
    except Exception as e:
        # Tentar com download recursivo de pasta
        baixados = []
        try:
            gdown.download_folder(id=folder_id, output=destino_dir, quiet=True)
        except Exception as e_inner:
            raise RuntimeError(f"Erro ao baixar pasta pública via gdown: {e_inner}")

    # Coletar caminhos de todos os arquivos baixados
    arquivos_encontrados = []
    for raiz, _, arquivos in os.walk(destino_dir):
        for arq in arquivos:
            ext = Path(arq).suffix.lower()
            if ext in [".pdf", ".docx", ".txt", ".md"]:
                arquivos_encontrados.append(os.path.join(raiz, arq))

    return arquivos_encontrados


def baixar_arquivos_drive_api(folder_id_ou_url: str, credenciais_path: str, destino_dir: str) -> List[str]:
    """
    Varre e baixa arquivos de uma pasta do Google Drive utilizando credenciais de Service Account.
    """
    if build is None or service_account is None:
        raise ImportError("Google Client API não está instalada.")

    folder_id = extrair_id_pasta(folder_id_ou_url)
    os.makedirs(destino_dir, exist_ok=True)

    scopes = ['https://www.googleapis.com/auth/drive.readonly']
    creds = service_account.Credentials.from_service_account_file(
        credenciais_path, scopes=scopes
    )
    service = build('drive', 'v3', credentials=creds)

    # Buscar itens dentro da pasta especificada
    query = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(
        q=query,
        fields="files(id, name, mimeType)",
        pageSize=100
    ).execute()

    files = results.get('files', [])
    arquivos_baixados = []

    for f in files:
        file_id = f['id']
        file_name = f['name']
        mime_type = f.get('mimeType', '')

        # Se for Google Docs nativo, exporta como PDF
        if mime_type == 'application/vnd.google-apps.document':
            request = service.files().export_media(fileId=file_id, mimeType='application/pdf')
            if not file_name.endswith('.pdf'):
                file_name += '.pdf'
        elif mime_type == 'application/vnd.google-apps.folder':
            # Ignora subpastas para varredura plana ou recursiva
            continue
        else:
            request = service.files().get_media(fileId=file_id)

        destino_arquivo = os.path.join(destino_dir, file_name)
        with open(destino_arquivo, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

        arquivos_baixados.append(destino_arquivo)

    return arquivos_baixados

