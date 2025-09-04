import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from jira import JIRA
from langchain.docstore.document import Document
from langchain_core.runnables import RunnablePassthrough

# --- 1. CONFIGURACIÓN INICIAL Y CONEXIÓN A JIRA ---
load_dotenv()
JIRA_SERVER = os.getenv("JIRA_SERVER")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")

if not JIRA_SERVER or not JIRA_EMAIL or not JIRA_API_TOKEN:
    print("Error: The JIRA credentials were not loaded. Please check your .env file.")
    exit()

try:
    jira_options = {'server': JIRA_SERVER}
    jira_connection = JIRA(options=jira_options, basic_auth=(JIRA_EMAIL, JIRA_API_TOKEN))
except Exception as e:
    print(f"Error connecting to Jira: {e}")
    exit()

pdf_path = "2020-Scrum-Guide-US.pdf"
model_path = "Lexi-Llama-3-8B-Uncensored_Q4_K_M.gguf"
persist_directory = "chroma_db"

# --- 2. FUNCIONES DE EXTRACCIÓN Y CREACIÓN DE LA BASE DE CONOCIMIENTO UNIFICADA ---

def obtener_documentos_de_jira():
    """
    Obtiene issues de un proyecto de Jira y los convierte en documentos de texto.
    """
    try:
        proyecto = "My RAG app" 
        tipo_issue = "Story"
        jql_query = f'project = "{proyecto}" AND issuetype = "{tipo_issue}" ORDER BY created DESC'
        issues = jira_connection.search_issues(jql_query, maxResults=50)

        documentos_jira = []
        for issue in issues:
            texto_issue = f"ID: {issue.key}\nResumen: {issue.fields.summary}\nDescripción: {issue.fields.description}\nEstado: {issue.fields.status.name}\nTipo: {issue.fields.issuetype.name}"
            documentos_jira.append(Document(page_content=texto_issue))
        
        return documentos_jira
    except Exception as e:
        print(f"Error al obtener datos de Jira: {e}")
        return []

def crear_base_de_conocimiento_unificada():
    """
    Carga documentos del PDF y datos de Jira, los combina y crea una única base de datos vectorial.
    """
    print("Cargando la Guía de Scrum...")
    loader = PyPDFLoader(pdf_path)
    documents_pdf = loader.load()

    print("Obteniendo documentos de Jira...")
    documents_jira = obtener_documentos_de_jira()
    
    all_documents = documents_pdf + documents_jira
    if not all_documents:
        print("No se encontraron documentos en ninguna fuente. No se creará la base de conocimiento.")
        return None

    print(f"Número de documentos cargados: {len(all_documents)}")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(all_documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    print("Base de conocimiento unificada (PDF y Jira) creada y persistida.")
    return db

# --- 3. FUNCIÓN PARA CONSULTAR LA BASE DE CONOCIMIENTO ---

def consultar_base_de_conocimiento(query):
    """
    Traduce la pregunta del usuario, busca en la base de datos y genera una respuesta con el LLM.
    """
    try:
        translated_query = GoogleTranslator(source='es', target='en').translate(query)
        print(f"Pregunta traducida (a inglés): {translated_query}")
    except Exception as e:
        print(f"Error al traducir la pregunta: {e}. Usando la pregunta original.")
        translated_query = query

    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=-1,
        n_batch=512,
        n_ctx=8192,
        callback_manager=None,
        verbose=False,
        temperature=0.1,  # Reducir la temperatura para respuestas más concisas
        max_tokens=1024,
        stop=["<|eot_id|>", "Question:"] # Detiene la generación cuando el modelo intenta responder otra pregunta
    )

    # Revertir a un prompt simple y efectivo
    template = """Use the following pieces of context to answer the user's question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Answer only the user's question directly, without adding extra comments, questions, or conversation.

{context}

Question: {question}
Helpful Answer:"""

    qa_prompt = PromptTemplate(
        template=template, 
        input_variables=["context", "question"]
    )

    # Usar RetrievalQA que es más simple y confiable con prompts simples
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": qa_prompt}
    )
    
    # La cadena se invoca solo con la pregunta
    result_en = qa.invoke({"query": translated_query})

    # Extracción de la respuesta
    if isinstance(result_en, dict) and 'result' in result_en:
        result_text_en = result_en['result']
    else:
        result_text_en = str(result_en)

    try:
        result_es = GoogleTranslator(source='en', target='es').translate(result_text_en)
    except Exception as e:
        print(f"Error al traducir la respuesta: {e}. Mostrando respuesta en inglés.")
        result_es = result_text_en

    print(f"Pregunta original: {query}")
    print(f"Respuesta (en español): {result_es}")

# --- 4. PUNTO DE ENTRADA DEL SCRIPT ---

if __name__ == "__main__":
    if not os.path.exists(persist_directory):
        print("La base de conocimiento no existe. Creando una nueva...")
        db_creada = crear_base_de_conocimiento_unificada()
        if db_creada is None:
            exit()
    else:
        print("La base de conocimiento ya existe. Omitiendo la creación y cargando la existente.")

    while True:
        pregunta = input("Ingresa tu pregunta (o 'salir' para terminar): ")
        if pregunta.lower() == "salir":
            break
        consultar_base_de_conocimiento(pregunta)