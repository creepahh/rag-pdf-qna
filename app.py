import os
import tempfile

# Disable Chroma telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
import ollama
import streamlit as st
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder


# ---------------------------------------------------------
# STREAMLIT CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="RAG PDF Q&A",
    page_icon="📄",
    layout="wide",
)


# ---------------------------------------------------------
# SYSTEM PROMPT
# ---------------------------------------------------------

system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely
on the given context.

Your goal is to analyze the information provided and formulate a
comprehensive, well-structured response to the question.

The context will be provided as "Context:"
The user question will be provided as "Question:"

Rules:

1. Thoroughly analyze the context.
2. Identify the information relevant to the question.
3. Answer using ONLY the provided context.
4. Do not use outside knowledge or assumptions.
5. If the context does not contain enough information, clearly say so.
6. Give a concise but useful answer.
7. Use bullet points or numbered lists where appropriate.
8. Keep the response easy to read.

Important:
Base your entire response solely on the information provided in the context.
Do not include external knowledge or assumptions.
"""


# ---------------------------------------------------------
# PROCESS PDF
# ---------------------------------------------------------

def process_document(uploaded_file):
    """
    Save uploaded PDF temporarily, extract its text,
    split it into chunks, then return the chunks.
    """

    temp_file = tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".pdf",
        delete=False,
    )

    try:
        temp_file.write(uploaded_file.getvalue())
        temp_file.close()

        loader = PyMuPDFLoader(temp_file.name)
        docs = loader.load()

    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            ".",
            "?",
            "!",
            " ",
            "",
        ],
    )

    return text_splitter.split_documents(docs)


# ---------------------------------------------------------
# CHROMA VECTOR DATABASE
# ---------------------------------------------------------

@st.cache_resource
def get_vector_collection():

    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    chroma_client = chromadb.PersistentClient(
        path="./demo-rag-chroma"
    )

    collection = chroma_client.get_or_create_collection(
        name="rag_app",
        embedding_function=ollama_ef,
        metadata={
            "hnsw:space": "cosine"
        },
    )

    return collection


# ---------------------------------------------------------
# ADD DOCUMENT TO VECTOR DATABASE
# ---------------------------------------------------------

def add_to_vector_collection(all_splits, file_name):

    collection = get_vector_collection()

    documents = []
    metadatas = []
    ids = []

    for idx, split in enumerate(all_splits):

        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

    return len(documents)


# ---------------------------------------------------------
# QUERY VECTOR DATABASE
# ---------------------------------------------------------

def query_collection(prompt, n_results=10):

    collection = get_vector_collection()

    results = collection.query(
        query_texts=[prompt],
        n_results=n_results,
    )

    return results


# ---------------------------------------------------------
# RE-RANK RESULTS
# ---------------------------------------------------------

@st.cache_resource
def get_cross_encoder():

    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


def re_rank_cross_encoders(
    query,
    documents,
    top_k=3,
):

    if not documents:
        return "", []

    encoder_model = get_cross_encoder()

    top_k = min(top_k, len(documents))

    ranks = encoder_model.rank(
        query,
        documents,
        top_k=top_k,
    )

    relevant_text = ""
    relevant_text_ids = []

    for rank in ranks:

        corpus_id = rank["corpus_id"]

        relevant_text += (
            documents[corpus_id]
            + "\n\n"
        )

        relevant_text_ids.append(corpus_id)

    return relevant_text, relevant_text_ids


# ---------------------------------------------------------
# CALL OLLAMA
# ---------------------------------------------------------

def call_llm(context, prompt):

    response = ollama.chat(
        model="llama3.2",
        stream=True,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{prompt}"
                ),
            },
        ],
    )

    for chunk in response:

        if not chunk.get("done", False):

            message = chunk.get("message", {})

            content = message.get(
                "content",
                "",
            )

            if content:
                yield content


# ---------------------------------------------------------
# CHECK OLLAMA CONNECTION
# ---------------------------------------------------------

def check_ollama_connection():

    try:

        ollama.list()

        return True

    except Exception:

        return False


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []


if "doc_info" not in st.session_state:

    st.session_state.doc_info = None


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"],
        accept_multiple_files=False,
    )

    if uploaded_file:

        st.info(
            f"**File:** {uploaded_file.name}"
        )

    process = st.button(
        "Process Document",
        use_container_width=True,
    )

    if uploaded_file and process:

        with st.spinner(
            "Processing PDF..."
        ):

            try:

                all_splits = process_document(
                    uploaded_file
                )

                normalized_name = uploaded_file.name.translate(
                    str.maketrans(
                        {
                            "-": "_",
                            ".": "_",
                            " ": "_",
                        }
                    )
                )

                num_chunks = add_to_vector_collection(
                    all_splits,
                    normalized_name,
                )

                st.session_state.doc_info = {
                    "name": uploaded_file.name,
                    "chunks": num_chunks,
                }

                st.success(
                    f"Indexed {num_chunks} chunks "
                    f"from **{uploaded_file.name}**"
                )

            except Exception as e:

                st.error(
                    f"Processing failed: {e}"
                )

    st.divider()

    st.header("📊 Document Status")

    if st.session_state.doc_info:

        st.write(
            f"**Loaded:** "
            f"{st.session_state.doc_info['name']}"
        )

        st.write(
            f"**Chunks:** "
            f"{st.session_state.doc_info['chunks']}"
        )

    else:

        st.caption(
            "No document loaded yet."
        )

    st.divider()

    if st.button(
        "Clear Chat History",
        use_container_width=True,
    ):

        st.session_state.chat_history = []

        st.rerun()


# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------

st.title("📄 RAG PDF Q&A")

st.caption(
    "Upload a PDF and ask questions about its contents."
)


# ---------------------------------------------------------
# DISPLAY CHAT HISTORY
# ---------------------------------------------------------

for msg in st.session_state.chat_history:

    with st.chat_message(
        msg["role"]
    ):

        st.markdown(
            msg["content"]
        )


# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------

prompt = st.chat_input(
    "Ask a question about your document..."
)


if prompt:

    # ---------------------------------------------
    # CHECK DOCUMENT
    # ---------------------------------------------

    if not st.session_state.doc_info:

        st.warning(
            "Please upload and process a PDF first."
        )

        st.stop()


    # ---------------------------------------------
    # CHECK OLLAMA
    # ---------------------------------------------

    if not check_ollama_connection():

        st.error(
            "Cannot connect to Ollama. "
            "Make sure Ollama is running on "
            "`localhost:11434`."
        )

        st.stop()


    # ---------------------------------------------
    # ADD USER MESSAGE
    # ---------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):

        st.markdown(prompt)


    # ---------------------------------------------
    # ASSISTANT RESPONSE
    # ---------------------------------------------

    with st.chat_message("assistant"):

        # -----------------------------------------
        # RETRIEVE
        # -----------------------------------------

        with st.spinner(
            "Retrieving relevant context..."
        ):

            try:

                results = query_collection(
                    prompt,
                    n_results=10,
                )

                docs = results.get(
                    "documents",
                    [[]],
                )[0]

                if not docs:

                    st.warning(
                        "No relevant information "
                        "was found in the document."
                    )

                    st.stop()

            except Exception as e:

                st.error(
                    f"Vector query failed: {e}"
                )

                st.stop()


        # -----------------------------------------
        # RE-RANK
        # -----------------------------------------

        with st.spinner(
            "Finding the most relevant sections..."
        ):

            try:

                relevant_text, _ = (
                    re_rank_cross_encoders(
                        prompt,
                        docs,
                        top_k=3,
                    )
                )

            except Exception as e:

                st.error(
                    f"Re-ranking failed: {e}"
                )

                st.stop()


        # -----------------------------------------
        # GENERATE ANSWER
        # -----------------------------------------

        with st.spinner(
            "Generating answer..."
        ):

            response_placeholder = st.empty()

            full_response = ""

            try:

                for token in call_llm(
                    relevant_text,
                    prompt,
                ):

                    full_response += token

                    response_placeholder.markdown(
                        full_response + "▌"
                    )

                response_placeholder.markdown(
                    full_response
                )

            except Exception as e:

                st.error(
                    f"LLM call failed: {e}"
                )

                st.stop()


    # ---------------------------------------------
    # SAVE ASSISTANT RESPONSE
    # ---------------------------------------------

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": full_response,
        }
    )