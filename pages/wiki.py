import streamlit as st
from app.document_loader import load_document, load_from_wikipedia
from app.embedding_store import insert_or_fetch_embeddings
from app.qa_system import ask_and_get_answer, ask_with_memory
from app.embedding_store import chunk_data
from app.utils import sanitize_index_name


st.title("LangChain X Pinecone | Document Q&A")

st.title("`Wikipedia Document Loader`")

wiki_query = st.text_input("Write your query here...")

# Initializing session state variables
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if wiki_query:
    with st.spinner("Processing data obtained from wikipedia..."):
        # Load document
        data = load_from_wikipedia(wiki_query)
        st.write(f"Loaded {len(data)} pages.")

        chunks = chunk_data(data)
        st.write(f"Document split into {len(chunks)} chunks.")
        
        suffix = sanitize_index_name(wiki_query)
        index_name = f"pinecone-{suffix}"
        
        # Create vector store and save to session state
        vector_store = insert_or_fetch_embeddings(index_name, chunks)
        st.session_state.vector_store = vector_store
        
        # Mark document as processed
        st.session_state.document_processed = True
        st.success("Embeddings created successfully!")

# Query section
if st.session_state.vector_store is not None:
    st.subheader("Ask a Question")
    query = st.text_input("Enter your question:")
    if query:
        answer = ask_and_get_answer(st.session_state.vector_store, query)
        st.write("Answer:")
        st.write(answer)

# Conversational Memory Section
if st.session_state.vector_store is not None:
    st.subheader("Chat with Memory")
    
    chat_input = st.text_input("Your question (with memory):", key="chat_memory")
    if st.button("Ask"):
        result, chat_history = ask_with_memory(
            st.session_state.vector_store, 
            chat_input, 
            st.session_state.chat_history
        )
        st.write(result["answer"])
        st.session_state.chat_history = chat_history

    # Display chat history
    st.subheader("Chat History")
    for q, a in st.session_state.chat_history:
        st.write(f"**Q**: {q}")
        st.write(f"**A**: {a}")

# Optional: Add a button to reset the document processing
if st.session_state.document_processed:
    if st.sidebar.button("Reset Document"):
        st.session_state.vector_store = None
        st.session_state.document_processed = False
        st.session_state.chat_history = []
        st.rerun()