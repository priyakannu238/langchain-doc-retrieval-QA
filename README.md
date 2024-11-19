# LangChain Document Retrieval and Question-Answering System

A modular, scalable system designed for conversational document retrieval and question-answering. It supports PDF, DOCX, and TXT files, leverages OpenAI embeddings, and uses Chroma vector stores for efficient vector-based search. The project includes a Streamlit-based user interface for seamless document interaction.

---

## Features

- **Multi-format Document Support**: Load and process PDF, DOCX, and TXT files into embeddings.
- **Vector Storage**: Efficient document storage using Chroma vector database.
- **Conversational Question-Answering**: Supports memory-based question-answering for interactive conversations.
- **Streamlit GUI**: User-friendly interface for uploading documents, querying, and retrieving answers.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI API Key (stored in a `.env` file)

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/priyakannu238/langchain-doc-retrieval-QA.git
   cd langchain-doc-retrieval-QA
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up API Key**:
   Create a `.env` file in the `config/` directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

4. **Run the Application**:
   ```bash
   streamlit run app/main.py
   ```

---

## Usage

1. Upload a document in PDF, DOCX, or TXT format through the Streamlit GUI.
2. Ask questions based on the content of the document.
3. View detailed responses or engage in memory-based conversations with the document.

---

## Example Queries

- *"What is the document about?"*
- *"Summarize the key points."*
- *"Explain the section on 'resoning and acting in LLMs'."*

---

## Dependencies

- [Streamlit](https://streamlit.io/)
- [LangChain](https://github.com/hwchase17/langchain)
- [Chroma](https://www.trychroma.com/)
- [OpenAI](https://platform.openai.com/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)
- [python-docx](https://python-docx.readthedocs.io/)

---

## Future Enhancements

- Add support for multilingual document embeddings.
- Integrate more advanced memory and conversational models.
- Extend GUI for analytics and visualization of embedding statistics.
