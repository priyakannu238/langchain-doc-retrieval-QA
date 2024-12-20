# LangChain Document Retrieval and Question-Answering System

A modular, scalable system designed for conversational document retrieval and question-answering. It supports PDF, DOCX, and TXT files, leverages OpenAI embeddings, and uses Pinecone vector stores for efficient vector-based search. The project includes a Streamlit-based user interface for seamless document interaction.

---

## Interface
![image](https://github.com/user-attachments/assets/7e1c6a7b-4142-4c07-bb14-e4415ba7bc1b)

![image](https://github.com/user-attachments/assets/412fd2e6-def0-413f-a468-4e848210690d)


## Features

- **Multi-format Document Support**: Load and process PDF, DOCX, and TXT files into embeddings.
- **Vector Storage**: Efficient document storage using Pinecone vector database.
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
   PINECONE_API_KEY=your_pinecone_api_key
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
- [Pinecone](https://www.tryPinecone.com/)
- [OpenAI](https://platform.openai.com/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)
- [python-docx](https://python-docx.readthedocs.io/)

---

## Future Enhancements

- Add support for multilingual document embeddings.
- Integrate more advanced memory and conversational models.
- Extend GUI for analytics and visualization of embedding statistics.
