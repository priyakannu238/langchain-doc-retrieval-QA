import os
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import WikipediaLoader

def load_document(uploaded_file):
    # Creating a temporary file to save the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # file extension
    name, extension = os.path.splitext(uploaded_file.name)

    try:
        if extension.lower() == '.pdf':
            print(f'Loading {uploaded_file.name}')
            loader = PyPDFLoader(tmp_file_path)
        elif extension.lower() == '.docx':
            print(f'Loading {uploaded_file.name}')
            loader = Docx2txtLoader(tmp_file_path)
        elif extension.lower() == '.txt':
            loader = TextLoader(tmp_file_path)
        else:
            print('Document format is not supported!')
            return None

        data = loader.load()
        return data
    finally:
        # Clean up the temporary file
        os.unlink(tmp_file_path)


# wiki
def load_from_wikipedia(query, lang='en', load_max_docs=2):
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    data = loader.load()
    return data

  