from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_openai import ChatOpenAI


def ask_and_get_answer(vector_store, q, k=10):
    llm = ChatOpenAI(model='gpt-4o', temperature=0.1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever) 
    answer = chain.invoke(q)
    return answer

def ask_with_memory(vector_store, question, chat_history=[]):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    crc = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.1), retriever)
    result = crc({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    return result, chat_history
