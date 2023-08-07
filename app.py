import streamlit as st
import pickle
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain


#Siderbar contents
with st.sidebar:
    st.title('Kyra Chat Bot')
    st.markdown('''
    ## About
    This app is an LLM-Powered chatbotbuilt using
    - [streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/docs/models)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    
    ''')
    add_vertical_space(5)
    st.write('Made by Vermeg DATA team')



def main():
    st.header("Kyra ChatBot")
    load_dotenv()


    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type= 'pdf')

    if pdf is not None:

        pdf_reader = PdfReader(pdf)
        #st.write(pdf_reader)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)

        # embeddings
        
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
        else: 
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)      
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
        
        # Accept user questions/query
        query = st.text_input("Ask question about you PDF file:")

        if query:

            docs = VectorStore.similarity_search(query=query, k=3)

            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write(response)



if __name__ == '__main__':
    main()