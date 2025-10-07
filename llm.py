from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

file_path = "./p2.pdf"
loader = PyPDFLoader(file_path)

pages = []
for page in loader.load():
    pages.append(page.page_content)

full_text = '\n'.join(pages)
# print(full_text[:500])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

chunks = text_splitter.split_text(full_text)
# print("\nTotal Chunks:", len(chunks))
# print("\nFirst Chunk: ", chunks[0])
# print("\nSecond Chunk: ", chunks[1])

model = HuggingFaceEmbeddings(model_name="ibm-granite/granite-embedding-english-r2")

embeddings = model.embed_documents(chunks)

# print(embeddings[0][:5])

vectorstore = FAISS.from_texts(chunks, model)

query = "What is this document about?"
results = vectorstore.similarity_search(query, k=2)

# for i, res in enumerate(results):
#     print(res.page_content[:300])

load_dotenv()
api_key = os.getenv("API_KEY")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.0,
    api_key=api_key,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)


queries = ['Summarize the document', 'What are the key points?']

for query in queries:
    response = qa_chain.run(query)

    print("\n\n Query: ", query)
    print("\n\n LLM Reply: ", response)