from langchain_google_vertexai import VertexAI
from langchain_community.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
llm = VertexAI(max_output_tokens=1000, temperature=0.9)

# # Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"

def create_vector_db():
    # Load data from FAQ sheet
    loader = CSVLoader(file_path='qanda.csv', source_column="question")
    data = loader.load()
    # Create a FAISS instance for vector database from 'data'
    vectordb = FAISS.from_documents(documents=data,
                                    embedding=instructor_embeddings)
    # Save vector database locally
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)
 
   # Creating a Prompt/Context to influence the chatbox behavior
 
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes. You can add or remove words in your answers according to the questions.
    If the answer is not found in the context, kindly state "I'm sorry, I don't have the information you're looking for at the moment. For further assistance, please reach out to us at info-emildai@dcu.ie" Don't try to make up an answer.
    
    

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain


if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()

    print(chain("How are you?"))

    

'''
from InstructorEmbedding import INSTRUCTOR
model = INSTRUCTOR('hkunlp/instructor-large')
sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments"
instruction = "Represent the Science title:"
embeddings = model.encode([[instruction,sentence]])
print(embeddings)
'''

#conda create -p vertexaione python=3.11 (powershell)
#(cmd) conda activate C:\VL\Racoon\01mar24\vertexaione
#gcloud auth application-default login
#pip install langchain
#pip install google-cloud-aiplatform
#pip install -r requirements.txt
#pip install InstructorEmbedding
#from langchain.llms import VertexAI
#from langchain_community.llms import VertexAI

#from langchain_google_vertexai import ChatVertexAI
#llm = VertexAI(model_name="code-bison", max_output_tokens=1000, temperature=0.3)
#llm = ChatVertexAI(model_name="chat-bison-32k")
#*print(llm("Write a poem about Law and AI"))

#27 may
#PS C:\VL\Racoon\01mar24> pip list | findstr google-cloud-aiplatform
#PS C:\VL\Racoon\01mar24> pip install google-cloud-aiplatform==1.52.0
#PS C:\VL\Racoon\01mar24> pip install google-cloud-aiplatform --user

