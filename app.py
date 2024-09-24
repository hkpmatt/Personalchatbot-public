import os
import torch
#from dotenv import load_dotenv,find_dotenv
from langchain.llms import HuggingFacePipeline, HuggingFaceHub
from langchain import PromptTemplate, LLMChain, GoogleSerperAPIWrapper
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline, Conversation ,AutoModel, AutoModelForSeq2SeqLM, AutoConfig, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from datasets import load_dataset
#from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
#from langchain.agents import initialize_agent, Tool, AgentType
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
import streamlit as st
#from sentence_transformers import SentenceTransformer

#load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

loader = CSVLoader(file_path="data.csv", encoding='utf-8-sig')
documents = loader.load()

embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(documents, embeddings)

def retrieve_info(query):

    similar_response = db.similarity_search(query, k=2)

    page_contents_array = [doc.page_content for doc in similar_response]

    print(page_contents_array)
    return page_contents_array


#apillm = HuggingFaceHub(repo_id="google/flan-t5-large",model_kwargs={"temperature": 0, "max_length": 400})

#model_id = "garage-bAInd/Platypus2-70B-instruct"
#model_id = "google/flan-t5-large"
model_id = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id) 
pipe = pipeline(
    model=model,
    tokenizer=tokenizer,
    pad_token_id=50256,
    max_length=700
)
local_llm = HuggingFacePipeline(pipeline=pipe)

template =""" 
You are my personal assitant that suppose to know my career and academic background. You will be asked question about me.
Response should be vry similar or even identical to the data.

Below is the question asked about me:
{message}

Here is the similar response of the question from dataset:
{answer}

Answer the question with the first response from the similar response:
"""

prompt = PromptTemplate(
    input_variables=['message','answer'],
    template= template
)
lmchain = LLMChain(prompt=prompt,llm=local_llm
                   #llm=apillm
                   )

def gen_reponse(message):
    answer = retrieve_info(message)
    repsonse = lmchain.run(message = message, answer=answer)
    print(repsonse)
    return repsonse


def main():
    st.set_page_config(
        page_title="Customer response generator", page_icon=":bird:"
    )
    st.header("Resume chat bot :bird:")
    message = st.text_area("Please ask your question below (e.g. What is your academic background?, Talk about your work experience ): ")

    if message:
        st.write("Generating answer...")
        result = gen_reponse(message)
        st.text_area(result)

if __name__== '__main__':
    main()
