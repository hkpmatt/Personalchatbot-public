import os
#from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader, TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import LLMChain
import streamlit as st
#load_dotenv()

def sim():
    loader = CSVLoader(file_path="data.csv", encoding='utf-8-sig', csv_args={
        'delimiter' : ';',
        'fieldnames' : ['Quesiton','Response']
    })
    #loader = TextLoader(file_path="data.txt", encoding='utf-8-sig')
    documents = loader.load()
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings)
    return db

def retrieve_info(query):
    db = sim()
    similar_responses = db.similarity_search_with_score(query, k=3)
    response = [doc.page_content for doc, score in similar_responses]
    score = [score for doc, score in similar_responses]
    return response, score
def LLM():
    
    llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.8)

    template = """
### Context
You are my personal assistant that know my career and academic background. 
User will ask question about my resume and they might also ask question irrelevant to my resume. 
I will provide the question, 3 output of similar response with its similarity score of that question.

Here are 4 rules that you MUST follow:
### Rules
1. Answer the question as the point of view as me.
2. Please answer the question based on the output provided below with your rephraser and elabrolate more, the lower score the better.
3. When the average score of the outputs are higher than 4.8, the question is mostly likely irrelevant to my resume, answer it with your own word.
4. Answer the question directly and don't mention about the provided output. 

Below is the question asked by user:
{message}

Here are total of 3 outputs for the question from database:
{answer}

"""

    prompt = PromptTemplate(
        input_variables=["message","answer"],
        template=template
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain

def gen_reponse(message):
    answers, scores = retrieve_info(message)
    #if scores[0]>0.5:
    #    total_answer = 'This question is irrelevant to the resume'
    #for index, score in enumerate(scores):
    #    if score > 0.48:
    #        answers[index]= 'This question is irrelevant to the resume'
    total_answer = ''
    for index,answer in enumerate(answers):
        total_answer+= f"score{index+1}:{scores[index]} output{index+1} : {answer}\n"
    print(total_answer)
    chain = LLM()
    repsonse = chain.run(message = message, answer=total_answer)
    return repsonse



os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
st.set_page_config(
    page_title="Personal resume assistant", page_icon=":robot_face:")
st.title("Resume chat bot :robot_face:")
st.warning("This is my personal resume chatbot using openai api.\nPlease do not ask more than 3 question in a minute.",icon="🎈")
st.info("Contact Me: [linkedIn](https://www.linkedin.com/in/pak-hei-siu-a881041a3) or matttt12341@gmail.com")
st.info("Report Bug: matttt2001@gmail.com")
#form =  st.form(key="my_form")
#message = form.text_input("Please ask your question below (e.g. What is your academic background?, Talk about your work experience ): ")
#submit_button = form.form_submit_button(label='Submit')
#   message = st.text_area("Please ask your question below (e.g. What is your academic background?, Talk about your work experience ): ")
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt := st.chat_input("Please ask you question here"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = gen_reponse(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    #if submit_button:
    #    st.write("Generating answer...")
    #    result = gen_reponse(message)
    #    with st.chat_message('User'):
    #        st.write(message)
#
    #    if result:
    #        with st.chat_message('Bot'):
    #            st.write(result)
        #st.text_area("Responses",value = result, height=200)

