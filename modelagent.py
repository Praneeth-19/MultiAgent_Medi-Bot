from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
# Update the imports for HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.agents import Tool, AgentExecutor, create_react_agent
# Update the import for ChatMessageHistory
from langchain.memory import ConversationBufferMemory
import chainlit as cl

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

agent_prompt_template = """You are a helpful medical AI assistant. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Previous conversation history:
{chat_history}

Question: {input}
{agent_scratchpad}
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# Create an agent with tools
def create_medical_agent():
    llm = load_llm()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                      model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Create the medical knowledge tool
    medical_qa = retrieval_qa_chain(llm, set_custom_prompt(), db)
    
    tools = [
        Tool(
            name="Medical Knowledge Base",
            func=lambda q: medical_qa.invoke({"query": q})["result"],
            description="Useful for answering questions about medical topics, diseases, treatments, and medications."
        ),
        Tool(
            name="Symptom Analyzer",
            func=lambda symptoms: analyze_symptoms(symptoms),
            description="Analyzes symptoms and provides general information. Input should be a brief description of symptoms."
        )
    ]
    
    # Create a memory for conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create the agent prompt
    agent_prompt = PromptTemplate.from_template(agent_prompt_template)
    
    # Create the agent
    agent = create_react_agent(llm, tools, agent_prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

#chainlit code
@cl.on_chat_start
async def start():
    agent = create_medical_agent()
    msg = cl.Message(content="Starting the Medical AI Agent...")
    await msg.send()
    msg.content = "Hi, I'm your Medical AI Assistant. I can answer medical questions, analyze symptoms, and provide general health information. How can I help you today?"
    await msg.update()

    cl.user_session.set("agent", agent)

@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    
    cb = cl.AsyncLangchainCallbackHandler(
        # Remove the 'verbose' parameter if it exists
        # Keep other necessary parameters
    )
    
    try:
        response = await agent.ainvoke(
            {"input": message.content},
            callbacks=[cb]
        )
        
        answer = response["output"]
    except Exception as e:
        answer = f"I apologize, but I encountered an error processing your request: {str(e)}\n\nPlease try asking a shorter or simpler question."
    
    await cl.Message(content=answer).send()


def analyze_symptoms(symptoms):
    # More sophisticated symptom analysis
    common_conditions = medical_qa({"query": f"What conditions are associated with {symptoms}?"})["result"]
    return f"Based on the symptoms '{symptoms}', possible related conditions include: {common_conditions}\n\nPlease note this is not a diagnosis. Always consult with a healthcare professional for proper medical advice."
