from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
import chainlit as cl

DB_FAISS_PATH = 'vectorstore/db_faiss'

# Base prompt template for general medical knowledge
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Agent prompt templates for specialized agents
general_agent_prompt = """You are a helpful general medical AI assistant. You have access to the following tools:

{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (should be a simple string, not a list of questions)
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

diagnosis_agent_prompt = """You are a specialized medical AI assistant focused on diagnosis. You analyze symptoms and suggest possible conditions.
You have access to the following tools:

{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (should be a simple string, not a list of questions)
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

treatment_agent_prompt = """You are a specialized medical AI assistant focused on treatments and medications. You provide information about treatments, medications, and their effects.
You have access to the following tools:

{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (should be a simple string, not a list of questions)
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

# Router prompt to decide which agent to use
router_prompt_template = """You are a medical query router. Your job is to analyze the user's question and determine which specialized medical agent should handle it.
Based on the question, choose one of the following agents:

1. General Medical Agent: For general medical questions, medical terminology, anatomy, and basic health information.
2. Diagnosis Agent: For questions about symptoms, possible conditions, and diagnostic information.
3. Treatment Agent: For questions about treatments, medications, side effects, and therapeutic approaches.

User question: {question}

Think step by step about the nature of the question, then respond with ONLY the name of the most appropriate agent (exactly as written above).
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5,
        context_length = 2048  # Increased context length
    )
    return llm

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

def analyze_symptoms(symptoms):
    # More sophisticated symptom analysis
    llm = load_llm()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                      model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    medical_qa = retrieval_qa_chain(llm, set_custom_prompt(), db)
    common_conditions = medical_qa({"query": f"What conditions are associated with {symptoms}?"})["result"]
    return f"Based on the symptoms '{symptoms}', possible related conditions include: {common_conditions}\n\nPlease note this is not a diagnosis. Always consult with a healthcare professional for proper medical advice."

def analyze_treatment(treatment_query):
    # Treatment analysis
    llm = load_llm()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                      model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    medical_qa = retrieval_qa_chain(llm, set_custom_prompt(), db)
    treatment_info = medical_qa({"query": treatment_query})["result"]
    return f"Treatment information: {treatment_info}\n\nPlease consult with a healthcare professional before starting any treatment."

# Create specialized agents
def create_general_agent():
    llm = load_llm()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                      model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Create the medical knowledge tool
    medical_qa = retrieval_qa_chain(llm, set_custom_prompt(), db)
    
    tools = [
        Tool(
            name="Medical Knowledge Base",
            func=lambda q: medical_qa({"query": q})["result"],
            description="Useful for answering general medical questions, terminology, anatomy, and basic health information."
        )
    ]
    
    # Create a memory for conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create the agent prompt
    agent_prompt = PromptTemplate.from_template(general_agent_prompt)
    
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

def create_diagnosis_agent():
    llm = load_llm()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                      model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Create the medical knowledge tool
    medical_qa = retrieval_qa_chain(llm, set_custom_prompt(), db)
    
    tools = [
        Tool(
            name="Medical Knowledge Base",
            func=lambda q: medical_qa({"query": q})["result"],
            description="Useful for answering questions about medical conditions and diseases."
        ),
        Tool(
            name="Symptom Analyzer",
            func=lambda symptoms: analyze_symptoms(symptoms),
            description="Analyzes symptoms and provides information about possible related conditions. Input should be a brief description of symptoms."
        )
    ]
    
    # Create a memory for conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create the agent prompt
    agent_prompt = PromptTemplate.from_template(diagnosis_agent_prompt)
    
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

def create_treatment_agent():
    llm = load_llm()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                      model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Create the medical knowledge tool
    medical_qa = retrieval_qa_chain(llm, set_custom_prompt(), db)
    
    tools = [
        Tool(
            name="Medical Knowledge Base",
            func=lambda q: medical_qa({"query": q})["result"],
            description="Useful for answering questions about medical treatments and medications."
        ),
        Tool(
            name="Treatment Analyzer",
            func=lambda query: analyze_treatment(query),
            description="Provides information about treatments, medications, and their effects. Input should be a specific treatment or medication query."
        )
    ]
    
    # Create a memory for conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create the agent prompt
    agent_prompt = PromptTemplate.from_template(treatment_agent_prompt)
    
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

# Router function to determine which agent to use
def route_query(query):
    llm = load_llm()
    router_prompt = PromptTemplate.from_template(router_prompt_template)
    
    # Get the router's decision
    router_input = router_prompt.format(question=query)
    router_output = llm(router_input)
    
    # Clean up the output to get just the agent name
    if "General Medical Agent" in router_output:
        return "general"
    elif "Diagnosis Agent" in router_output:
        return "diagnosis"
    elif "Treatment Agent" in router_output:
        return "treatment"
    else:
        # Default to general agent if router is confused
        return "general"

# Multi-agent system
class MultiAgentSystem:
    def __init__(self):
        self.general_agent = create_general_agent()
        self.diagnosis_agent = create_diagnosis_agent()
        self.treatment_agent = create_treatment_agent()
        
    async def process_query(self, query, callbacks=None):
        # Determine which agent should handle the query
        agent_type = route_query(query)
        
        # Route to the appropriate agent
        if agent_type == "diagnosis":
            agent = self.diagnosis_agent
            prefix = "🔍 Diagnosis Agent: "
        elif agent_type == "treatment":
            agent = self.treatment_agent
            prefix = "💊 Treatment Agent: "
        else:
            agent = self.general_agent
            prefix = "📚 General Medical Agent: "
        
        # Process the query with the selected agent
        try:
            response = await agent.ainvoke(
                {"input": query},
                callbacks=callbacks
            )
            
            answer = prefix + response["output"]
        except Exception as e:
            answer = f"I apologize, but I encountered an error processing your request: {str(e)}\n\nPlease try asking a shorter or simpler question."
        
        return answer

# Chainlit code
@cl.on_chat_start
async def start():
    multi_agent_system = MultiAgentSystem()
    msg = cl.Message(content="Starting the Multi-Agent Medical AI System...")
    await msg.send()
    msg.content = "Hi, I'm your Multi-Agent Medical AI Assistant. I have specialized agents for general medical knowledge, diagnosis, and treatments. How can I help you today?"
    await msg.update()

    cl.user_session.set("multi_agent_system", multi_agent_system)

@cl.on_message
async def main(message: cl.Message):
    multi_agent_system = cl.user_session.get("multi_agent_system")
    
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["Final", "Answer"]
    )
    
    thinking_msg = cl.Message(content="Thinking... (routing your query to the most appropriate medical agent)")
    await thinking_msg.send()
    
    answer = await multi_agent_system.process_query(message.content, callbacks=[cb])
    
    await thinking_msg.remove()
    await cl.Message(content=answer).send()