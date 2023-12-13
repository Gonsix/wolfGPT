import os
import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

llm = ChatOpenAI(
        temperature=0.1, 
        openai_api_key=OPENAI_API_KEY,
        model="gpt-4-1106-preview",
        streaming=True )
agent_kwargs = {
    "extra_prompt_message" : [MessagesPlaceholder(variable_name="memory")],
}
memory = ConversationBufferMemory(memory_key="memory", return_messages=True)


tools = load_tools(["ddg-search"])
agent = initialize_agent(
    tools, 
    llm,
    agent=AgentType.OPENAI_FUNCTIONS, 
    agent_kwargs=agent_kwargs,
    memory=memory

)   

with st.sidebar:
    "💬 Chatbot with search engine"

st.title("🦜🔗 Langchain Chatbot")
st.caption("🚀 A streamlit chatbot powered by OpenAI GPT-4 turbo")

#Initialize agent
if "agent" not in st.session_state:
    st.session_state.agent = agent

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# React to user input
if prompt := st.chat_input("What is up?"):
    if not OPENAI_API_KEY:
        st.info("Missed OpenAI API key.")
        st.stop()

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # ここにst_cb があれば、ストリーミングで出力される。＊めっちゃええ感じに
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = st.session_state.agent.run(prompt, callbacks=[st_cb])
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
