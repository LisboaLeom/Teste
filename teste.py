import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# ==========================================
# CARREGAR VARIÁVEIS DE AMBIENTE
# ==========================================
load_dotenv()

# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(
    page_title="SimulAI - Entrevistador Técnico",
    page_icon="🤖",
    layout="centered"
)


# ==========================================
# 1. INICIALIZAÇÃO DA IA E MEMÓRIA (OpenAI)
# ==========================================
@st.cache_resource
def init_llm_and_memory():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=500
    )

    memory = ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=1000,
        return_messages=True
    )

    return llm, memory


llm, memory = init_llm_and_memory()


# ==========================================
# 2. DEFINIÇÃO DE PROMPTS
# ==========================================
def get_interview_prompt(cargo):
    system_template = f"""
    Você é um Engenheiro de Software Sênior e Recrutador conduzindo uma entrevista técnica para a vaga de '{cargo}'.

    Regras estritas:
    1. Faça APENAS UMA pergunta por vez.
    2. Aguarde a resposta do candidato.
    3. Avalie criticamente a resposta.
    4. Se estiver superficial, aprofunde com perguntas.
    5. Seja profissional, educado e exigente.
    6. Comece com uma saudação e a primeira pergunta técnica.
    """

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])


def get_feedback_prompt():
    system_template = """
    Você agora é um Mentor de Carreira.

    Analise o histórico da entrevista e gere:

    - Pontos Fortes
    - Pontos a Melhorar
    - Nota de 0 a 10
    - Recomendações de estudo
    - Caso apenas tenho recebido menos de 3 respostas, diga que não foi possivel validar o canditado e ele foi desclassificado.

    Use Markdown com bullet points.
    Seja construtivo e claro.
    """

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])


# ==========================================
# 3. INTERFACE
# ==========================================
st.title("🤖 SimulAI: Simulador de Entrevistas Técnicas")

# Estados
if "fase" not in st.session_state:
    st.session_state.fase = "setup"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation" not in st.session_state:
    st.session_state.conversation = None

# ==========================================
# FASE 1: SETUP
# ==========================================
if st.session_state.fase == "setup":
    st.markdown("### Prepare-se para sua entrevista")

    cargo = st.text_input("Qual vaga você quer simular?")

    if st.button("Iniciar Entrevista") and cargo:
        st.session_state.conversation = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=get_interview_prompt(cargo),
            verbose=False
        )

        st.session_state.fase = "entrevista"
        st.session_state.messages = [
            {"role": "system", "content": f"Simulação iniciada para: {cargo}"}
        ]

        st.rerun()

# ==========================================
# FASE 2: ENTREVISTA
# ==========================================
elif st.session_state.fase == "entrevista":

    # Histórico
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Primeira pergunta da IA
    if len(st.session_state.messages) == 1:
        with st.chat_message("assistant"):
            with st.spinner("Gerando pergunta..."):
                resposta = st.session_state.conversation.predict(
                    input="Olá, estou pronto para a entrevista."
                )

                st.markdown(resposta)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": resposta
                })

    # Input do usuário
    user_input = st.chat_input("Digite sua resposta ou 'Finalizar'")

    if user_input:

        if user_input.lower().strip() == "finalizar":
            st.session_state.fase = "feedback"
            st.rerun()

        # Mensagem usuário
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        # Resposta IA
        with st.chat_message("assistant"):
            with st.spinner("Analisando resposta..."):
                resposta = st.session_state.conversation.predict(
                    input=user_input
                )

                st.markdown(resposta)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": resposta
                })

# ==========================================
# FASE 3: FEEDBACK
# ==========================================
elif st.session_state.fase == "feedback":

    st.markdown("### 📊 Relatório de Desempenho")

    with st.spinner("Gerando feedback..."):
        feedback_chain = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=get_feedback_prompt()
        )

        relatorio = feedback_chain.predict(
            input="Gere o feedback completo da entrevista."
        )

        st.markdown(relatorio)

    if st.button("Reiniciar"):
        st.session_state.clear()
        st.rerun()