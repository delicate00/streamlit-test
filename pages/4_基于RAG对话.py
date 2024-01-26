import time

import streamlit as st

from llamaindex_neo4j import query_KG, query_RAG, query_Vector, query_KG_and_Vector

i = 0

st.set_page_config(page_title="RAG对话", layout="wide")  # 配置页面的默认设置

st.title("基于RAG对话")   # 在应用程序中显示一个标题，设置标题的内容

if "messages4" not in st.session_state:
    st.session_state["messages4"] = []

with st.container():                    # 创建一个容器，用于组织下面的内容
    st.header("Chat with Tongyi")       # 显示一个标题

    for message in st.session_state["messages4"]:            # 遍历st.session_state["messages"]中的消息
        i = i+1
        if i % 2 == 1:
            with st.chat_message("user"):   # 在聊天界面中显示一个用户消息
                st.markdown(message)        # 在用户消息中显示message.content的内容
        else:
            with st.chat_message("ai"):     # 在聊天界面中显示一个用户消息
                st.markdown(message)        # 在用户消息中显示message.content的内容
    prompt = st.text_input(  # 在聊天界面中显示一个输入框，提示用户输入内容.
        "请在下方进行提问..."
    )
    if prompt:                                              # prompt不为空None时
        st.session_state["messages4"].append(prompt)         # 将用户输入的消息添加到st.session_state["messages"]中
        start_time = time.time()                            # 开始计时
        ai_message = query_RAG(prompt)                   # 使用chat对象对用户输入的消息进行处理，并将返回的助手消息赋值给ai_message变量
        end_time = time.time()
        load_time = end_time - start_time
        st.session_state["messages4"].append(ai_message)     # 将助手消息添加到st.session_state["messages"]列表中。
        with st.chat_message("assistant"):  # 在聊天界面中显示一个助手消息
            st.markdown(ai_message)  # 在助手消息中显示助手的回复内容
            st.markdown(f"响应时间：{load_time:.2f} 秒")
