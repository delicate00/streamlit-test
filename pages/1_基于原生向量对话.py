import streamlit as st
from llamaindex_neo4j import query_KG, query_RAG, query_Vector, query_KG_and_Vector
import time

i = 0
load_time = 0

st.set_page_config(page_title="原生向量对话", layout="wide")  # page_title为页面名称；layout是页面布局：centered、wide

st.title("基于原生向量对话")                                # 在应用程序中显示一个标题，设置标题的内容


if "messages" not in st.session_state:
    st.session_state["messages"] = []

# with语句的作用是在进入代码块时创建容器，并在离开代码块时自动关闭容器。
# 这样可以确保在容器内的内容被正确地组织和显示，并且在不需要容器时可以自动释放资源。
with st.container():                    # 创建一个容器，用于组织下面的内容
    st.header("Chat with Tongyi")       # 显示一个标题

    for message in st.session_state["messages"]:            # 遍历st.session_state["messages"]中的消息
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
        st.session_state["messages"].append(prompt)         # 将用户输入的消息添加到st.session_state["messages"]中
        start_time = time.time()                            # 开始计时
        ai_message = query_Vector(prompt)                   # 使用chat对象对用户输入的消息进行处理，并将返回的助手消息赋值给ai_message变量
        end_time = time.time()
        load_time = end_time - start_time
        st.session_state["messages"].append(ai_message)     # 将助手消息添加到st.session_state["messages"]列表中。
        with st.chat_message("assistant"):  # 在聊天界面中显示一个助手消息
            st.markdown(ai_message)  # 在助手消息中显示助手的回复内容
            st.markdown(f"响应时间：{load_time:.2f} 秒")
