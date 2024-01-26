import streamlit as st

import subprocess


# def install_dependencies():
#     subprocess.check_call("pip install -r requirements.txt".split())
#
#
# install_dependencies()


st.set_page_config(page_title="欢迎访问智能小帮手", layout="centered")  # 配置页面的默认设置

st.title("👋🏼 欢迎访问智能小帮手")   # 在应用程序中显示一个标题，设置标题的内容

st.subheader("功能：可实现外接知识库的智能问答")

st.write("1. 基于原生向量的对话")
st.write("2. 基于知识图谱的对话")
st.write("3. 基于自定义混合的对话")
st.write("4. 基于RAG的对话")

