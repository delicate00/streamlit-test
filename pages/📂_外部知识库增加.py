import pandas as pd
import streamlit as st

from llamaindex_neo4j import load_new_documents1

# 11. st.file_uploader() 上传文件
st.title("📂 外部知识库上传")
upload_file = st.file_uploader(
    label="请上传新文件"
)

if upload_file is not None:
    # 不为空
    filename = upload_file.name
    filename = f"doc/{filename}"
    df = load_new_documents1(filename)
    st.success("上传文件成功！")
else:
    st.stop()   # 退出

