import streamlit as st

sample_page = st.Page("samples_zh.py", title="样本可视化", icon="🖼️")
sample_page_en = st.Page("samples_en.py", title="VLM Eval Visualization", icon="🖼️")
pg = st.navigation([sample_page, sample_page_en])

pg.run()