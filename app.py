import pickle
import numpy as np
import pandas as pd
import streamlit as st
from recommender import data, similarity, recommend  # reuse from your file

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommender (Content-Based)")

movie_name = st.text_input("Enter a movie you like:", value="Avatar")

if st.button("Recommend"):
    recs = recommend(movie_name, top_n=5)
    if not recs:
        st.info("I couldn't find that title. Try another.")
    else:
        st.subheader("Top suggestions")
        for i, r in enumerate(recs, 1):
            st.write(f"{i}. {r}")
