from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd 
# import numpy as np
import regex as re
import json
import nltk
nltk.download('stopwords')
nltk.download('punkt')
# from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import RegexpTokenizer
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
import pickle5 as pickle 
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Sentimen Analysis",
    page_icon='https://cdn-icons-png.flaticon.com/512/1998/1998664.png',
    layout='centered',
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
st.write("""
<center><h2 style = "text-align: justify;">ANALISIS SENTIMEN PADA WISATA DIENG DENGAN ALGORITMA K-NEAREST NEIGHBOR (K-NN)</h2></center>
""",unsafe_allow_html=True)
st.write("### Dosen Pengampu : Dr. FIKA HASTARITA RACHMAN, ST., M.Eng",unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
        st.write("""<h3 style = "text-align: center;"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTyxC04xzaLSQorRMjT-4XJIjITb9sACMhbEA&usqp=CAU" width="120" height="120"></h3>""",unsafe_allow_html=True), 
        ["Implementation"], 
            icons=['house', 'bar-chart','check2-square', 'person'], menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#412a7a"},
                "icon": {"color": "white", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color":"white"},
                "nav-link-selected":{"background-color": "#412a7a"}
            }
        )



    if selected == "Implementation":
        #Getting input from user
        iu = st.text_area('Masukkan kata yang akan di analisa :')

        submit = st.button("submit")

        if submit:
            with open('tfidf_data.pkl', 'rb') as file:
                loaded_data_tfid = pickle.load(file)
            with open('dt_model.pkl', 'rb') as file:
                loaded_model = pickle.load(file)
            def prepodatainput(data_uji):
                ulasan_case_folding = data_uji.lower()

                #Cleansing
                clean_tag  = re.sub("@[A-Za-z0-9_]+","", ulasan_case_folding)
                clean_hashtag = re.sub("#[A-Za-z0-9_]+","", clean_tag)
                clean_https = re.sub(r'http\S+', '', clean_hashtag)
                clean_symbols = re.sub("[^a-zA-Z ]+"," ", clean_https)

                #Inisialisai fungsi tokenisasi dan stopword
                #Tokenizer
                tokenizer = RegexpTokenizer(r'[a-z]+')
                tokens = tokenizer.tokenize(clean_symbols)

                #Stop Words
                stop_factory = StopWordRemoverFactory()
                more_stopword = ["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 'kalo', 'amp', 'biar', 'bikin', 'bilang',
                                                'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 'si', 'tau', 'tdk', 'tuh', 'utk', 'ya',
                                                'jd', 'jgn', 'sdh', 'aja', 'n', 't', 'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                                                '&amp', 'yah']
                data = stop_factory.get_stop_words()+more_stopword
                removed = []
                if tokens not in data:
                    removed.append(tokens)

                                #list to string
                gabung =' '.join([str(elem) for elem in removed])

                                #Steaming
                factory = StemmerFactory()
                stemmer = factory.create_stemmer()
                stem = stemmer.stem(gabung)
                return(ulasan_case_folding,clean_symbols,tokens,gabung,stem)

                ulasan_case_folding,clean_symbols,tokens,gabung,stem = prepodatainput(data_uji)
                data_akhir = loaded_data_tfid.transform([stem]).toarray()
                y_preds = loaded_model.predict(data_akhir)
            
                st.subheader('Prediksi')
                if y_preds == "positive":
                    st.success('Positive')
                else:
                    st.error('Negative')

        
