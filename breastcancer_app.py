import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

from keras.models import load_model


import warnings
warnings.filterwarnings("ignore")



st.markdown("<h2 style='text-align: center;'>Breast Cancer Classification</h2>", unsafe_allow_html=True)
st.markdown('---'*10)

model_final = joblib.load('breastcancer_pipeline.pkl')
model_final.named_steps['modeling'].model = load_model('model_bc_keras.h5')

pilihan = st.selectbox('Apa yang ingin Anda lakukan?',['Prediksi dari file excel','Input Manual'])

if pilihan == 'Prediksi dari file excel':
    def set_bg_1(main_bg):

        main_bg_ext = "png"
            
        st.markdown(
             f"""
             <style>
             .stApp {{
                 background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
                 background-position: center;
                 background-size: 720px 520px;
                 background-repeat: no-repeat
             }}
             </style>
             """,
             unsafe_allow_html=True
         )
    set_bg_1('bg_breastcancer.png')
    
    # Mengupload file
    upload_file = st.file_uploader('Pilih file excel', type='xlsx')
    if upload_file is not None:
        dataku1 = pd.read_excel(upload_file)
        dataku1.rename(columns={"concave points_mean":"concave_points_mean","concave points_se":"concave_points_se","concave points_worst":"concave_points_worst"}, inplace=True)
        dataku = dataku1.copy()
        dataku.drop(['id'],axis=1,inplace=True)
        dataku = pd.DataFrame(columns=['radius_mean',
                 'texture_mean',
                 'smoothness_mean',
                 'compactness_mean',
                 'concavity_mean',
                 'symmetry_mean',
                 'fractal_dimension_mean',
                 'radius_se',
                 'texture_se',
                 'smoothness_se',
                 'compactness_se',
                 'concavity_se',
                 'concave_points_se',
                 'symmetry_se',
                 'fractal_dimension_se',
                 'smoothness_worst',
                 'compactness_worst',
                 'concavity_worst',
                 'symmetry_worst',
                 'fractal_dimension_worst'], data=dataku)
        st.write(dataku1)
        st.success('File berhasil diupload')
        if st.button('Show Prediction'):
            hasil = model_final.predict(dataku)
            #st.write('Prediksi',hasil)
            # Keputusan
            for i in range(len(hasil)):
                if hasil[i] == 1:
                    st.write('Klasifikasi Breast Cancer : ID',dataku1['id'][i],' diprediksi Malignant')
                else:
                    st.write('Klasifikasi Breast Cancer : ID',dataku1['id'][i],' diprediksi Benign')
    else:
        st.error('File yang diupload kosong, silakan pilih file yang valid')
        #st.markdown('File yang diupload kosong, silakan pilih file yang valid')
else:
    def set_bg_2(main_bg):

        main_bg_ext = "png"
            
        st.markdown(
             f"""
             <style>
             .stApp {{
                 background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
                 background-position: center;
                 background-color: white;
                 background-size: 90px 70px;
                 background-repeat: repeat;

             }}
             </style>
             """,
             unsafe_allow_html=True
         )
    set_bg_2('Floral_Breast_Cancer_Ribbon-fixed.png')

    #1
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            radius_mean = st.number_input('radius_mean', value=17)
        with col2:
            texture_mean = st.number_input('texture_mean', value=18)
    
    
    #2
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            smoothness_mean = st.number_input('smoothness_mean', value=0.15)
        with col2:
            compactness_mean = st.number_input('compactness_mean', value=0.30)
    
    
    #3
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            concavity_mean = st.number_input('concavity_mean', value=0.1)
        with col2:
            symmetry_mean = st.number_input('symmetry_mean', value=0.2)
        
    
    #4
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            fractal_dimension_mean = st.number_input('fractal_dimension_mean', value=0.06)
        with col2:
            radius_se = st.number_input('radius_se', value=0.5)

    
    
    #5
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            texture_se = st.number_input('texture_se', value=0.8)
        with col2:
            smoothness_se = st.number_input('smoothness_se', value=0.008)
        
    
    #6
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            compactness_se = st.number_input('compactness_se', value=0.05)
        with col2:
            concavity_se = st.number_input('concavity_se', value=0.07)
        
    
    #7
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            concave_points_se = st.number_input('concave_points_se', value=0.02)
        with col2:
            symmetry_se = st.number_input('symmetry_se', value=0.07)
        
    
    #8
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            fractal_dimension_se = st.number_input('fractal_dimension_se', value=0.01)
        with col2:
            smoothness_worst = st.number_input('smoothness_worst', value=0.11)
        
    
    #9
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            compactness_worst = st.number_input('compactness_worst', value=0.9)
        with col2:
            concavity_worst = st.number_input('concavity_worst', value=0.8)
    
    #10
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            symmetry_worst = st.number_input('symmetry_worst', value=0.8)
        with col2:
            fractal_dimension_worst = st.number_input('fractal_dimension_worst', value=0.2)
        
    # Inference
    data = {
            'radius_mean': radius_mean,
            'texture_mean': texture_mean,
            'smoothness_mean': smoothness_mean,
            'compactness_mean': compactness_mean,
            'concavity_mean': concavity_mean,
            'symmetry_mean': symmetry_mean,
            'fractal_dimension_mean': fractal_dimension_mean,
            'radius_se': radius_se,
            'texture_se': texture_se,
            'smoothness_se': smoothness_se,
            'compactness_se': compactness_se,
            'concavity_se': concavity_se,
            'concave_points_se': concave_points_se,
            'symmetry_se': symmetry_se,
            'fractal_dimension_se': fractal_dimension_se,
            'smoothness_worst': smoothness_worst,
            'compactness_worst': compactness_worst,
            'concavity_worst': concavity_worst,
            'symmetry_worst': symmetry_worst,
            'fractal_dimension_worst': fractal_dimension_worst
            }
    
    
    # Tabel data
    kolom = list(data.keys())
    df = pd.DataFrame([data.values()], columns=kolom)
    
    mystyle = '''
    <style>
        p {
            text-align: center;
            font-weight: bold;
            font-weight: 100px
        }
    </style>
    '''
    # Memunculkan hasil di Web 
    st.write('***'*10)
    if st.button('Show Prediction'):
        prediksi = model_final.predict(df)

        if (prediksi[0] == 1):
            st.write(mystyle,'Malignant',unsafe_allow_html=True)
        else:
            st.write(mystyle,'Benign',unsafe_allow_html=True)
    