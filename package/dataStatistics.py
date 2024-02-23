
import streamlit as st

 

def dataStatistics2008(data):

    
    #Hiển thị 5 dòng đầu tiên của tập dữ liệu
    st.subheader('1. Five first Row of Data')
    st.write(data.head())

    # #Số lượng dòng và cột trong tập dữ liệu
    num_rows, num_cols = data.shape
    st.subheader('2. Number Columns and Rows')
    st.write('Số lượng dòng Data 2008: ',num_rows)
    st.write('Số lượng cột Data 2008: ',num_cols)

    # Số lượng giá trị duy nhất trong từng cột
    st.subheader('3. Number Unique Value in Columns')
    st.write('Số lượng giá trị duy nhất trong từng cột 2008:')
    st.write(data.nunique())

    #Tóm tắt dữ liệu
    st.subheader('4. Summary Data')
    st.write(data.describe())

def dataStatistics2009(data):

    #Hiển thị 5 dòng đầu tiên của tập dữ liệu
    st.subheader('1. Five first Row of Data')
    st.write(data.head())

    # #Số lượng dòng và cột trong tập dữ liệu
    num_rows, num_cols = data.shape
    st.subheader('2. Number Columns and Rows')
    st.write('Số lượng dòng Data 2009: ',num_rows)
    st.write('Số lượng cột Data 2009: ',num_cols)

    # Số lượng giá trị duy nhất trong từng cột
    st.subheader('3. Number Unique Value in Columns')
    st.write('Số lượng giá trị duy nhất trong từng cột 2009:')
    st.write(data.nunique())

    #Tóm tắt dữ liệu
    st.subheader('4. Summary Data')
    st.write(data.describe())


