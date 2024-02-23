
import streamlit as st


def handleMissingData2008(data):
    st.header('TIỀN XỬ LÝ DỮ LIỆU',divider = 'rainbow')
    st.header('DATA 2008')
    # Drop rows with missing data
    cleaned_data = data.dropna()
    # Get the number of rows and columns in the cleaned dataset
    num_rows_cleaned, num_cols_cleaned = cleaned_data.shape
    st.subheader('1. Process Missing Data')
    st.write('Số lượng dòng Data 2008 sau khi xóa: ', num_rows_cleaned)
    st.write('Số lượng cột Data 2008: ', num_cols_cleaned)

def handleMissingData2009(data):
    # Drop rows with missing data
    cleaned_data = data.dropna()
    st.header('DATA 2009')
    # Get the number of rows and columns in the cleaned dataset
    st.subheader('1. Process Missing Data')
    num_rows_cleaned, num_cols_cleaned = cleaned_data.shape
    st.write('Số lượng dòng Data 2009 sau khi xóa: ', num_rows_cleaned)
    st.write('Số lượng cột: ', num_cols_cleaned)


def handleDuplicateData2008(data):


    # Check for duplicate rows
    st.subheader('2. Process Duplicate Data')
    duplicates = data.duplicated()
    st.write("Số lượng cột trùng năm 2008:", duplicates.sum())

    # Remove duplicate rows
    cleaned_data = data.drop_duplicates()

    # Optionally, you can include the 'inplace=True' parameter to drop duplicates directly in the original DataFrame
    # data.drop_duplicates(inplace=True)

    # Check the shape of the data after removing duplicates
    st.write("Sau khi xóa cột trùng năm 2008:", cleaned_data.shape)

def handleDuplicateData2009(data):

    # Check for duplicate rows
    st.subheader('2. Process Duplicate Data')
    duplicates = data.duplicated()
    st.write("Số lượng cột trùng năm 2008:", duplicates.sum())

    # Remove duplicate rows
    cleaned_data = data.drop_duplicates()

    # Optionally, you can include the 'inplace=True' parameter to drop duplicates directly in the original DataFrame
    # data.drop_duplicates(inplace=True)

    # Check the shape of the data after removing duplicates
    st.write("Sau khi xóa cột trùng năm 2008:", cleaned_data.shape)

def checkDataType2008(data):
    st.subheader('3. Check Data Type')
    st.write('Data 2008',data.dtypes)

def checkDataType2009(data):
    st.subheader('3. Check Data Type')
    st.write('Data 2009',data.dtypes)

