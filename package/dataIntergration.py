
import streamlit as st
# def mergeData(data1, data2):
# # Read the Excel files
# # Merge the DataFrames
#     data = pd.concat([data1, data2], ignore_index=True)
#     st.header('Tích hợp dữ liệu',divider = 'rainbow')
#     # Save the merged DataFrame to a new Excel file
#     data.to_excel('Contoso 2008-2009 Data.xlsx', index=False)
def dI_preProcessing(data):
    st.header('TÍCH HỢP DỮ LIỆU',divider='rainbow')
    num_rows, num_cols = data.shape
    st.write('Số lượng dòng Data sau khi gộp bảng: ',num_rows)
    st.write('Số lượng cột Data sau khi gộp bảng: ',num_cols)

    
