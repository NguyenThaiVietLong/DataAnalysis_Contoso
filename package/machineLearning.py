
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression
pd.options.display.float_format = '{:.2f}'.format
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt 
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import random

def heatMap(data):
    corr = data.corr()

    # Tạo heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={"shrink": .82})
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(plt)

def dataReduction(data):
    data = data.drop('SalesKey', axis = 1)
    data = data.drop('DateKey', axis = 1)
    data = data.drop('channelKey', axis = 1)
    data = data.drop('StoreKey', axis = 1)
    data = data.drop('ProductKey', axis = 1)
    data = data.drop('PromotionKey', axis = 1)
    data = data.drop('GeographyKey', axis = 1)
    data = data.drop('Date', axis = 1)
    return data
    


def machineLearning(data):
    # Split the data
    x = data.drop('SalesAmount', axis=1)
    y = data['SalesAmount']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Define the models
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(),
        "GradientBoostingRegressor": GradientBoostingRegressor(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "DecisionTreeRegressor": DecisionTreeRegressor(),
        "KNeighborsRegressor": KNeighborsRegressor(),
    }

    # Function to fit and score models
    model_scores = {}
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        y_preds = model.predict(x_test)  # Make predictions on the test data
        score = r2_score(y_test, y_preds)
        model_scores[model_name] = score

    # Set a threshold for classification (this should be determined by your specific use case)
    threshold = y_test.median()

    # Convert actual and predicted values to binary classes
    y_test_binary = (y_test > threshold).astype(int)
    y_preds_binary = {model_name: (model.predict(x_test) > threshold).astype(int) for model_name, model in models.items()}

    # Calculate metrics for each model
    model_f1_scores = {model_name: f1_score(y_test_binary, y_preds) for model_name, y_preds in y_preds_binary.items()}
    model_recall_scores = {model_name: recall_score(y_test_binary, y_preds) for model_name, y_preds in y_preds_binary.items()}
    model_precision_scores = {model_name: precision_score(y_test_binary, y_preds) for model_name, y_preds in y_preds_binary.items()}

    # Combine all metrics into a single DataFrame
    new_df = pd.DataFrame({
        'Accuracy Scores': model_scores,
        'F1 Scores': model_f1_scores,
        'Recall Scores': model_recall_scores,
        'Precision Scores': model_precision_scores
    })

    # Transpose the DataFrame to have models as rows and metrics as columns
    new_df = new_df.T


    # Display the DataFrame as a table in Streamlit
    st.table(new_df)

def predictValue(data, model_gr, model_lr):
    x = data.drop('SalesAmount',axis = 1)
    y = data['SalesAmount']
    index = st.number_input('Nhập index vào đây',value=None,placeholder='Nhập vị trí muốn dự đoán vào đây')
    random_index = st.button('Chọn 1 dòng random')

    if random_index:
        index = random.randint(0, 480080)
        st.write(x.iloc[index])
        st.write(f"Giá trị `SalesAmount` gốc: {y.iloc[index]}")
        # inputs = np.array([48.92,95.95,9,0,0,1,0,1876.68]).reshape(1, -1)
        inputs = x.iloc[index].to_numpy().reshape(1, -1)
        outputs_lr = model_lr.predict(inputs)
        outputs_rf = model_gr.predict(inputs)

        st.write(f"Giá trị Linear Regression dự đoán: {outputs_lr[0]}")
        st.write(f"Giá trị Random Forest dự đoán: {outputs_rf[0]}")
        st.write(f"sai số Linear Regression: {(y.iloc[index] - outputs_lr)[0]}")
        st.write(f"sai số Random Forest: {(y.iloc[index] - outputs_rf)[0]}")
    elif index is not None:
        index = int(index)
        
        st.write(x.iloc[index])
        st.write(f"Giá trị `SalesAmount` gốc: {y.iloc[index]}")
        # inputs = np.array([48.92,95.95,9,0,0,1,0,1876.68]).reshape(1, -1)
        inputs = x.iloc[index].to_numpy().reshape(1, -1)
        outputs_lr = model_lr.predict(inputs)
        outputs_rf = model_gr.predict(inputs)

        st.write(f"Giá trị Linear Regression dự đoán: {outputs_lr[0]}")
        st.write(f"Giá trị Random Forest dự đoán: {outputs_rf[0]}")
        st.write(f"Sai số Linear Regression: {(y.iloc[index] - outputs_lr)[0]}")
        st.write(f"Sai số Random Forest: {(y.iloc[index] - outputs_rf)[0]}")

    # st.write(x.iloc[index])
    # st.write(x.iloc[index])
# index = random.randint(0, 480080)
# print(x.iloc[index])
# print(y.iloc[index])
# # inputs = np.array([48.92,95.95,9,0,0,1,0,1876.68]).reshape(1, -1)
# inputs = x.iloc[index].to_numpy().reshape(1, -1)
# outputs = model.predict(inputs)
# print(outputs)
# print(f"sai số {(y.iloc[index] - outputs)[0]}")
def predictValue2(model_rf):
    a1 = st.number_input("Nhập `UnitCost`")
    a2 = st.number_input("Nhập `UnitPrice`")
    a3 = st.number_input("Nhập `SalesQuantity`")
    a4 = st.number_input("Nhập `ReturnQuantity`")
    a5 = st.number_input("Nhập `ReturnAmount`")
    a6 = st.number_input("Nhập `DiscountQuantity`")
    a7 = st.number_input("Nhập `DiscountAmount`")
    a8 = st.number_input("Nhập `TotalCost`")
    inputs = [a1, a2, a3, a4, a5, a6, a7, a8]
    inputs = np.array(inputs).reshape(1, -1)
    outputs = model_rf.predict(inputs)[0]

    button = st.button('Dự đoán')
    if button:
        st.write(outputs)



