import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st

def revenueOverTime(data1, data2):
# Convert Date to datetime and extract year and quarter
    data1['Year'] = data1['Date'].dt.year
    data2['Year'] = data2['Date'].dt.year

    data1['Quarter'] = data1['Date'].dt.to_period('Q')
    data2['Quarter'] = data2['Date'].dt.to_period('Q')

    # Calculate Profit for each row
    data1['Profit'] = data1['SalesAmount'] - data1['TotalCost']
    data2['Profit'] = data2['SalesAmount'] - data2['TotalCost']

    # Group by Year and Quarter to get the total Profit for each Quarter
    profit_by_quarter_2008 = data1.groupby(['Year', 'Quarter'])['Profit'].sum().reset_index()
    profit_by_quarter_2009 = data2.groupby(['Year', 'Quarter'])['Profit'].sum().reset_index()




    # Calculate Profit for each row


    # Group by Year and Quarter to get the total Profit for each Quarter


    # Pivot the data to get a better structure for plotting with streamlit
    profit_pivot_2008 = profit_by_quarter_2008.pivot(index='Quarter', columns='Year', values='Profit').fillna(0)
    profit_pivot_2009 = profit_by_quarter_2009.pivot(index='Quarter', columns='Year', values='Profit').fillna(0)

    profit_pivot_2008_formatted = profit_pivot_2008.applymap(lambda x: '{:.2f}'.format(x))
    profit_pivot_2009_formatted = profit_pivot_2009.applymap(lambda x: '{:.2f}'.format(x))

    profit_pivot_2008.index = profit_pivot_2008.index.astype(str)
    profit_pivot_2009.index = profit_pivot_2009.index.astype(str)

    profit_pivot_2008_formatted
    profit_pivot_2009_formatted


    plt.figure(figsize=(10, 5))
    plt.plot(profit_pivot_2008[2008], marker='o', label='2008')
    plt.plot(profit_pivot_2009[2009], marker='o', label='2009')  # Giả định bạn đã có dữ liệu này
    plt.title('Profit over time (Year/Quarter)')
    plt.xlabel('Quarter')
    plt.ylabel('Profit')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Hiển thị biểu đồ trên Streamlit
    st.pyplot(plt.gcf())

def revenuebyManufacturer(data,product_data):
    
    # Clean the Product DataFrame if necessary (e.g., strip whitespace from 'BrandName')
    product_data['BrandName'] = product_data['BrandName'].str.strip()

    # Group the sales data by 'ProductKey' and sum the 'SalesAmount'
    total_sales_by_product = data.groupby('ProductKey')['SalesAmount'].sum().reset_index()

    # Merge the sales data with the product information to get the 'BrandName'
    merged_data = pd.merge(total_sales_by_product, product_data[['ProductKey', 'BrandName']], on='ProductKey', how='left')

    # Group the merged data by 'BrandName' and sum the 'SalesAmount'
    total_sales_by_brand = merged_data.groupby('BrandName')['SalesAmount'].sum().sort_values(ascending=False).reset_index()

    # Convert the sales amount to billions for better readability in the chart
    total_sales_by_brand['SalesAmount'] = total_sales_by_brand['SalesAmount'] / 1e9
    
    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    bar_chart = plt.bar(total_sales_by_brand['BrandName'], total_sales_by_brand['SalesAmount'], color='skyblue')

    # Adding the values on top of the bars
    for bar in bar_chart:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f"${yval:.2f} tỷ", va='bottom')  # Adjust the position of the text

    # Setting the labels and title
    plt.xlabel('Manufacturer')
    plt.ylabel('Revenue (in billions)')
    plt.title('Revenue by Manufacturer')
    plt.xticks(rotation=45)  # Rotate x-axis labels to show them more clearly

    # Formatting the y-axis to show the currency and billions unit
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"${x:.2f} tỷ"))

    # Show the plot
    plt.tight_layout()
    st.pyplot(plt)
    # st.subheader('- Nhà phân phối sản phẩm của công ty rất đa dạng')
    # st.subheader('- Doanh thu lớn nhất đến từ Contoso và Fabrikam')
def quantitySalesByClass(data, product_data):
    # Merge the product data with sales data on ProductKey to get the ClassName for each sale
    # Merge the product data with sales data on ProductKey to get the ClassName for each sale
    merged_data = pd.merge(data, product_data[['ProductKey', 'ClassName']], on='ProductKey', how='left')

    # Group the merged data by ClassName and sum the SalesQuantity for each class
    quantity_by_class = merged_data.groupby('ClassName')['SalesQuantity'].sum().reset_index()

    # Group the merged data by ClassName and sum the SalesAmount for each class to get the revenue
    revenue_by_class = merged_data.groupby('ClassName')['SalesAmount'].sum().reset_index()

    # Data to plot for quantity
    labels_quantity = quantity_by_class['ClassName']
    sizes_quantity = quantity_by_class['SalesQuantity']
    colors_quantity = ['#1f77b4', '#ff7f0e', '#2ca02c']
    explode_quantity = (0.1, 0, 0)  # explode 1st slice (Deluxe)

    # Data to plot for revenue
    labels_revenue = revenue_by_class['ClassName']
    sizes_revenue = revenue_by_class['SalesAmount']
    colors_revenue = ['#1f77b4', '#ff7f0e', '#2ca02c']
    explode_revenue = (0, 0, 0.1)  # explode 3rd slice (Regular)

    # Plot for Quantity Sales by Class
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes_quantity, explode=explode_quantity, labels=labels_quantity, colors=colors_quantity, autopct='%1.1f%%', shadow=True, startangle=140)
    ax1.set_title('Quantity Sales by Class')
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Display the figure with streamlit
    st.subheader('2. SỐ LƯỢNG SẢN PHẨM THEO LOẠI')
    st.pyplot(fig1)

    # Plot for Revenue by Class
    fig2, ax2 = plt.subplots()
    ax2.pie(sizes_revenue, explode=explode_revenue, labels=labels_revenue, colors=colors_revenue, autopct='%1.1f%%', shadow=True, startangle=140)
    ax2.set_title('Revenue by Class')
    ax2.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle

    # Display the figure with streamlit
    st.subheader('3. LỢI NHUẬN THEO LOẠI ')  
    st.pyplot(fig2)
def quantitySoldByBrand(data, product_data):
    # Plotting the quantity sold by each brand as a bar chart
# Clean up the BrandName column to remove any trailing spaces
    product_data['BrandName'] = product_data['BrandName'].str.strip()

    # Merge the product data with sales data on ProductKey to get the BrandName for each sale
    merged_data_brand = pd.merge(data, product_data[['ProductKey', 'BrandName']], on='ProductKey', how='left')

    # Group the merged data by BrandName and sum the SalesQuantity for each brand
    quantity_by_brand = merged_data_brand.groupby('BrandName')['SalesQuantity'].sum().reset_index()


    # Set the size of the figure
    plt.figure(figsize=(10, 6))

    # Create a bar chart
    plt.bar(quantity_by_brand['BrandName'], quantity_by_brand['SalesQuantity'], color='orange')

    # Add title and labels to the plot
    plt.title('Quantity Sold by Brand')
    plt.xlabel('Brand Name')
    plt.ylabel('Quantity Sold')
    plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability

    # Display the plot
    plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
    st.pyplot(plt)
def quantitySalesByProductSub(product_data, product_subcategory_data, data):
    
    product_subcategory_columns = product_subcategory_data.columns.tolist()

    # Merge the product data with the product subcategory data
    merged_product_subcategory = pd.merge(
        product_data, 
        product_subcategory_data[['ProductSubcategoryKey', 'ProductSubcategoryName']], 
        on='ProductSubcategoryKey', 
        how='left'
    )

    # Merge the merged_product_subcategory with the sales data on ProductKey
    merged_sales_subcategory = pd.merge(
        data, 
        merged_product_subcategory[['ProductKey', 'ProductSubcategoryName']], 
        on='ProductKey', 
        how='left'
    )

    # Group the merged data by ProductSubcategoryName and sum the SalesQuantity
    quantity_sales_by_subcategory = merged_sales_subcategory.groupby('ProductSubcategoryName')['SalesQuantity'].sum().reset_index()

    # Create a bar chart using Matplotlib
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(quantity_sales_by_subcategory['ProductSubcategoryName'], quantity_sales_by_subcategory['SalesQuantity'], color='skyblue')
    ax.set_title('Quantity Sales by Product Subcategory')
    ax.set_xlabel('Quantity Sold')
    ax.set_ylabel('Product Subcategory Name')

    # Display the Matplotlib figure in Streamlit
    st.pyplot(fig)
def profitByQuarterForAll(data, geography_data):
    # First, we need to parse the Date column in the sales data to datetime format
# and then extract the quarter from the date.

    # Assuming sales_data is already loaded and defined
    data['Date'] = pd.to_datetime(data['Date'])
    data['Quarter'] = data['Date'].dt.to_period('Q').astype(str)
    # Merge the sales data with the geography data
    merged_data = pd.merge(
        data, 
        geography_data[['GeographyKey', 'ContinentName']], 
        on='GeographyKey', 
        how='left'
    )
    merged_data['Profit'] = merged_data['SalesAmount'] - merged_data['TotalCost']

    # Group data by ContinentName and Quarter and calculate profit
    profit_by_continent_quarter = merged_data.groupby(['ContinentName', 'Quarter'])['Profit'].sum().unstack(level=0)

    # Create separate line plots for each continent
    num_continents = profit_by_continent_quarter.shape[1]
    fig, axes = plt.subplots(num_continents, 1, figsize=(15, num_continents * 5), sharex=True)

    for i, continent in enumerate(profit_by_continent_quarter.columns):
        axes[i].plot(profit_by_continent_quarter.index.values, profit_by_continent_quarter[continent], marker='o', linestyle='-')
        axes[i].set_title(f'Profit by Quarter for {continent}')
        axes[i].set_ylabel('Profit')
        axes[i].set_xlabel('Quarter')
        axes[i].grid(True)

    # Adjust layout for Streamlit display
    plt.tight_layout()

    # Show the plot in Streamlit
    st.pyplot(fig)

def revenueByYearForCatalog(data, channel_data):
    # Load the channel data
    

    # Assuming sales_data is already loaded and defined
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year

    # Merge sales data with channel data on 'channelKey'
    merged_sales_channel = pd.merge(data, channel_data, left_on='channelKey', right_on='ChannelKey', how='left')

    # Calculate revenue by ChannelName and Year
    revenue_by_channel_year = merged_sales_channel.groupby(['ChannelName', 'Year'])['SalesAmount'].sum().unstack(level=0)

    # Convert the groupby object to a DataFrame
    revenue_by_channel_year_df = revenue_by_channel_year.reset_index()

    # Set up the matplotlib figure for the line plots
    num_channels = revenue_by_channel_year_df.shape[1] - 1  # Subtract 1 for the Year column
    fig, axes = plt.subplots(num_channels, 1, figsize=(12, num_channels * 4), sharex=True)

    # Create a line plot for each channel
    for i, channel in enumerate(revenue_by_channel_year_df.columns[1:]):  # Skip the first column which is Year
        axes[i].plot(revenue_by_channel_year_df['Year'], revenue_by_channel_year_df[channel], marker='o', linestyle='-')
        axes[i].set_title(f'Revenue by Year for {channel}')
        axes[i].set_ylabel('Revenue')
        axes[i].set_xlabel('Year')
        axes[i].grid(True)

    # Adjust layout for Streamlit display
    plt.tight_layout()

    # Show the plot in Streamlit
    st.pyplot(fig)

def revenueByChannelPieChart(data, channel_data):
    # Ensure the Date is in datetime format and extract Year
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year

    # Merge sales data with channel data on 'channelKey'
    merged_sales_channel = pd.merge(data, channel_data, left_on='channelKey', right_on='ChannelKey', how='left')

    # Calculate total revenue by ChannelName
    total_revenue_by_channel = merged_sales_channel.groupby('ChannelName')['SalesAmount'].sum()

    # Create a pie chart for total revenue by channel
    fig, ax = plt.subplots()
    ax.pie(total_revenue_by_channel, labels=total_revenue_by_channel.index, autopct='%1.1f%%', startangle=90)
    ax.set_title('Revenue by Channel')

    # Show the plot in Streamlit
    st.pyplot(fig)

def storeCount(geography_data):
    # Load the geography data from the Excel file
    # Assuming that each row in the geography data represents a store
    # We will count the number of rows for each continent as the number of stores
    store_count_by_continent = geography_data['ContinentName'].value_counts().reset_index()
    store_count_by_continent.columns = ['Khu vực', 'Số lượng cửa hàng']

    # Display the result in Streamlit as a table
    st.table(store_count_by_continent)


