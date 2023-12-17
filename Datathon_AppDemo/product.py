# product.py
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import subprocess
import matplotlib.pyplot as plt


# Read data
# products = pd.read_excel('InventoryAndSale_snapshot_data\MasterData\Productmaster.xlsx', nrows=300)

# Set page config first
st.set_page_config(
    page_title="Nike Shop",
    page_icon="shopping_trolley",
    layout="wide",
    initial_sidebar_state="expanded"
)

# def get_price_by_id(product_id):
#     """Function to get the price of a product by its ID."""
#     product_row = products[products['product_id'] == product_id]
#     if not product_row.empty:
#         return product_row.iloc[0]['listing_price']
#     else:
#         return None

# def get_brand_by_id(product_id):
#     """Function to get the brand of a product by its ID."""
#     product_row = products[products['product_id'] == product_id]
#     if not product_row.empty:
#         return product_row.iloc[0]['brand_name']
#     else:
#         return None

from CF import *
# Initialize session_state
def product_page():

    selected_option = option_menu(None, ["Product","Home", "Sign In", "Statistical", 'Setting'],
                            icons=['house', '', "list-task", 'gear'],
                            menu_icon="cast", default_index=0, orientation="horizontal")
    
    # if selected_option == "Home":
    #     subprocess.run(["streamlit", "run", "app.py"])
    navigation = st.sidebar.selectbox("Go to", ["Recommend Product", "Statiscal"])

    # st.sidebar.image('Logo.png',)
    # st.sidebar.title("Lifestyle")
    # st.sidebar.markdown('---')
    # st.sidebar.selectbox("Gender", products.gender.unique())
    # st.sidebar.markdown('---')
    # st.sidebar.selectbox("Lifestyle", products.lifestyle_group.unique())
    # st.sidebar.markdown('---')
    # st.sidebar.selectbox("Shop By Price", products.price_group.unique())

    # Read product_id from the file
    with open("selected_product.txt", "r") as file:
        product_id = file.read()

        col1, col2 = st.columns([2, 5])
        with col1.container(border=True):
            st.image('https://static.nike.com/a/images/c_limit,w_592,f_auto/t_product_v1/9296d8da-52d3-43f9-9378-59919dae381f/zoom-vomero-5-shoes-qZG4RJ.png', use_column_width=True)

        # with col2.container(border=True):
        #     st.title(product_id)
        #     st.write(f"Brand: {get_brand_by_id(product_id)}")
        #     st.write(f"Price: {get_price_by_id(product_id)}")
    with open("rmd.txt", "r", encoding="utf-8") as file:
        lines = file.readlines()
        data = [line.strip().split(', ') for line in lines]
    if navigation == "Recommend Product":
        with st.container(border=True):
            st.title("Recommend Items")
            col1, col2, col3 = st.columns([4, 4, 4])

            for i, (listing_price, lifestyle_group, description, color, size, product_id) in enumerate(data[:3]):
                product_image_url = 'https://static.nike.com/a/images/c_limit,w_592,f_auto/t_product_v1/9296d8da-52d3-43f9-9378-59919dae381f/zoom-vomero-5-shoes-qZG4RJ.png'  # Assuming image file names are based on product_id

                # Select the current column based on the index
                if i % 3 == 0:
                    current_col = col1
                elif i % 3 == 1:
                    current_col = col2
                else:
                    current_col = col3

                with current_col.container(border=True):
                    st.image(product_image_url, caption="Product Image", use_column_width=True)
                    st.write(f'Listing Price: {listing_price}')
                    st.write(f'Lifestyle Group: {lifestyle_group}')
                    st.write(f'Name Description: {description}')
                    st.write(f'Product ID: {product_id}')
    




# Run the product_page function
product_page()
