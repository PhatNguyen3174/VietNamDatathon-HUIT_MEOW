import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd

# Read data
products = pd.read_excel("sales_and_inventory_mentor_data\MasterData\Productmaster.xlsx", nrows=200)

# Set page config first
st.set_page_config(
    page_title="Nike Shop",
    page_icon="shopping_trolley",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to create a Bootstrap card for a product
def product_card(product_id, price, description):
    card = f"""
        <div class="card" style="width: 18rem; margin: 10px;">
            <img src="https://via.placeholder.com/150" class="card-img-top" alt="Product Image">
            <div class="card-body">
                <h5 class="card-title">Product ID: {product_id}</h5>
                <p class="card-text">Price: {price}</p>
                <p class="card-text">Description: {description}</p>
                <a href="#" class="btn btn-primary">Add to Cart</a>
            </div>
        </div>
    """
    return card

def main():
    # Use Streamlit commands after setting page config
    selected2 = option_menu(None, ["Home", "Sign In", "Help", 'Join Us'],
                            icons=['house', '', "list-task", 'gear'],
                            menu_icon="cast", default_index=0, orientation="horizontal")
    selected2

    st.sidebar.title("Lifestyle")
    st.sidebar.markdown('---')
    st.sidebar.markdown('---')

    # Display product cards horizontally
    row_html = ""
    for index, row in products.iterrows():
        product_id = row['product_id']
        listing_price = row['listing_price']
        brand_name = row['brand_name']

        card_html = product_card(product_id, listing_price, brand_name)
        row_html += card_html

    st.markdown(row_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
