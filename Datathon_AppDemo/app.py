# app.py
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import subprocess

# Read data
# products_s = pd.read_excel('sales_and_inventory_private_p1_data\MasterData\Productmaster.xlsx', nrows=300)
k = 3
# Set page config first
st.set_page_config(
    page_title="Nike Shop",
    page_icon="shopping_trolley",
    layout="wide",
    initial_sidebar_state="expanded"
)
from CF import *
def main():
    
    user_cols = ['customer_id', 'customer_name', 'b2b_b2c']
    users = pd.read_excel('sales_and_inventory_private_p1_data\MasterData\Distribution Channel.xlsx', usecols=user_cols)

    product_cols = ['product_id', 'name_description', 'listing_price', 'lifestyle_group']
    products = pd.read_excel('sales_and_inventory_private_p1_data\MasterData\Productmaster.xlsx', usecols=product_cols)

    sales_cols = ['customer_id', 'product_id', 'sold_quantity']
    sales = pd.read_excel('sales_and_inventory_private_p1_data\Sales_private1\TT T06-2022_split_4_private_p1.xlsx', usecols=sales_cols)

    # Check the length of users and unique customer_id in sales
    print("Length of users:", len(users))
    print("Length of unique customer_id in sales:", len(sales['customer_id'].unique()))

    # Remove users not present in sales
    users = users[users['customer_id'].isin(sales['customer_id'].unique())]

    # Check the length of products and unique product_id in sales
    print("Length of products:", len(products))
    print("Length of unique product_id in sales:", len(sales['product_id'].unique()))

    # Remove products not present in sales
    products = products[products['product_id'].isin(sales['product_id'].unique())]

    def build_rating_sparse_tensor(sales_df):
        """
        Args:
            sales_df: a pd.DataFrame with `customer_id`, `product_id`, and `sold_quantity` columns.
        Returns:
            a tf.SparseTensor representing the ratings matrix.
        """
        # Convert 'sold_quantity' to numeric, coercing errors to NaN
        sales_df['sold_quantity'] = pd.to_numeric(sales_df['sold_quantity'], errors='coerce')

        # Drop rows with NaN in 'sold_quantity' (non-numeric values)
        sales_df = sales_df.dropna(subset=['sold_quantity'])

        # Convert 'sold_quantity' to integers
        sales_df['sold_quantity'] = sales_df['sold_quantity'].astype(int)

        # Create unique integer indices for 'customer_id' and 'product_id'
        customer_ids = pd.factorize(sales_df['customer_id'])[0]
        # product_ids = pd.factorize(sales_df['product_id'])[0]
        product_ids, product_id_mapping = pd.factorize(sales_df['product_id'])


        indices = np.column_stack((customer_ids, product_ids))
        values = sales_df['sold_quantity'].values

        # Use unique counts for dense_shape
        # dense_shape = [len(sales_df['customer_id'].unique()), len(sales_df['product_id'].unique())]
        dense_shape=[users.shape[0], products.shape[0]]

        return tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape), product_id_mapping
    
    def build_model(sales, embedding_dim=3, init_stddev=1.):
        """
        Args:
            sales: a DataFrame of the sales
            embedding_dim: the dimension of the embedding vectors.
            init_stddev: float, the standard deviation of the random initial embeddings.
        Returns:
            model: a CFModel.
        """
        # Split the sales DataFrame into train and test.
        train_sales, test_sales = split_dataframe(sales)
        print('ok split')
        print(train_sales)

        # SparseTensor representation of the train and test datasets.
        A_train, product_id_mapping = build_rating_sparse_tensor(train_sales)
        print('ok train split')
        A_test, _ = build_rating_sparse_tensor(test_sales)

        # Initialize the embeddings using a normal distribution.
        U = tf.Variable(tf.random_normal(
            [A_train.dense_shape[0], embedding_dim], stddev=init_stddev))
        V = tf.Variable(tf.random_normal(
            [A_train.dense_shape[1], embedding_dim], stddev=init_stddev))
        train_loss = sparse_mean_square_error(A_train, U, V)
        test_loss = sparse_mean_square_error(A_test, U, V)
        metrics = {
            'train_error': train_loss,
            'test_error': test_loss
        }
        embeddings = {
            "customer_id": U,
            "product_id": V
        }
        return CFModel(embeddings, train_loss, [metrics], product_id_mapping)
    def movie_neighbors(model, product_id_substring, measure=DOT, k=6):
        a = np.array(model._product_id_mapping)
        print(a)
        product_id = np.where(a == product_id_substring)[0][0]

        scores = compute_scores(
            model.embeddings["product_id"][product_id], model.embeddings["product_id"],
            measure)
        score_key = measure + ' score'
        df = pd.DataFrame({
            score_key: list(scores)
        })
        top_indices = list(df.sort_values([score_key], ascending=False).head(k).index)
        product_info = products.iloc[top_indices]  # Fix this line
        # print(product_info.head(k))mm,
        # display.display(df.sort_values([score_key], ascending=False).head(k))
        return product_info.head(k), df.sort_values([score_key], ascending=False).head(k)


    model = build_model(sales, embedding_dim=30, init_stddev=0.001)
    model.train(num_iterations=1500, learning_rate=10.)
    print("Done build model")

    selected2 = option_menu(None, ["Home", "Sign In", "Statistical", 'Setting'],
                            icons=['house', '', "list-task", 'gear'],
                            menu_icon="cast", default_index=0, orientation="horizontal")
    
    
    st.sidebar.image('Logo.png')
    st.sidebar.title("Datathon")

    # Sidebar navigation


    # Display product cards horizontally
    col1, col2, col3 = st.columns([4, 4, 4])

    for i, row in products[:25].iterrows():
        product_id = row['product_id']
        listing_price = row['listing_price']
        # brand_name = row['brand_name']
            
        if i % 3 == 0:
            current_col = col1
        elif i % 3 == 1:
            current_col = col2
        else:
             current_col = col3
        with current_col.container(border=True):
                st.image('https://static.nike.com/a/images/c_limit,w_592,f_auto/t_product_v1/9296d8da-52d3-43f9-9378-59919dae381f/zoom-vomero-5-shoes-qZG4RJ.png', caption='Product Image', use_column_width=True)
                st.write(f"Product ID: {product_id}")
                st.write(f"Price: {listing_price}")
                # st.write(f"Description: {brand_name}")
                button_key = f"Add_to_Cart_{product_id}"
                if st.button(f"Add to Cart - {product_id}"):
                    product_df, scores = movie_neighbors(model, product_id, DOT, k)
                    print(product_df)
                    with open("rmd.txt", "w",encoding="utf-8") as f_r:
                        for p, row2 in product_df.iterrows():
                            listing_price_1 = row2['listing_price']
                            lifestyle_group = row2 ['lifestyle_group']
                            name_description = row2['name_description']
                            product_id_1 = row2['product_id']
                         
                            line_to_write = f"{listing_price_1}, {lifestyle_group}, {name_description}, {product_id_1}\n"
                            f_r.write(line_to_write)

                    with open("selected_product.txt", "w") as f:
                        f.write(product_id)
                    st.session_state.product_id = product_id
                    subprocess.run(["streamlit", "run", "product.py"])

if __name__ == "__main__":
    main()