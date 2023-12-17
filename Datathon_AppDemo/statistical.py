import streamlit as st
import pandas as pd
import altair as alt

def filtered_hist(field, label, filter):
    base = alt.Chart().mark_bar().encode(
        x=alt.X(field, bin=alt.Bin(maxbins=10), title=label),
        y="count()",
    ).properties(
        width=300,
    )
    return alt.layer(
        base.transform_filter(filter),
        base.encode(color=alt.value('lightgray'), opacity=alt.value(.7)),
    ).resolve_scale(y='independent')

def main():
    # Load data
    user_cols = ['customer_id', 'customer_name', 'b2b_b2c']
    users = pd.read_excel('sales_and_inventory_private_p1_data\MasterData\Distribution Channel.xlsx', usecols=user_cols, nrows=300)

    product_cols = ['product_id', 'name_description', 'listing_price', 'lifestyle_group']
    products = pd.read_excel('sales_and_inventory_private_p1_data\MasterData\Productmaster.xlsx', usecols=product_cols, nrows=300)

    sales_cols = ['customer_id', 'product_id', 'sold_quantity']
    sales = pd.read_excel('sales_and_inventory_private_p1_data\Sales_private1\TT T01-2022_split_4_private_p1.xlsx', usecols=sales_cols, nrows=300)
    sales = sales[['customer_id', 'product_id', 'sold_quantity']]

    # Altair visualization code
    def filtered_hist(field, label, filter):
        base = alt.Chart().mark_bar().encode(
            x=alt.X(field, bin=alt.Bin(maxbins=10), title=label),
            y="count()",
        ).properties(
            width=300,
        )
        return alt.layer(
            base.transform_filter(filter),
            base.encode(color=alt.value('lightgray'), opacity=alt.value(.7)),
        ).resolve_scale(y='independent')

    # Create filters to be used to slice the data.
    b2b_b2c_filter = alt.selection_multi(fields=["b2b_b2c"])
    b2b_b2c_chart = alt.Chart().mark_bar().encode(
        x="count()",
        y=alt.Y("b2b_b2c:N"),
        color=alt.condition(
            b2b_b2c_filter,
            alt.Color("b2b_b2c:N", scale=alt.Scale(scheme='category20')),
            alt.value("lightgray")),
    ).properties(width=300, height=300, selection=b2b_b2c_filter)

    # Merge data
    users_sales = (
        sales
        .groupby('customer_id', as_index=False)
        .agg({'sold_quantity': ['count', 'mean']})
    )

    # Flatten multi-level columns
    users_sales.columns = [' '.join(col).strip() for col in users_sales.columns.values]

    # Merge with users
    users_sales = users_sales.merge(users, on='customer_id')

    # Streamlit App
    st.title("Sales Dashboard")

    # Display Altair charts using st.altair_chart
    st.altair_chart(filtered_hist('sold_quantity count', '# sales / user', b2b_b2c_filter))
    st.altair_chart(filtered_hist('sold_quantity mean', 'mean user sale', b2b_b2c_filter))
    st.altair_chart(b2b_b2c_chart)

    # Display raw data if needed
    st.write(users_sales)

if __name__ == "__main__":
    main()
