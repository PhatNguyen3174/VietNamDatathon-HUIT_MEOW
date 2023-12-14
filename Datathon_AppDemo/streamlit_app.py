import streamlit as st

# Create a Streamlit app
st.title("Fashion Recommendation App")

# User input: preferences (e.g., style, color, occasion)
style = st.selectbox("Select your preferred style:", ["Casual", "Formal", "Sporty"])
color = st.selectbox("Choose a color:", ["Red", "Blue", "Black", "White"])
occasion = st.selectbox("Select an occasion:", ["Work", "Party", "Outdoor"])

# Based on user input, display recommended fashion items
st.header("Recommended Items:")
if style == "Casual":
    st.write("**Casual Outfit:** Jeans, T-shirt, Sneakers")
elif style == "Formal":
    st.write("**Formal Outfit:** Suit, Dress Shirt, Oxfords")
else:
    st.write("**Sporty Outfit:** Leggings, Tank Top, Running Shoes")

# You can add more logic here to customize recommendations further

# Footer
st.markdown("---")
st.write("Explore more options or refine your preferences!")
