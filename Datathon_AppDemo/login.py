import streamlit as st
import os

# Set the background image for the entire page
page_bg_img = """
<style>
    [data-testid="stAppViewContainer"] {
        background-image: url(https://ts.hufi.edu.vn/tttstt/images/tt-tstt/5155dac03c64c13a9875_1.jpg);
        background-size: cover;
        margin-top: 20px;
        padding-top: 0px;
    }
    
    .stApp * {
        color: black !important;
    }

    [data-testid="stHeader"] {
        background-color: rgba(0, 0, 0, 0);
    }
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


def login():
    st.title("Đăng nhập")
    email = st.text_input("Tài khoản")
    password = st.text_input("Mật khẩu", type="password")
    login_button = st.button("Đăng nhập")

    if login_button:
        if email == "" or password == "":
            st.write("Error: Email and password cannot be empty.")
        elif email == "admin" and password == "1234":
            st.success("Success Login")
            # Điều hướng đến trang main.py
            os.system("streamlit run app.py")
        else:
            st.error("False Login")


if __name__ == "__main__":
    login()
