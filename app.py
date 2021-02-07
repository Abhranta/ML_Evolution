import streamlit as st
import app1
import app2
import streamlit as st
PAGES = {
    "Regression": app1,
    "Classification": app2
     
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()