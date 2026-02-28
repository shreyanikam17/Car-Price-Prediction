
import streamlit as st
import numpy as np
import pandas as pd
import pickle as pkl
st.title("Car Price Prediction App")
st.write("Welcome To Miss. Shreya Nikam Project")
st.write("We take following inputs from user:")
st.write("1. Company:")
st.write("2. Name:")
st.write("3. Year:")
st.write("4. Kms Driven:")
st.write("5. Fuel Type:")

model=pkl.load(open("CPP.pkl", "rb+"))
ds=pd.read_csv("cleaned_data.csv")
companies=sorted(ds["company"].unique())
fuel_types=sorted(ds["fuel_type"].unique())

years=[]
for i in range(1995, 2025):
    years.append(i)


company=st.sidebar.selectbox("Select Company", companies, key="company")
names=sorted(ds[ds["company"]==company]["name"].unique())
name=st.sidebar.selectbox("Select name", names, key="name")

year=st.sidebar.selectbox("Select year", years, key="year")
kms_driven=st.sidebar.text_input("Enter kms_driven","10000")
fuel_type=st.sidebar.selectbox("Select fuel Type", fuel_types, key="fuel_types")
st.write("You Selected:", company, name, year)

if st.sidebar.button("predict"):
    columns=["company", "name", "year", "kms_driven","fuel_type"]
    data=[[company, name, year, kms_driven, fuel_type]]

    st.write("You Provided Following information")
    st.write("1. Company:", company)
    st.write("2. Name:", name)
    st.write("3. Year:", year)
    st.write("4. Kms Driven:", kms_driven)
    st.write("5. Fuel Type:", fuel_type)

    myinput=pd.DataFrame(data=data, columns=columns)
    result= model.predict(myinput)
    st.write("Predicted Price:", round(result[0,0]))
