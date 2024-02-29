import streamlit as st
import pandas as pd
import shap
import pickle
import matplotlib.pyplot as plt


st.write("""
# Boston House Price Prediction App

This App predicts the **The Boston House Price!**
""")

st.write("---")

X = pd.read_csv("./test.csv")
boston = X
boston["chas_float"] = boston["chas"].astype(float)
boston["rad_float"] = boston["rad"].astype(float)
boston["tax_float"] = boston["tax"].astype(float)

# Sidebar
# Header of Specify of Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    crim = st.sidebar.slider('CRIM', boston.crim.min(), boston.crim.max(), boston.crim.mean())
    zn = st.sidebar.slider('ZN', boston.zn.min(), boston.zn.max(), boston.zn.mean())
    indus = st.sidebar.slider('INDUS', boston.indus.min(), boston.indus.max(), boston.indus.mean())
    chas = st.sidebar.slider('CHAS', boston.chas_float.min(), boston.chas_float.max(), boston.chas_float.mean())
    nox = st.sidebar.slider('NOX', boston.nox.min(), boston.nox.max(), boston.nox.mean())
    rm = st.sidebar.slider('RM', boston.rm.min(), boston.rm.max(), boston.rm.mean())
    age = st.sidebar.slider('AGE', boston.age.min(), boston.age.max(), boston.age.mean())
    dis = st.sidebar.slider('DIS', boston.dis.min(), boston.dis.max(), boston.dis.mean())
    rad = st.sidebar.slider('RAD', boston.rad_float.min(), boston.rad_float.max(), boston.rad_float.mean())
    tax = st.sidebar.slider('TAX', boston.tax_float.min(), boston.tax_float.max(), boston.tax_float.mean())
    ptratio = st.sidebar.slider('PTRATIO', boston.ptratio.min(), boston.ptratio.max(), boston.ptratio.mean())
    black = st.sidebar.slider('BLACK', boston.black.min(), boston.black.max(), boston.black.mean())
    lstat = st.sidebar.slider('LSTAT', boston.lstat.min(), boston.lstat.max(), boston.lstat.mean())

    data = {'crim': crim,
            'zn': zn,
            'indus': indus,
            'chas': chas,
            'nox': nox,
            'rm': rm,
            'age': age,
            'dis': dis,
            'rad': rad,
            'tax': tax,
            'ptratio': ptratio,
            'black': black,
            'lstat': lstat}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

with open('./modelbuilding/model.pkl', 'rb') as f:
    model = pickle.load(f)

prediction = model.predict(df)
st.header('Prediction of MDEV')
st.write(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')