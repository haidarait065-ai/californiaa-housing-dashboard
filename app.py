import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
from math import sqrt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# دالة لتصحيح عرض النص العربي داخل الرسوم البيانية
def ar(text):
    return get_display(arabic_reshaper.reshape(text))


st.set_page_config(page_title="California Housing", layout="wide")
st.title("لوحة استكشاف بيانات إسكان كاليفورنيا")
st.caption("California Housing Dashboard - AML601")


@st.cache_data
def load_data():
    return fetch_california_housing(as_frame=True).frame


@st.cache_data
def prepare_regression():
    df = load_data()
    X = df.drop(columns="MedHouseVal")
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train)
    X_test_s = sc.transform(X_test)
    return X_train_s, X_test_s, y_train, y_test


@st.cache_data
def get_pca():
    df = load_data()
    X = df.drop(columns="MedHouseVal")
    X_all_s = StandardScaler().fit_transform(X)
    return PCA(n_components=2, random_state=42).fit_transform(X_all_s)


@st.cache_data
def run_kmeans(k):
    X_pca = get_pca()
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_pca)
    sil = silhouette_score(X_pca, km.labels_, sample_size=5000, random_state=42)
    return km.labels_, km.inertia_, sil


df = load_data()
X = df.drop(columns="MedHouseVal")

tab1, tab2, tab3 = st.tabs(["البيانات", "الانحدار", "التجميع"])

# ---- تبويب البيانات
with tab1:
    st.write("عدد الصفوف:", df.shape[0])
    st.dataframe(df.describe())
    feature = st.selectbox("اختر ميزة لعرض توزعها", list(df.columns))
    fig, ax = plt.subplots()
    ax.hist(df[feature], bins=40)
    ax.set_title(feature)
    st.pyplot(fig)
    plt.close(fig)

# ---- تبويب الانحدار
with tab2:
    model_name = st.selectbox("اختر النموذج", ["Linear Regression", "Ridge", "Lasso"])
    alpha = st.slider("قيمة alpha (لنموذجي Ridge و Lasso)", 0.001, 5.0, 0.1)

    X_train_s, X_test_s, y_train, y_test = prepare_regression()

    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Ridge":
        model = Ridge(alpha=alpha)
    else:
        model = Lasso(alpha=alpha)

    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", round(sqrt(mean_squared_error(y_test, y_pred)), 3))
    c2.metric("MAE", round(mean_absolute_error(y_test, y_pred), 3))
    c3.metric("R2", round(r2_score(y_test, y_pred), 3))

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, s=3)
    ax.set_xlabel(ar("القيمة الحقيقية"))
    ax.set_ylabel(ar("القيمة المتنبأ بها"))
    st.pyplot(fig)
    plt.close(fig)

# ---- تبويب التجميع
with tab3:
    k = st.slider("عدد العناقيد k", 2, 8, 4)

    X_pca = get_pca()
    labels, inertia, sil = run_kmeans(k)
    st.write("Inertia:", round(inertia, 2), " | Silhouette:", round(sil, 3))

    col1, col2 = st.columns(2)

    fig1, ax1 = plt.subplots()
    ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="rainbow", s=3)
    ax1.set_title(ar("العناقيد في فضاء PCA"))
    col1.pyplot(fig1)
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    ax2.scatter(df["Longitude"], df["Latitude"], c=labels, cmap="rainbow", s=3)
    ax2.set_title(ar("العناقيد على الخريطة"))
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    col2.pyplot(fig2)
    plt.close(fig2)
