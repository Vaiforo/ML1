import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np

# Функция для загрузки модели


def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Ошибка при загрузке модели: {e}")
        return None


# Заглушка для данных и моделей (замените реальными путями)
DATA_PATH = "sample_data.csv"
MODEL_PATHS = {
    "Linear Regression": "linear_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
    "CatBoost": "catboost.pkl",
    "Stacking": "stacking.pkl",
    "Neural Network": "neural_network.pkl"
}

# Навигация
st.sidebar.title("Навигация")
page = st.sidebar.radio("Перейти на страницу", [
                        "Информация о разработчике", "Информация о датасете", "Визуализации", "Прогнозирование"])

if page == "Информация о разработчике":
    st.title("Информация о разработчике")
    st.write("**Имя:** Бобков Андрей Сергеевич")
    st.write("**Номер группы:** МО-231")
    st.image("logo.jpg", caption="Фото дебила", width=200)
    st.write("**Тема РГР:** Разработка Web-приложения (дашборда) для инференса моделей ML и анализа данных")

elif page == "Информация о датасете":
    st.title("Информация о датасете")
    st.markdown("""
    ### Описание датасета
    Датасет содержит данные для задачи классификации. Он включает 62630 записей и 13 признаков.

    ### Признаки
    - **UTC:**  
    - **Temperature[C]:**
    - **Temperature[C]:**
    - **Temperature[C]:**
    - **Temperature[C]:**
    - **Temperature[C]:**
    - **Temperature[C]:**
    - **Temperature[C]:**
    - **Temperature[C]:**
    - **Temperature[C]:**
    - **Temperature[C]:**
    - **Temperature[C]:**
    - **Temperature[C]:**
    - **Temperature[C]:**
    - **Temperature[C]:**
    - **Temperature[C]:**

    ### Предобработка
    - Удаление пропущенных значений
    - Нормализация числовых признаков
    - Кодирование категориальных переменных

    ### Исследовательский анализ данных (EDA)
    Проведен анализ распределений, корреляций и выбросов.
    """)

# elif page == "Визуализации":
#     st.title("Визуализация зависимостей данных")
#     try:
#         data = pd.read_csv(DATA_PATH)

#         # Визуализация 1: Рассеяние
#         st.subheader("Диаграмма рассеяния")
#         fig1, ax1 = plt.subplots()
#         sns.scatterplot(
#             data=data, x=data.columns[0], y=data.columns[1], ax=ax1)
#         st.pyplot(fig1)

#         # Визуализация 2: Гистограмма
#         st.subheader("Гистограмма")
#         fig2, ax2 = plt.subplots()
#         sns.histplot(data[data.columns[0]], bins=20, ax=ax2)
#         st.pyplot(fig2)

#         # Визуализация 3: Ящик с усами
#         st.subheader("Ящик с усами")
#         fig3, ax3 = plt.subplots()
#         sns.boxplot(data=data, y=data.columns[1], ax=ax3)
#         st.pyplot(fig3)

#         # Визуализация 4: Тепловая карта
#         st.subheader("Тепловая карта корреляций")
#         fig4, ax4 = plt.subplots()
#         sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax4)
#         st.pyplot(fig4)
#     except Exception as e:
#         st.error(f"Ошибка загрузки данных: {e}")

# elif page == "Прогнозирование":
#     st.title("Интерфейс прогнозирования")

#     # Выбор модели
#     model_choice = st.selectbox("Выберите модель", list(MODEL_PATHS.keys()))
#     model = load_model(MODEL_PATHS[model_choice])

#     if model:
#         # Выбор метода ввода
#         input_method = st.radio("Выберите метод ввода данных", [
#                                 "Загрузить CSV", "Ручной ввод"])

#         if input_method == "Загрузить CSV":
#             uploaded_file = st.file_uploader("Выберите CSV файл", type="csv")
#             if uploaded_file is not None:
#                 try:
#                     data = pd.read_csv(uploaded_file)
#                     st.write("Загруженные данные:", data.head())
#                     predictions = model.predict(data)
#                     st.write("Результаты предсказаний:", predictions)
#                 except Exception as e:
#                     st.error(f"Ошибка обработки файла: {e}")

#         else:
#             st.subheader("Ручной ввод данных")
#             # Пример полей ввода (настройте под свои признаки)
#             feature1 = st.number_input("Признак 1", value=0.0)
#             feature2 = st.number_input("Признак 2", value=0.0)

#             if st.button("Сделать предсказание"):
#                 input_data = pd.DataFrame([[feature1, feature2]], columns=[
#                                           'feature1', 'feature2'])
#                 try:
#                     prediction = model.predict(input_data)
#                     st.success(f"Предсказание: {prediction[0]}")
#                 except Exception as e:
#                     st.error(f"Ошибка предсказания: {e}")
