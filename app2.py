import streamlit as st
import pandas as pd
import plotly.express as px
import shap
from catboost import CatBoostRegressor, Pool
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix
from sklearn.model_selection import train_test_split



df = pd.read_csv('wine8.csv')
st.set_page_config(layout="wide")

st.title('Гушчян Сергей 2023-ФГиИБ-ПИ-1б')
st.write('winery — название винодельни. wine — наименование вина. year — год сбора винограда. rating — средняя оценка вина, выставленная пользователями (по шкале от 1 до 5). num_reviews — количество пользователей, оставивших отзыв о вине. country — страна происхождения (в данном наборе данных — Испания). region — регион производства вина. price — цена в евро (€). type — сорт (разновидность) вина. body — оценка насыщенности (body), отражающая плотность и вес вина во рту (по шкале от 1 до 5). acidity — оценка кислотности, отражающая степень свежести, терпкости и “слюнкообразующий” эффект вина (по шкале от 1 до 5).')
tab1, tab2, tab3, tab4 = st.tabs(["Данные", "Графики зависимостей", "SHAP-анализ",'Метрики модели'])

with tab1:
    st.header("Исходные данные")

    st.dataframe(df)
with tab2:
    st.subheader("Рейтинг vs Цена")
    fig = px.scatter(
        df,
        x="rating",
        y="price",
        labels={"rating": "Рейтинг", "price": "Цена"},
        title="Зависимость рейтинга от цены",
        trendline = 'ols',
        trendline_color_override = 'red'
    )

    st.plotly_chart(fig, use_container_width=True)
    st.write('Видим общее направление движения цены. Оно линейно и направлено вверх')
    st.subheader("Распределения рейтинга вин")
    fig = px.histogram(
        df,
        x='rating',
        nbins=30,
        title='Распределение рейтинга вин',
        labels={'rating': 'Рейтинг','count': 'Количество'},

    )
    fig.update_layout(yaxis_title="Количество")
    st.plotly_chart(fig, use_container_width=True)
    st.write("Большинство вин получили рейтинг 4.2 (самое низкое значение в датасете), далее видим резкий спад на 4.3 и его продолжение вплоть до 4.9" )


    rating_reviews = df.groupby('rating')['num_reviews'].sum().reset_index()
    rating_reviews = rating_reviews.sort_values('rating')
    fig = px.bar(
        rating_reviews,
        x='rating',
        y='num_reviews',
        title='Количество отзывов по рейтингу',
        labels={
            'rating': 'Рейтинг',
            'num_reviews': 'Количество отзывов'
        },
    )

    st.plotly_chart(fig, use_container_width=True)
    st.write('Логичное продолжение прошлого графика. Видим что на рейтинг 4.2 пришлось огромное количество отзывов(2.5млн!)')
with tab3:
    features = [
        'winery_category', 'wine_category', 'region_category', 'type_category',
        'norm_num_reviews', 'price', 'acidity', 'body', 'year'
    ]
    target = 'rating'
    cat_features = [
        'winery_category', 'wine_category', 'region_category', 'type_category'
    ]

    # features = [
    #     'winery', 'wine', 'region', 'type',
    #     'norm_num_reviews', 'price', 'acidity', 'body', 'year'
    # ]
    # target = 'rating'
    # cat_features = [
    #     'winery', 'wine', 'region', 'type'
    # ]

    model = CatBoostRegressor()
    # model2 = CatBoostRegressor()
    model.load_model('catboost_model.json',format='json')
    # model2.load_model('catboost_model123.cbm')
    fig = px.scatter(
        df,
        x='num_reviews',
    )

    explainer = shap.TreeExplainer(model)
    # explainer = shap.TreeExplainer(model2)
    shap_values = explainer.shap_values(df[features])

    st.title('SHAP анализ модели рейтинга вин')

    st.header("Важность признаков")
    fig1, ax1 = plt.subplots()
    shap.summary_plot(shap_values, df[features], show=False)
    st.pyplot(fig1)
    st.write('''На данном графике каждая точка отражает влияние определённого признака на рейтинг. Красный соответствует высоким значениям признака. Синий соответствует низким значениям признака.
    по оси X: SHAP value (положительное = увеличивает рейтинг, отрицательное = уменьшает).
     Можем видеть, что чем выше категория вина, тем положительнее оно влияет на предсказание. Так же и с ценой, чем она выше, тем это положительнее сказывается на предсказании. Но можем видеть что низкое количество отзывов упрощает предсказывание (по графику с прошлой вкладки было видно насколько огромнное колчичество отзывов приходится на низкий рейтинг)
    ''')
    st.header("Рейтинг важности признаков")
    fig2, ax2 = plt.subplots()
    shap.summary_plot(shap_values, df[features], plot_type="bar", show=False)
    st.pyplot(fig2)
    st.write('''wine_category (конкретное название вина) — на первом месте.
Это может означать, что модель уловила, что определённые вина (например, премиальные или культовые) стабильно получают высокие или низкие оценки
norm_num_reviews (нормализованное количество отзывов) — на втором месте.
Чем больше отзывов, тем выше доверие к рейтингу (вина с малым количеством отзывов могут иметь случайные оценки). Также популярные вина (с большим числом отзывов) часто коррелируют с качеством.
price (цена) — на третьем месте.
Цена часто отражает качество вина: дорогие вина обычно получают более высокие оценки из-за лучшего сырья, технологии производства или репутации.
winery_category (категория винодельни) — на четвёртом месте.
Возможно репутация винодельни играет роль(?). Известные винодельни чаще выпускают вина которые нравятся людям
''')

with tab4:
    st.header('Метрики модели')
    def evaluate_model(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        st.title(f"MSE: {mse:.4f} MAE: {mae:.4f} R²: {r2:.4f}\n")

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    test_pool = Pool(X_test, y_test, cat_features=cat_features)
    y_pred_cat = model.predict(test_pool)
    evaluate_model(y_test,y_pred_cat)
    confusion_matrix = confusion_matrix(y_test, y_pred_cat)
    st.write(confusion_matrix)
