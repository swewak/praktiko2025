import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("wine7.csv")

# df = df.drop('country', axis=1)
# df['type'] = df.groupby('winery')['type'].transform(
#     lambda x: x.fillna(x.mode()[0]) if not x.mode().empty else 'Unknown')
# df['body'] = df['body'].fillna(df['body'].median())
# df['acidity'] = df['acidity'].fillna(df['acidity'].median())
#
# def cat(rating):
#     if 4.0 <= rating <= 4.2999999: return 1
#     elif 4.3 <= rating <= 4.49999999: return 2
#     elif 4.5 <= rating <= 5.0: return 3
#     return 1  # значение по умолчанию
#
# # 3. Применение категоризации с проверкой наличия столбцов
# if 'winery' in df.columns:
#     mean_rating = df.groupby('winery')['rating'].mean().reset_index()
#     mean_rating['winery_category'] = mean_rating['rating'].apply(cat)
#     df = df.merge(mean_rating[['winery', 'winery_category']], on='winery', how='left')
#
# if 'wine' in df.columns:
#     mean_rating1 = df.groupby('wine')['rating'].mean().reset_index()
#     mean_rating1['wine_category'] = mean_rating1['rating'].apply(cat)
#     df = df.merge(mean_rating1[['wine', 'wine_category']], on='wine', how='left')
#
# if 'region' in df.columns:
#     mean_rating2 = df.groupby('region')['rating'].mean().reset_index()
#     mean_rating2['region_category'] = mean_rating2['rating'].apply(cat)
#     df = df.merge(mean_rating2[['region', 'region_category']], on='region', how='left')
#
# if 'type' in df.columns:
#     mean_rating3 = df.groupby('type')['rating'].mean().reset_index()
#     mean_rating3['type_category'] = mean_rating3['rating'].apply(cat)
#     df = df.merge(mean_rating3[['type', 'type_category']], on='type', how='left')
#
# for cat in ['winery_category', 'wine_category', 'region_category', 'type_category']:
#     if cat in df.columns:
#         df[cat] = df[cat].fillna(1)
#
# cols_to_normalize = ['rating','num_reviews','price']
# cols_to_normalize = [col for col in cols_to_normalize if col in df.columns]
#
# if cols_to_normalize:
#     from sklearn.preprocessing import MinMaxScaler
#     scaler = MinMaxScaler()
#     df[[f'norm_{col}' for col in cols_to_normalize]] = scaler.fit_transform(df[cols_to_normalize])
#
# df['year'] = df['year'].replace(0, 'N.V.')
#
# df.to_csv("wine6.csv", index=False)
#
#
# print("Обработка завершена успешно!")
# print("Сохраненные файлы:")
# print("- wine_processed.csv (полные обработанные данные)")
# print("- wine_train.csv (обучающая выборка)")
# print("- wine_test.csv (тестовая выборка)")

# df['year_not_defined'] = df['year'].apply(lambda x: 1 if x == 'N.V.' else 0)

df['year'] = df['year'].replace("N.V.", 0)

df.to_csv("wine8.csv", index=False)