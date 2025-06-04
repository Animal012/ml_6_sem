from sklearn.datasets import load_wine
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка датасета
wine_data = load_wine()
wine_df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)

# Описание датасета
print("Описание датасета:")
print("Датасет 'Wine' содержит информацию о различных химических свойствах вин.")
print("Он используется для классификации вин на основе их химического состава.")
print("\nОсновные характеристики:")
print(f"Количество образцов: {wine_df.shape[0]}")
print(f"Количество признаков: {wine_df.shape[1]}")
print("Типы данных признаков:")
print(wine_df.dtypes)

# Корреляционная матрица
corr_matrix = wine_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Корреляционная матрица признаков вин")
plt.show()

# Регрессионная матрица (pairplot)
selected_features = ["alcohol", "malic_acid", "color_intensity", "proline"]
sns.pairplot(wine_df[selected_features], kind="reg") 
plt.suptitle("Регрессионная матрица для выбранных параметров", y=1.02)
plt.show()
