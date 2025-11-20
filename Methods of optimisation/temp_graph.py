import pandas as pd
import matplotlib.pyplot as plt

# Имя файла с измерениями
FILENAME = "timings.csv"

# Загружаем CSV
data = pd.read_csv(FILENAME)

# Ожидаемые столбцы в CSV:
# Omega, Recursive, ACC, Classic
# Пример:
# 1, 0.00013, 0.00004, 0.00007
# 2, 0.00150, 0.00093, 0.00028
# 3, 0.01580, 0.00750, 0.00130
# 4, 0.60000, 0.04000, nan
# 5, 1.20000, 0.15000, nan
# ...

# Строим график
plt.figure(figsize=(10,5))

plt.plot(data["Omega"], data["Recursive"], marker='o', label="Recursive Magnus")
plt.plot(data["Omega"], data["ACC"], marker='s', label="ACC Magnus")
plt.plot(data["Omega"], data["Classic"], marker='^', label="Classic Magnus")

# Подписи
plt.xlabel("Порядок Ω (Omega)")
plt.ylabel("Время вычисления (сек)")
plt.title("Сравнение производительности методов Магнуса")

# Легенда и сетка
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Подписи точек (по желанию)
for i, row in data.iterrows():
    plt.text(row["Omega"], row["Recursive"], f"{row['Recursive']:.3e}", fontsize=8)
    plt.text(row["Omega"], row["ACC"], f"{row['ACC']:.3e}", fontsize=8)
    plt.text(row["Omega"], row["Classic"], f"{row['Classic']:.3e}", fontsize=8)

# Сохраняем и показываем
plt.tight_layout()
plt.savefig("magnus_timing_plot.png", dpi=300)
plt.show()

print("График сохранен: magnus_timing_plot.png")
