import pandas as pd
import numpy as np
import statsmodels.api as sm

# Ganti 'nama_file.csv' dengan nama file CSV Anda
file_path = 'INFLASIIHK.csv'

# Membaca data dari file CSV ke dalam DataFrame
df = pd.read_csv(file_path, sep=';')

# Mengonversi kolom 'bulan' menjadi tipe data datetime
df['Bulan'] = pd.to_datetime(df['Bulan'], format='%B-%y', errors='coerce')

# Set kolom 'bulan' sebagai indeks waktu
df.set_index('Bulan', inplace=True)

# Menambahkan kolom dengan nilai 1 sebagai variabel dependen tambahan
df['const'] = 1

# Menambahkan variabel lag 'INFLASI_lag1', 'INFLASI_lag2', dan 'INFLASI_lag3'
order = 2  # Sesuaikan dengan derajat polinomial yang diinginkan
for t in range(order + 1):
    df[f'z{t}'] = pd.to_numeric(df['INFLASI'].shift(t).str.replace(',', '.'), errors='coerce')

# Membersihkan dan mengonversi kolom 'INFLASI' ke tipe data numerik
df['INFLASI'] = pd.to_numeric(df['INFLASI'].str.replace(',', '.'), errors='coerce')

# Mengonversi kolom 'IHK' ke tipe data numerik
df['IHK'] = pd.to_numeric(df['IHK'].str.replace(',', '.'), errors='coerce')

# Cetak informasi umum tentang DataFrame
print(df.info())

# Cek apakah ada nilai inf atau NaN dalam DataFrame
print("Inf or NaN values in DataFrame:")
print(df.isnull().sum())

# Cek apakah ada nilai inf atau NaN dalam variabel independen (exog)
print("Inf or NaN values in exog:")
exog = df[['const', 'z0', 'z1', 'z2']]
print(exog.isnull().sum())

# Mencetak 10 baris pertama dari DataFrame untuk pemeriksaan lebih lanjut
print(df.head(10))


# Menerapkan model regresi dinamis untuk 'INFLASI'
endog = df['INFLASI']
exog = df[['const', 'z0', 'z1', 'z2']]

# Print the 'exog' DataFrame to identify missing values
print("DataFrame 'exog' with missing values:")
print(exog)

# Check for missing or infinite values in 'exog'
print("Missing or infinite values in 'exog':")
print(exog.isnull().sum())
print(np.isinf(exog).sum())

# Handle missing or infinite values in 'exog'
exog_cleaned = exog.replace([np.inf, -np.inf], np.nan).dropna()

# Print the cleaned 'exog' DataFrame
print("Cleaned DataFrame 'exog':")
print(exog_cleaned)

# Handle missing or infinite values in 'exog'
exog_cleaned = exog.replace([np.inf, -np.inf], np.nan).dropna()

# Print the cleaned 'exog' DataFrame
print("Cleaned DataFrame 'exog':")
print(exog_cleaned)

# Check for missing or infinite values in the cleaned 'exog'
print("Missing or infinite values in cleaned 'exog':")
print(exog_cleaned.isnull().sum())
print(np.isinf(exog_cleaned).sum())

# Menerapkan model regresi dinamis untuk 'INFLASI'
model = sm.OLS(endog.loc[exog_cleaned.index], exog_cleaned)
results = model.fit()

# Menampilkan summary statistik model
print(results.summary())

# Menampilkan p-value untuk setiap variabel
p_values = results.pvalues
print("P-values:")
print(p_values)

# Menentukan tingkat signifikansi (contoh: 0.05)
alpha = 0.05

# Menentukan variabel yang p-value-nya lebih besar dari alpha
non_significant_vars = p_values[p_values > alpha].index

# Jika ada variabel yang tidak signifikan, lakukan regresi ulang tanpa variabel tersebut
if len(non_significant_vars) > 0:
    print(f"Non-significant variables: {non_significant_vars}")
    
    # Membuat model baru tanpa variabel non-signifikan
    exog_reduced = df[['const'] + [var for var in exog.columns if var not in non_significant_vars]]
    model_reduced = sm.OLS(endog, exog_reduced)
    results_reduced = model_reduced.fit()

    # Menampilkan summary statistik model yang baru
    print(results_reduced.summary())
else:
    print("Semua variabel signifikan.")

# Extracting p-values and coefficients for the first regression
p_values_regression1 = results.pvalues
coefficients_regression1 = results.params

# Displaying p-values and coefficients for the first regression
print("P-values for the first regression:")
print(p_values_regression1)

print("\nCoefficients for the first regression:")
print(coefficients_regression1)

# Performing the second regression
# Note: Make sure to replace 'endog' and 'exog_cleaned' with the actual data if they have changed
model2 = sm.OLS(endog.loc[exog_cleaned.index], sm.add_constant(exog_cleaned[['z0', 'z1', 'z2']]))
results2 = model2.fit()

# Extracting p-values and coefficients for the second regression
p_values_regression2 = results2.pvalues
coefficients_regression2 = results2.params

# Displaying p-values and coefficients for the second regression
print("\nP-values for the second regression:")
print(p_values_regression2)

print("\nCoefficients for the second regression:")
print(coefficients_regression2)

