import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("dataset.csv", encoding='utf-8')
print(data.head())  # Tampilkan baris awal untuk memeriksa apakah ada data.

# Bersihkan nama kolom: hilangkan spasi, ganti spasi dengan underscore
data.columns = data.columns.str.strip()
data.columns = data.columns.str.replace(' ', '_')

# Rename kolom untuk kesederhanaan dan kejelasan
data.rename(columns={
    'Choose_your_gender': 'Gender',
    'What_is_your_course?': 'Course',
    'Your_current_year_of_Study': 'Year_of_Study',
    'What_is_your_CGPA?': 'CGPA',
    'Do_you_have_Depression?': 'Depression',
    'Do_you_have_Anxiety?': 'Anxiety',
    'Do_you_have_Panic_attack?': 'Panic_Attack',
    'Did_you_seek_any_specialist_for_a_treatment?': 'Treatment_Seeked'
}, inplace=True)

# Debug: Tampilkan kolom yang tersedia setelah pembersihan
print("Kolom yang tersedia setelah pembersihan:", data.columns)

# Validasi kolom yang dibutuhkan
required_columns = ['Gender', 'Age', 'Course', 'Year_of_Study', 'CGPA', 'Marital_status', 'Depression', 'Anxiety', 'Panic_Attack', 'Treatment_Seeked']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"Kolom '{col}' tidak ditemukan. Periksa dataset Anda.")

# 1. Menghitung rata-rata CGPA dari kolom dengan format 'X - Y'
def calculate_average_cgpa(cgpa_range):
    start, end = map(float, cgpa_range.split(' - '))
    return (start + end) / 2

data['CGPA'] = data['CGPA'].apply(calculate_average_cgpa)

# 2. Convert Year_of_Study menjadi numerik
data['Year_of_Study'] = data['Year_of_Study'].str.extract(r'(\d+)').astype(int)

# 3. Konversi kolom 'Treatment_Seeked' menjadi boolean
data['Treatment_Seeked'] = data['Treatment_Seeked'].apply(lambda x: True if x == 'Yes' else False)

# 4. Mengelompokkan kategori 'Course' menjadi 3 kategori teratas, sisanya menjadi 'Others'
course_counts = data['Course'].value_counts()
top_courses = course_counts.nlargest(3).index
data['Course'] = data['Course'].apply(lambda x: x if x in top_courses else 'Others')

# 5. Encoding variabel kategorikal
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['Marital_status'] = label_encoder.fit_transform(data['Marital_status'])
data['Course'] = label_encoder.fit_transform(data['Course'])
data['Depression'] = label_encoder.fit_transform(data['Depression'])
data['Anxiety'] = label_encoder.fit_transform(data['Anxiety'])
data['Panic_Attack'] = label_encoder.fit_transform(data['Panic_Attack'])

# Pilih fitur untuk model
features = data[['Gender', 'Age', 'Course', 'Year_of_Study', 'CGPA', 'Marital_status', 'Depression', 'Anxiety', 'Panic_Attack']]
labels = data['Treatment_Seeked']

# Pembagian data ke train dan test set
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

# Simpan data latih dan uji ke file CSV
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Data berhasil diproses dan disimpan.")
