import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

# 1. Veriyi yükle
contest_df = pd.read_csv('contest_data.csv', encoding='latin1')
country_df = pd.read_csv('country_data.csv', encoding='latin1')
song_df = pd.read_csv('song_data.csv', encoding='latin1')

# 2. Song ve Contest verilerini birleştir (country sütununda sorun varsa kontrol et)
# Eğer country sütunu yoksa, verileri kontrol et
print("contest_df columns:", contest_df.columns)
print("song_df columns:", song_df.columns)
print("country_df columns:", country_df.columns)

df = contest_df.merge(song_df, on=['year'], how='inner')
df = df.merge(country_df, on='country', how='left')

# 3. winner sütunu oluştur (final_place == 1 ise kazanır)
df['winner'] = df['final_place'].apply(lambda x: 1 if x == 1 else 0)

# 4. Özellik ve hedef belirle
drop_cols = ['year', 'host', 'date', 'semi_countries', 'final_countries', 'jury_countries_voting_final',
             'televote_countries_voting_final', 'song_name', 'artist_name', 'winner', 'final_place']
X = df.drop(columns=[col for col in drop_cols if col in df.columns])
y = df['winner']

# 5. Kategorik verileri encode et
label_encoders = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# 6. Modeli eğit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("✅ Test Doğruluğu:", accuracy_score(y_test, y_pred))

# 7. 2025 tahmin veri setini oluştur
# 2025 yılında yarışacak ülkeleri önceki yıllardan alalım
countries_2025 = df['country'].unique()

# Modelin beklediği sütunlar
features = X_train.columns

predict_2025_rows = []

for country in countries_2025:
    row = {}
    for feature in features:
        if 'country' in feature.lower():
            # Eğer country sütunu kategorik encode ise
            # O sütunun kodlandığı ülkeyle eşleşirse 1, değilse 0 yap
            if feature in label_encoders:
                # Feature bir kategorik sütunsa o sütunun kendisi yok, encoded hali var
                # Burada country feature tek başına ise encoded değerini koyarız
                # Ama label encoder birden fazla ülke olduğu için feature country_ismi olamaz,
                # Biz burada direkt encoded değer için şu şekilde yapacağız:
                # Eğer feature == 'country' ise encoded değeri ver.
                # Çünkü encoded 'country' sütunu 0,1,2,... olarak var.
                # Bu yüzden ülkeyi label encoder ile encode edip ona eşit değer koyacağız
                # Diğer sütunlar 0.
                if feature == 'country':
                    # Country encoding
                    le = label_encoders[feature]
                    try:
                        code = le.transform([country])[0]
                    except:
                        code = 0
                    row[feature] = code
                else:
                    # Diğer sütunlar 0
                    row[feature] = 0
            else:
                row[feature] = 0
        else:
            # Numerik ise ortalama veya 0 koyabiliriz
            if feature in X_train.columns:
                # Ortalama koy
                row[feature] = X_train[feature].mean()
            else:
                row[feature] = 0
    predict_2025_rows.append(row)

X_2025 = pd.DataFrame(predict_2025_rows)

# 8. Tahmin yap
probs = model.predict_proba(X_2025)[:, 1]

# 9. Sonuçları derle
results = pd.DataFrame({'country': countries_2025, 'win_probability': probs})

# En yüksek olasılık ve ülke
winner = results.loc[results['win_probability'].idxmax()]
print(f"\n🏆 2025 Eurovision Tahmini Kazanan: {winner['country']} - Kazanma Olasılığı: {winner['win_probability']:.2%}")

# İlk 5 ülke
print("\n--- En Yüksek Olasılıklı İlk 5 Ülke ---")
print(results.sort_values(by='win_probability', ascending=False).head(5).to_string(index=False))
