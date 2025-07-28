import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

def predict_eurovision_winner_2025_multi_csv(contest_file, country_file, song_file):
    """
    Birden fazla Eurovision veri setini birleştirerek 2025 kazananını tahmin eder.

    Args:
        contest_file (str): Yarışma verilerini içeren CSV dosya yolu (örn: 'contest_data.csv').
        country_file (str): Ülke verilerini içeren CSV dosya yolu (örn: 'country_data.csv').
        song_file (str): Şarkı verilerini içeren CSV dosya yolu (örn: 'song_data.csv').

    Returns:
        str: Tahmini kazanan ülke veya bir hata mesajı.
    """
    try:
        # 1. Veri setlerini yükle (kodlama sorununu çözdüğün varsayımıyla latin1 kullandım)
        df_contest = pd.read_csv(contest_file, encoding='latin1')
        df_country = pd.read_csv(country_file, encoding='latin1')
        df_song = pd.read_csv(song_file, encoding='latin1')

        print(f"'{contest_file}' yüklendi. İlk 5 satır:\n{df_contest.head()}\n")
        print(f"'{country_file}' yüklendi. İlk 5 satır:\n{df_country.head()}\n")
        print(f"'{song_file}' yüklendi. İlk 5 satır:\n{df_song.head()}\n")

        # 2. Ana Veri Setini Seçme ve Birleştirme
        # Eurovision kazananını tahmin etmek için en uygun veri 'song_data.csv' gibi duruyor.
        # Bu dosyada 'year', 'country' ve puan bilgileri ('semi_total_points', 'total_points' varsa) mevcut.
        df_main = df_song.copy()

        # Eksik puanları doldurma: Eğer final_total_points varsa onu tercih et, yoksa semi_total_points kullan.
        # Veya sadece final_total_points olanları filtrele.
        # Senin çıktında 'semi_total_points' var, 'final_total_points' NAN (boş) görünüyor.
        # Eğer asıl amaç finaldeki kazananı bulmaksa, 'final_total_points' sütununa ihtiyacımız var.
        # Veri setinizde 'final_total_points' sütunu var mı? Varsa, onun ismini kullanmalıyız.
        # Şu anki çıktıya göre, 'final_total_points' sütunu ya yok ya da çoğu NaN.
        # En kesin puan sütununu bulmak için veri setinizin tamamını incelemeniz gerekebilir.

        # GEÇİCİ ÇÖZÜM: Varsayalım ki 'final_total_points' sütunu gerçekten var ve kullanmak istiyoruz.
        # Eğer bu sütun çoğu NaN ise (ki senin çıktında öyle), o zaman NaN değerleri olan satırları düşürelim
        # veya farklı bir strateji izleyelim (örn: sadece yarı final puanlarını kullan).

        # Eğer final puanları yoksa veya eksikse, yarı final puanlarını kullanabiliriz.
        # Ancak Eurovision kazananı final puanlarına göre belirlenir.
        # Eğer final puanlarınız song_data.csv'de 'total_points' veya 'final_total_points' ise, onu kullanın.
        
        # Bu örnekte, 'final_total_points' sütununun var olduğunu ve final sonuçlarını temsil ettiğini varsayalım.
        # Eğer yoksa veya eksikse, kodu aşağıdaki gibi ayarlayabiliriz:

        # Eğer 'final_total_points' sütunu varsa ve o gerçek final puanlarını içeriyorsa:
        if 'final_total_points' in df_main.columns:
            # Sadece finaldeki katılımcıları ve puanları içeren satırları al
            df_main = df_main.dropna(subset=['final_total_points', 'country', 'year'])
            df_main['Total_Points'] = df_main['final_total_points']
            print("Model, 'final_total_points' sütununu kullanarak eğitilecektir.")
        # Eğer 'semi_total_points' sadece varsa veya final puanları çok eksikse:
        elif 'semi_total_points' in df_main.columns:
            df_main = df_main.dropna(subset=['semi_total_points', 'country', 'year'])
            df_main['Total_Points'] = df_main['semi_total_points']
            print("Uyarı: 'final_total_points' sütunu bulunamadı veya çok eksik. Model, 'semi_total_points' sütununu kullanarak eğitilecektir. Bu, final kazananını tahmin etmek için daha az doğru olabilir.")
        else:
            raise ValueError("Şarkı veri setinde (song_data.csv) puanları içeren bir sütun ('final_total_points' veya 'semi_total_points') bulunamadı. Lütfen veri setinizi kontrol edin.")


        # Gerekli sütunların varlığını kontrol et
        # Artık 'song_data.csv'deki sütun isimlerine göre kontrol yapıyoruz
        required_columns = ['year', 'country', 'Total_Points'] # 'Total_Points' az önce oluşturuldu
        if not all(col in df_main.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df_main.columns]
            raise ValueError(f"Şarkı veri setinde (song_data.csv) eksik sütunlar var: {missing}. "
                             f"Lütfen veri setinizdeki sütun isimlerini kontrol edin ve 'required_columns' listesini güncelleyin.")

        # İsteğe bağlı: Diğer veri setlerinden özellik ekleme
        # Örneğin, 'country_data.csv'deki bölge bilgisi kullanılabilir.
        # df_main = pd.merge(df_main, df_country[['country', 'region']], on='country', how='left')
        # df_main = pd.get_dummies(df_main, columns=['region'], prefix='Region') # Bölgeyi one-hot encode et

        # 3. Her yılın kazananını belirle
        # NOT: Eğer song_data.csv'de her yılın birden fazla final katılımcısı varsa,
        # bu kod en yüksek puanı alan katılımcıyı o yılın kazananı olarak belirler.
        # Eğer veri setiniz sadece kazananları içeriyorsa bu adıma gerek olmayabilir.
        winners_df = df_main.loc[df_main.groupby('year')['Total_Points'].idxmax()]
        print(f"Her yılın kazananları (ilk 5):\n{winners_df.head()}\n")

        # 4. Özellik Mühendisliği
        # 'country' sütununu sayısal verilere dönüştür (One-Hot Encoding)
        country_dummies = pd.get_dummies(winners_df['country'], prefix='Country')
        X = pd.concat([winners_df[['year']], country_dummies], axis=1) # 'year' sütununu kullan

        # Hedef değişken: Kazanan ülke (Label Encoding)
        le = LabelEncoder()
        y_encoded = le.fit_transform(winners_df['country']) # 'country' sütununu kullan
        country_names = le.classes_ # Tahminleri geri dönüştürmek için ülke isimlerini sakla

        # 5. Modeli Eğit
        X_train, y_train = X, y_encoded
        print(f"Eğitim veri boyutu: {X_train.shape}, Etiket boyutu: {y_train.shape}\n")

        # Yeterli veri olup olmadığını kontrol et
        if X_train.empty or len(y_train) == 0:
            raise ValueError("Modeli eğitmek için yeterli veri bulunamadı. Lütfen 'song_data.csv' dosyasını kontrol edin ve 'year', 'country', 'final_total_points' (veya 'semi_total_points') sütunlarının doğru ve dolu olduğundan emin olun.")

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print("Model başarıyla eğitildi.\n")

        # 6. 2025 İçin Tahmin Yap
        predict_2025_df = pd.DataFrame()
        predict_2025_df['year'] = [2025] * len(country_names) # 'year' sütununu kullan

        for col in country_dummies.columns:
            predict_2025_df[col] = 0

        temp_list = []
        for i, country_name_encoded in enumerate(country_names):
            temp_row = {'year': 2025}
            for col in country_dummies.columns:
                temp_row[col] = 0
            
            country_dummy_col = f'Country_{country_name_encoded}'
            if country_dummy_col in country_dummies.columns:
                temp_row[country_dummy_col] = 1
            temp_list.append(temp_row)
        
        if not temp_list:
            return "Tahmin yapmak için yeterli ülke verisi bulunamadı. Lütfen veri setinizi ve sütun eşleşmelerini kontrol edin."

        predict_2025_df = pd.DataFrame(temp_list)

        # Eğitim verisindeki sütunlar ile tahmin verisindeki sütunları hizala
        missing_cols = set(X_train.columns) - set(predict_2025_df.columns)
        for c in missing_cols:
            predict_2025_df[c] = 0
        predict_2025_df = predict_2025_df[X_train.columns]

        # Tahminleri yap
        probabilities = model.predict_proba(predict_2025_df)

        predicted_country_indices = np.argmax(probabilities, axis=1)

        results_df = pd.DataFrame({
            'Predicted_Country_Index': predicted_country_indices,
            'Probability': np.max(probabilities, axis=1)
        })

        results_df['Predicted_Country'] = results_df['Predicted_Country_Index'].apply(lambda x: le.inverse_transform([x])[0])
        
        final_probabilities = results_df.groupby('Predicted_Country')['Probability'].sum().sort_values(ascending=False)

        if not final_probabilities.empty:
            predicted_winner = final_probabilities.index[0]
            max_probability = final_probabilities.iloc[0]
            return f"2025 Eurovision Tahmini Kazananı: **{predicted_winner}** (Olasılık: {max_probability:.2f})"
        else:
            return "Tahmin yapılamadı. Modelin eğitildiği veri seti yetersiz veya formatlama hatası var."

    except FileNotFoundError as e:
        return f"Hata: Bir dosya bulunamadı. Lütfen dosya adlarını ve konumlarını kontrol edin. Hata: {e}"
    except KeyError as e:
        return f"Hata: Veri setinde beklenen sütun bulunamadı: {e}. Lütfen veri setlerinizdeki sütun isimlerini kontrol edin ve kodu buna göre güncelleyin."
    except ValueError as e:
        return f"Hata: {e}"
    except Exception as e:
        return f"Beklenmeyen bir hata oluştu: {e}"

# --- Kodu Çalıştırma ---
if __name__ == "__main__":
    contest_data_file = "contest_data.csv"
    country_data_file = "country_data.csv"
    song_data_file = "song_data.csv"

    prediction_result = predict_eurovision_winner_2025_multi_csv(
        contest_data_file, country_data_file, song_data_file
    )
    print("\n--- Tahmin Sonucu ---")
    print(prediction_result)