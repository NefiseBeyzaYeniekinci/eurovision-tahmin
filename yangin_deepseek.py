import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

def load_data():
    try:
        print("Dosyalar yükleniyor...")
        contest_data = pd.read_csv("contest_data.csv", encoding='iso-8859-1')
        country_data = pd.read_csv("country_data.csv", encoding='iso-8859-1')
        song_data = pd.read_csv("song_data.csv", encoding='iso-8859-1')
        print("Dosyalar başarıyla yüklendi")
        
        # Sütun isimlerini standartlaştırma
        song_data.rename(columns={'artist_name': 'artist'}, inplace=True)
        
        return contest_data, country_data, song_data
    except Exception as e:
        print(f"Yükleme hatası: {str(e)}")
        return None, None, None

def preprocess_and_merge(contest_data, country_data, song_data):
    try:
        print("\nVeriler birleştiriliyor...")
        
        # Ana veri olarak song_data'yı kullanacağız
        main_data = song_data.copy()
        
        # Country_data'dan bölge bilgisini ekleyelim
        main_data = pd.merge(main_data, country_data, on='country', how='left')
        
        # Eksik veri kontrolü
        print("\nEksik veri analizi:")
        print(main_data.isnull().sum())
        
        # Temizlik
        main_data = main_data.dropna(subset=['final_total_points', 'country', 'year'])
        
        # Kategorik kodlama
        le_country = LabelEncoder()
        main_data['country_encoded'] = le_country.fit_transform(main_data['country'])
        
        # Sayısal sütunları işleme
        main_data['year'] = pd.to_numeric(main_data['year'], errors='coerce')
        main_data['final_total_points'] = pd.to_numeric(main_data['final_total_points'], errors='coerce')
        main_data = main_data.dropna(subset=['year', 'final_total_points'])
        
        print(f"\nİşlenmiş veri boyutu: {main_data.shape}")
        return main_data, le_country
        
    except Exception as e:
        print(f"\nBirleştirme hatası: {str(e)}")
        return None, None

def create_features(data):
    try:
        print("\nÖzellikler oluşturuluyor...")
        
        # Ülke istatistikleri
        country_stats = data.groupby('country_encoded').agg({
            'final_total_points': ['mean', 'max', 'min', 'count'],
            'year': ['nunique']
        }).reset_index()
        
        country_stats.columns = [
            'country_encoded', 'mean_points', 'max_points', 
            'min_points', 'participations', 'years_participated'
        ]
        
        data = pd.merge(data, country_stats, on='country_encoded', how='left')
        
        # Müzik özelliklerini işleme
        audio_features = ['BPM', 'energy', 'danceability', 'happiness', 
                         'loudness', 'acousticness', 'instrumentalness']
        
        for feature in audio_features:
            if feature in data.columns:
                data[feature] = pd.to_numeric(data[feature], errors='coerce')
                data[feature].fillna(data[feature].mean(), inplace=True)
            else:
                print(f"Uyarı: {feature} sütunu eksik, 0 ile dolduruluyor")
                data[feature] = 0
        
        # Dil özelliği
        if 'language' in data.columns:
            le_language = LabelEncoder()
            data['language_encoded'] = le_language.fit_transform(data['language'].fillna('Unknown'))
        else:
            data['language_encoded'] = 0
        
        # Yıl istatistikleri
        year_stats = data.groupby('year').agg({
            'final_total_points': ['mean', 'max']
        }).reset_index()
        year_stats.columns = ['year', 'year_mean_points', 'year_max_points']
        data = pd.merge(data, year_stats, on='year', how='left')
        
        # Özel özellikler
        data['points_ratio'] = data['final_total_points'] / (data['max_points'] + 1e-6)
        data['points_vs_year_mean'] = data['final_total_points'] / (data['year_mean_points'] + 1e-6)
        data['consistency'] = (data['max_points'] - data['min_points']) / (data['mean_points'] + 1e-6)
        
        print("Özellikler başarıyla oluşturuldu!")
        return data
        
    except Exception as e:
        print(f"\nÖzellik oluşturma hatası: {str(e)}")
        return None

def train_model(data):
    try:
        print("\nModel eğitiliyor...")
        
        # Özellik seçimi
        features = [
            'mean_points', 'max_points', 'min_points', 'participations',
            'points_ratio', 'points_vs_year_mean', 'consistency',
            'BPM', 'energy', 'danceability', 'happiness',
            'loudness', 'acousticness', 'instrumentalness',
            'language_encoded'
        ]
        
        # Etiket oluşturma (finalde ilk 3'e girenler)
        data['is_high_rank'] = (data['final_place'] <= 3).astype(int)
        
        # Veriyi bölme
        X = data[features]
        y = data['is_high_rank']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Model oluşturma
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Değerlendirme
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Doğruluğu: {accuracy:.2%}")
        
        # Özellik önemleri
        print("\nÖzellik Önemleri:")
        feat_importances = pd.Series(model.feature_importances_, index=features)
        print(feat_importances.sort_values(ascending=False))
        
        return model, features
        
    except Exception as e:
        print(f"\nModel eğitim hatası: {str(e)}")
        return None, None

def predict_2025(model, features, data, label_encoder):
    try:
        print("\n2025 tahmini yapılıyor...")
        
        # Son yılın verilerini al
        last_year = data['year'].max()
        last_year_data = data[data['year'] == last_year].copy()
        
        # Her ülke için ortalama özellikler
        agg_dict = {
            'mean_points': 'first',
            'max_points': 'first',
            'min_points': 'first',
            'participations': 'first',
            'points_ratio': 'mean',
            'points_vs_year_mean': 'mean',
            'consistency': 'first',
            'BPM': 'mean',
            'energy': 'mean',
            'danceability': 'mean',
            'happiness': 'mean',
            'loudness': 'mean',
            'acousticness': 'mean',
            'instrumentalness': 'mean',
            'language_encoded': lambda x: x.value_counts().index[0] if len(x) > 0 else 0
        }
        
        country_features = last_year_data.groupby('country_encoded').agg(agg_dict).reset_index()
        
        # Eksik özellikleri kontrol et
        for feature in features:
            if feature not in country_features.columns:
                print(f"Uyarı: {feature} sütunu eksik, 0 ile dolduruluyor")
                country_features[feature] = 0
        
        # Modelin beklediği sırayla özellikleri seç
        X_2025 = country_features[features]
        
        # Olasılık tahmini
        predictions = model.predict_proba(X_2025)[:, 1]
        
        # Sonuçları hazırla
        results = pd.DataFrame({
            'country_encoded': country_features['country_encoded'],
            'win_probability': predictions
        })
        
        results['country'] = label_encoder.inverse_transform(results['country_encoded'])
        results = results.sort_values('win_probability', ascending=False)
        
        return results
        
    except Exception as e:
        print(f"\nTahmin hatası: {str(e)}")
        return None

def visualize_results(results):
    plt.figure(figsize=(14, 10))
    sns.set_style("whitegrid")
    
    top20 = results.head(20)
    palette = sns.color_palette("viridis", len(top20))
    
    ax = sns.barplot(data=top20, x='win_probability', y='country', palette=palette)
    
    plt.title('2025 Eurovision Kazanma Olasılıkları (Top 20 Ülke)', fontsize=16, pad=20)
    plt.xlabel('Kazanma Olasılığı', fontsize=12)
    plt.ylabel('Ülke', fontsize=12)
    
    # Olasılık değerlerini ekle
    for i, prob in enumerate(top20['win_probability']):
        ax.text(prob + 0.01, i, f"{prob:.2%}", va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('eurovision_2025_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("Eurovision 2025 Tahmin Modeli Başlatılıyor...\n")
    
    # Verileri yükle
    contest_data, country_data, song_data = load_data()
    if contest_data is None or country_data is None or song_data is None:
        return
    
    # Verileri birleştir ve ön işle
    merged_data, label_encoder = preprocess_and_merge(contest_data, country_data, song_data)
    if merged_data is None:
        return
    
    # Özellikleri oluştur
    final_data = create_features(merged_data)
    if final_data is None:
        return
    
    # Modeli eğit
    model, features = train_model(final_data)
    if model is None:
        return
    
    # 2025 tahmini yap
    predictions = predict_2025(model, features, final_data, label_encoder)
    if predictions is None:
        return
    
    # Sonuçları göster
    print("\n2025 Eurovision Kazanma Olasılıkları (Top 20):")
    print(predictions[['country', 'win_probability']].head(20).to_string(index=False))
    
    # Görselleştirme
    visualize_results(predictions)

if __name__ == "__main__":
    main()