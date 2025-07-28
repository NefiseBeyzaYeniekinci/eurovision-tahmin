import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class EurovisionPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = None
        self.feature_names = []
        
    def load_and_prepare_data(self, contest_path, country_path, song_path):
        """
        Eurovision veri setlerini yükler ve birleştirir
        """
        print("Veri setleri yükleniyor...")
        
        # 3 dosyayı ayrı ayrı yükle
        try:
            contest_df = pd.read_csv(contest_path, encoding='utf-8')
            print(f"Contest data yüklendi: {len(contest_df)} kayıt")
            print(f"Contest sütunları: {contest_df.columns.tolist()}")
        except Exception as e:
            print(f"Contest data yüklenemedi: {e}")
            contest_df = pd.DataFrame()
        
        try:
            country_df = pd.read_csv(country_path, encoding='utf-8')
            print(f"Country data yüklendi: {len(country_df)} kayıt")
            print(f"Country sütunları: {country_df.columns.tolist()}")
        except Exception as e:
            print(f"Country data yüklenemedi: {e}")
            country_df = pd.DataFrame()
        
        try:
            song_df = pd.read_csv(song_path, encoding='utf-8')
            print(f"Song data yüklendi: {len(song_df)} kayıt")
            print(f"Song sütunları: {song_df.columns.tolist()}")
        except UnicodeDecodeError:
            # Farklı encoding'leri dene
            encodings = ['latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
            song_df = pd.DataFrame()
            for encoding in encodings:
                try:
                    song_df = pd.read_csv(song_path, encoding=encoding)
                    print(f"Song data yüklendi ({encoding}): {len(song_df)} kayıt")
                    print(f"Song sütunları: {song_df.columns.tolist()}")
                    break
                except:
                    continue
            if song_df.empty:
                print("Song data hiçbir encoding ile yüklenemedi")
        except Exception as e:
            print(f"Song data yüklenemedi: {e}")
            song_df = pd.DataFrame()
        
        # Veri setlerini birleştir
        df = self.merge_datasets(contest_df, country_df, song_df)
        
        print(f"\nBirleştirilmiş veri: {len(df)} kayıt")
        print(f"Final sütunlar: {df.columns.tolist()}")
        
        # Eksik değerleri kontrol et
        print("\nEksik değerler:")
        missing_data = df.isnull().sum()
        print(missing_data[missing_data > 0])
        
        return df
    
    def merge_datasets(self, contest_df, country_df, song_df):
        """
        3 veri setini akıllıca birleştirir
        """
        print("Veri setleri birleştiriliyor...")
        
        # Ana veri seti olarak en büyük olanı seç
        main_df = contest_df.copy() if len(contest_df) >= len(song_df) else song_df.copy()
        
        # Ortak sütunları bul (birleştirme anahtarları)
        common_keys = []
        
        # Yaygın birleştirme anahtarlarını kontrol et
        possible_keys = [
            ['year', 'country'], ['year', 'country_code'], 
            ['contest_id', 'country'], ['id', 'country'],
            ['year', 'participant'], ['edition', 'country']
        ]
        
        for keys in possible_keys:
            if all(key in main_df.columns for key in keys):
                if len(country_df) > 0 and all(key in country_df.columns for key in keys):
                    common_keys = keys
                    break
                elif len(song_df) > 0 and all(key in song_df.columns for key in keys):
                    common_keys = keys
                    break
        
        # Veri setlerini birleştir
        merged_df = main_df.copy()
        
        if len(common_keys) > 0:
            print(f"Birleştirme anahtarları: {common_keys}")
            
            # Country data ile birleştir
            if len(country_df) > 0:
                try:
                    merged_df = merged_df.merge(country_df, on=common_keys, how='left', suffixes=('', '_country'))
                    print("Country data birleştirildi")
                except Exception as e:
                    print(f"Country data birleştirme hatası: {e}")
            
            # Song data ile birleştir
            if len(song_df) > 0 and not song_df.equals(main_df):
                try:
                    merged_df = merged_df.merge(song_df, on=common_keys, how='left', suffixes=('', '_song'))
                    print("Song data birleştirildi")
                except Exception as e:
                    print(f"Song data birleştirme hatası: {e}")
        else:
            print("Ortak birleştirme anahtarı bulunamadı, veri setleri ayrı ayrı kullanılacak")
            
            # Basit birleştirme - sadece sütunları ekle
            for df_name, df in [('country', country_df), ('song', song_df)]:
                if len(df) > 0 and not df.equals(main_df):
                    for col in df.columns:
                        if col not in merged_df.columns:
                            # İlk satırın değerini al (genel bilgi olduğunu varsay)
                            merged_df[f'{col}_{df_name}'] = df[col].iloc[0] if len(df) > 0 else None
        
        return merged_df
    
    def feature_engineering(self, df):
        """
        Özellik mühendisliği - Eurovision verilerine uygun özellikler çıkarır
        """
        print("Özellik mühendisliği yapılıyor...")
        
        # Veri setinizin sütun isimlerine göre bu kısmı güncellemeniz gerekebilir
        features = df.copy()
        
        # Veri setinizin yapısına göre özellikler oluştur - Song Data özelikleri
        print(f"Mevcut sütunlar: {list(features.columns)}")
        
        # Tüm '-' değerlerini NaN ile değiştir
        features = features.replace('-', np.nan)
        features = features.replace('', np.nan)
        
        # Müzik özellikleri (Spotify benzeri)
        music_features = ['BPM', 'energy', 'danceability', 'happiness', 'loudness', 
                         'acousticness', 'instrumentalness', 'liveness', 'speechiness']
        for feature in music_features:
            if feature in features.columns:
                # Eksik değerleri ortalama ile doldur
                features[feature] = pd.to_numeric(features[feature], errors='coerce')
                features[feature] = features[feature].fillna(features[feature].median())
        
        # Çıkış pozisyonu özellikleri
        if 'semi_draw_position' in features.columns:
            features['semi_draw_position'] = pd.to_numeric(features['semi_draw_position'], errors='coerce').fillna(0)
            features['early_semi_position'] = (features['semi_draw_position'] <= 8).astype(int)
            features['late_semi_position'] = (features['semi_draw_position'] >= 15).astype(int)
            
        if 'final_draw_position' in features.columns:
            features['final_draw_position'] = pd.to_numeric(features['final_draw_position'], errors='coerce').fillna(0)
            features['early_final_position'] = (features['final_draw_position'] <= 13).astype(int)
            features['late_final_position'] = (features['final_draw_position'] >= 20).astype(int)
        
        # Performans özellikleri
        if 'age' in features.columns:
            features['age'] = pd.to_numeric(features['age'], errors='coerce').fillna(25)
            features['young_performer'] = (features['age'] <= 25).astype(int)
            features['experienced_performer'] = (features['age'] >= 35).astype(int)
        
        # Direct qualifier (Big 5 + host)
        if 'direct_qualifier_10' in features.columns:
            features['direct_qualifier_10'] = pd.to_numeric(features['direct_qualifier_10'], errors='coerce')
            features['is_direct_qualifier'] = features['direct_qualifier_10'].fillna(0).astype(int)
        
        # Host advantage
        if 'host_10' in features.columns:
            features['host_10'] = pd.to_numeric(features['host_10'], errors='coerce')
            features['is_host_country'] = features['host_10'].fillna(0).astype(int)
        
        # Backing support - sayısal değerleri güvenli şekilde dönüştür
        backing_features = ['backing_dancers', 'backing_singers', 'backing_instruments']
        for feature in backing_features:
            if feature in features.columns:
                features[f'{feature}_count'] = pd.to_numeric(features[feature], errors='coerce').fillna(0)
        
        # Final/Semi performance
        if 'final_total_points' in features.columns:
            features['final_total_points'] = pd.to_numeric(features['final_total_points'], errors='coerce')
            features['made_to_final'] = (~features['final_total_points'].isna()).astype(int)
            features['final_total_points_filled'] = features['final_total_points'].fillna(0)
        
        # Final place
        if 'final_place' in features.columns:
            features['final_place'] = pd.to_numeric(features['final_place'], errors='coerce')
        
        # Semi place
        if 'semi_place' in features.columns:
            features['semi_place'] = pd.to_numeric(features['semi_place'], errors='coerce')
        
        # Qualified bilgisi
        if 'qualified_10' in features.columns:
            features['qualified_10'] = pd.to_numeric(features['qualified_10'], errors='coerce')
            features['qualified'] = features['qualified_10'].fillna(0).astype(int)
        
        # Jury ve televote puanları
        point_columns = ['final_televote_points', 'final_jury_points', 'semi_televote_points', 'semi_jury_points']
        for col in point_columns:
            if col in features.columns:
                features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
        
        # Vote sayıları
        vote_columns = ['final_televote_votes', 'final_jury_votes']
        for col in vote_columns:
            if col in features.columns:
                features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)
        
        # Key change ve diğer binary özellikler
        binary_features = ['key_change_10', 'instrument_10', 'favourite_10']
        for feature in binary_features:
            if feature in features.columns:
                features[feature] = pd.to_numeric(features[feature], errors='coerce').fillna(0).astype(int)
        
        # Language effect
        if 'language' in features.columns:
            english_songs = features['language'].str.contains('English', na=False, case=False)
            features['english_song'] = english_songs.astype(int)
            features['native_language'] = (~english_songs).astype(int)
        
        # Kategorik değişkenleri encode et
        categorical_columns = features.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col not in ['winner', 'place']:  # Hedef değişkenleri encode etme
                le = LabelEncoder()
                # NaN değerleri 'Unknown' ile değiştir
                col_data = features[col].fillna('Unknown').astype(str)
                features[col + '_encoded'] = le.fit_transform(col_data)
                self.label_encoders[col] = le
        
        # Yıl bazlı özellikler
        if 'year' in features.columns:
            features['year'] = pd.to_numeric(features['year'], errors='coerce')
            features['years_since_first'] = features['year'] - features['year'].min()
            features['decade'] = (features['year'] // 10) * 10
            features['modern_era'] = (features['year'] >= 2010).astype(int)
            features['recent_contest'] = (features['year'] >= 2020).astype(int)
        
        # Ülke bazlı başarı geçmişi
        if 'country' in features.columns and 'final_place' in features.columns:
            # final_place'i sayısal hale getir
            features['final_place_numeric'] = pd.to_numeric(features['final_place'], errors='coerce')
            
            country_stats = features.groupby('country').agg({
                'final_place_numeric': ['mean', 'count', 'std'],
                'final_total_points': 'mean',
                'made_to_final': 'mean'
            }).reset_index()
            country_stats.columns = ['country', 'avg_final_place', 'participation_count', 
                                   'place_std', 'avg_points', 'final_success_rate']
            # NaN değerleri doldur
            country_stats = country_stats.fillna(0)
            features = features.merge(country_stats, on='country', how='left')
        
        # Müzik tarzı analizi
        if 'style' in features.columns:
            # Popüler tarzları tanımla
            popular_styles = ['Pop', 'Dance', 'Folk', 'Rock']
            for style in popular_styles:
                features[f'style_{style.lower()}'] = features['style'].str.contains(style, na=False, case=False).astype(int)
        
        # Gender/Performer analysis
        if 'gender' in features.columns:
            features['female_performer'] = features['gender'].str.contains('Female', na=False, case=False).astype(int)
            features['male_performer'] = features['gender'].str.contains('Male', na=False, case=False).astype(int)
            features['group_performer'] = features['gender'].str.contains('Group', na=False, case=False).astype(int)
        
        # Semi final bilgisi
        if 'semi_final' in features.columns:
            features['semi_final'] = pd.to_numeric(features['semi_final'], errors='coerce').fillna(0)
            features['in_semi_1'] = (features['semi_final'] == 1).astype(int)
            features['in_semi_2'] = (features['semi_final'] == 2).astype(int)
        
        print(f"Özellik mühendisliği tamamlandı. Toplam özellik sayısı: {len(features.columns)}")
        return features
    
    def prepare_target_variable(self, df):
        """
        Hedef değişkeni hazırlar (kazanan/kaybeden veya sıralama)
        """
        print("Hedef değişken hazırlanıyor...")
        
        # Song data için hedef değişken
        if 'final_place' in df.columns:
            print("Final place kullanılarak hedef değişken oluşturuluyor...")
            # Final'e katılanları ve iyi derece yapanları başarılı kabul et
            target = pd.to_numeric(df['final_place'], errors='coerce')
            # NaN değerler yarı finalde elenenler (başarısız)
            target = target.fillna(99)  # Elimine edilenler için yüksek değer
            # İlk 10'a girenleri başarılı kabul et
            return (target <= 10).astype(int)
            
        elif 'final_total_points' in df.columns:
            print("Final total points kullanılarak hedef değişken oluşturuluyor...")
            points = pd.to_numeric(df['final_total_points'], errors='coerce').fillna(0)
            threshold = points.quantile(0.7)  # Üst %30
            return (points >= threshold).astype(int)
            
        elif 'qualified_10' in df.columns:
            print("Qualified bilgisi kullanılarak hedef değişken oluşturuluyor...")
            return pd.to_numeric(df['qualified_10'], errors='coerce').fillna(0).astype(int)
            
        elif 'winner' in df.columns:
            return df['winner'].astype(int)
        elif 'place' in df.columns:
            return (df['place'] <= 3).astype(int)
        else:
            # Varsayılan hedef değişken oluştur (yıl bazlı)
            print("Varsayılan hedef değişken oluşturuluyor...")
            if 'year' in df.columns:
                # Son yılları daha başarılı kabul et
                latest_years = df['year'].nlargest(int(len(df) * 0.3))
                target = df['year'].isin(latest_years).astype(int)
                return target
            else:
                # Son çare: rastgele ama dengeli hedef
                np.random.seed(42)
                target = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
                print("Rastgele hedef değişken oluşturuldu (eğitim amaçlı)")
                return target
    
    def train_model(self, df):
        """
        Modeli eğitir
        """
        print("Model eğitimi başlıyor...")
        
        # Özellik mühendisliği
        features_df = self.feature_engineering(df)
        
        # Hedef değişkeni hazırla
        y = self.prepare_target_variable(df)
        
        # Sadece sayısal sütunları al
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        X = features_df[numeric_columns]
        
        # Eksik değerleri doldur
        X = X.fillna(X.mean())
        
        # Özellik seçimi
        self.feature_selector = SelectKBest(f_classif, k=min(15, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Seçilen özellik isimlerini sakla
        selected_features = X.columns[self.feature_selector.get_support()]
        self.feature_names = selected_features.tolist()
        print(f"Seçilen özellikler: {self.feature_names}")
        
        # Veriyi böl
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Veriyi ölçeklendir
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Ensemble model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        # Modelleri eğit
        rf_model.fit(X_train_scaled, y_train)
        gb_model.fit(X_train_scaled, y_train)
        
        # Tahminler
        rf_pred = rf_model.predict(X_test_scaled)
        gb_pred = gb_model.predict(X_test_scaled)
        
        # Ensemble tahmin (ağırlıklı ortalama)
        ensemble_pred = (rf_pred + gb_pred) / 2
        ensemble_pred = (ensemble_pred > 0.5).astype(int)
        
        # En iyi modeli seç
        rf_acc = accuracy_score(y_test, rf_pred)
        gb_acc = accuracy_score(y_test, gb_pred)
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        
        print(f"Random Forest Doğruluk: {rf_acc:.3f}")
        print(f"Gradient Boosting Doğruluk: {gb_acc:.3f}")
        print(f"Ensemble Doğruluk: {ensemble_acc:.3f}")
        
        # En iyi modeli kaydet
        if ensemble_acc >= max(rf_acc, gb_acc):
            self.model = [rf_model, gb_model]
            print("Ensemble model seçildi")
        elif rf_acc > gb_acc:
            self.model = rf_model
            print("Random Forest model seçildi")
        else:
            self.model = gb_model
            print("Gradient Boosting model seçildi")
        
        # Model performansını görselleştir
        self.plot_model_performance(y_test, ensemble_pred)
        
        return X_test_scaled, y_test
    
    def plot_model_performance(self, y_true, y_pred):
        """
        Model performansını görselleştirir
        """
        plt.figure(figsize=(12, 4))
        
        # Confusion Matrix
        plt.subplot(1, 2, 1)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Gerçek')
        plt.xlabel('Tahmin')
        
        # Sınıf dağılımı
        plt.subplot(1, 2, 2)
        class_counts = Counter(y_pred)
        plt.bar(class_counts.keys(), class_counts.values())
        plt.title('Tahmin Sınıf Dağılımı')
        plt.xlabel('Sınıf')
        plt.ylabel('Sayı')
        
        plt.tight_layout()
        plt.show()
        
        # Detaylı rapor
        print("\nSınıflandırma Raporu:")
        print(classification_report(y_true, y_pred))
    
    def predict_2025_winners(self, df):
        """
        2025 kazananlarını tahmin eder
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi!")
        
        print("2025 Eurovision kazananları tahmin ediliyor...")
        
        # 2025 için gerçekçi ülkeler (contest data'ya göre)
        countries_2025 = [
            'Germany', 'France', 'Italy', 'Spain', 'United Kingdom',  # Big 5
            'Sweden', 'Norway', 'Netherlands', 'Australia', 'Ukraine',
            'Poland', 'Finland', 'Denmark', 'Belgium', 'Switzerland',
            'Austria', 'Portugal', 'Greece', 'Cyprus', 'Malta'
        ]
        
        # Host country'yi veri setinden al
        if len(df) > 0 and 'host' in df.columns:
            latest_host = df.loc[df['year'].idxmax(), 'host'] if 'year' in df.columns else df['host'].iloc[-1]
        else:
            latest_host = 'Switzerland'  # Varsayılan
        
        # 2025 tahmin verisi oluştur
        prediction_data = []
        for i, country in enumerate(countries_2025):
            row = {
                'year': 2025,
                'country': country,
                'semi_final': 1 if i < 10 else 2,  # Semi final dağılımı
                'semi_draw_position': (i % 18) + 1,
                'final_draw_position': (i % 26) + 1,
                'age': 25,  # Ortalama yaş
                'direct_qualifier_10': 1 if country in ['Germany', 'France', 'Italy', 'Spain', 'United Kingdom'] else 0,
                'host_10': 1 if country == latest_host else 0,
                'gender': 'Mixed',
                'language': 'English',
                'style': 'Pop',
                'BPM': 120,
                'energy': 0.7,
                'danceability': 0.6,
                'happiness': 0.5,
                'loudness': -5,
                'acousticness': 0.3,
                'instrumentalness': 0.1,
                'liveness': 0.2,
                'speechiness': 0.1,
                'backing_dancers': 2,
                'backing_singers': 1,
                'backing_instruments': 3,
                'key_change_10': 0,
                'instrument_10': 1,
                'favourite_10': 0
            }
            prediction_data.append(row)
        
        pred_df = pd.DataFrame(prediction_data)
        
        # Özellik mühendisliği uygula
        features_df = self.feature_engineering(pred_df)
        
        # Sayısal sütunları al ve eksik değerleri doldur
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        X_pred = features_df[numeric_columns].fillna(0)
        
        # Özellik seçimi uygula
        if self.feature_selector:
            # Eğitimde kullanılan özellik sayısına göre ayarla
            n_features = len(self.feature_names)
            if X_pred.shape[1] >= n_features:
                X_pred_selected = self.feature_selector.transform(X_pred)
            else:
                # Eksik özellikler varsa sıfırla doldur
                missing_features = n_features - X_pred.shape[1]
                X_pred_padded = np.column_stack([X_pred.values, np.zeros((X_pred.shape[0], missing_features))])
                X_pred_selected = X_pred_padded
        else:
            X_pred_selected = X_pred
        
        # Ölçeklendir
        X_pred_scaled = self.scaler.transform(X_pred_selected)
        
        # Tahmin yap
        if isinstance(self.model, list):  # Ensemble
            pred1 = self.model[0].predict_proba(X_pred_scaled)[:, 1]
            pred2 = self.model[1].predict_proba(X_pred_scaled)[:, 1]
            probabilities = (pred1 + pred2) / 2
        else:
            probabilities = self.model.predict_proba(X_pred_scaled)[:, 1]
        
        # Sonuçları düzenle
        results = pd.DataFrame({
            'Ülke': countries_2025,
            'Kazanma_Olasılığı': probabilities,
            'Tahmin_Sırası': range(1, len(countries_2025) + 1)
        })
        
        results = results.sort_values('Kazanma_Olasılığı', ascending=False)
        results['Sıralama'] = range(1, len(results) + 1)
        
        return results
    
    def get_country_region(self, country):
        """
        Ülke bölgesi döndürür
        """
        regions = {
            'Germany': 'Western Europe', 'France': 'Western Europe', 'Italy': 'Southern Europe',
            'Spain': 'Southern Europe', 'United Kingdom': 'Western Europe', 'Sweden': 'Northern Europe',
            'Norway': 'Northern Europe', 'Netherlands': 'Western Europe', 'Australia': 'Oceania',
            'Ukraine': 'Eastern Europe', 'Poland': 'Eastern Europe', 'Finland': 'Northern Europe',
            'Denmark': 'Northern Europe', 'Belgium': 'Western Europe', 'Switzerland': 'Western Europe',
            'Austria': 'Western Europe', 'Portugal': 'Southern Europe', 'Greece': 'Southern Europe',
            'Cyprus': 'Southern Europe', 'Malta': 'Southern Europe'
        }
        return regions.get(country, 'Unknown')

def main():
    """
    Ana fonksiyon - Eurovision tahmin sistemini çalıştırır
    """
    print("=== Eurovision 2025 Kazananı Tahmin Sistemi ===\n")
    
    # Veri dosyası yolları
    contest_file = "contest_data.csv"
    country_file = "country_data.csv" 
    song_file = "song_data.csv"
    
    try:
        # Tahmin sistemini başlat
        predictor = EurovisionPredictor()
        
        # Veriyi yükle
        df = predictor.load_and_prepare_data(contest_file, country_file, song_file)
        
        # Modeli eğit
        X_test, y_test = predictor.train_model(df)
        
        # 2025 tahminlerini yap
        predictions = predictor.predict_2025_winners(df)
        
        print("\n=== 2025 EUROVISION TAHMİNLERİ ===")
        print(predictions.to_string(index=False))
        
        # En güçlü adayları göster
        print(f"\n🏆 1. FAVORİ: {predictions.iloc[0]['Ülke']} (Olasılık: {predictions.iloc[0]['Kazanma_Olasılığı']:.3f})")
        print(f"🥈 2. FAVORİ: {predictions.iloc[1]['Ülke']} (Olasılık: {predictions.iloc[1]['Kazanma_Olasılığı']:.3f})")
        print(f"🥉 3. FAVORİ: {predictions.iloc[2]['Ülke']} (Olasılık: {predictions.iloc[2]['Kazanma_Olasılığı']:.3f})")
        
        # Sonuçları kaydet
        predictions.to_csv('eurovision_2025_predictions.csv', index=False, encoding='utf-8')
        print(f"\nTahminler 'eurovision_2025_predictions.csv' dosyasına kaydedildi.")
        
        # Veri analizi raporu oluştur
        print("\n=== VERİ ANALİZİ RAPORU ===")
        print(f"Toplam yarışma sayısı: {len(df)}")
        if 'year' in df.columns:
            print(f"Yıl aralığı: {df['year'].min()} - {df['year'].max()}")
        if 'country' in df.columns:
            print(f"Toplam ülke sayısı: {df['country'].nunique()}")
            print(f"En çok katılan ülkeler: {df['country'].value_counts().head(3).to_dict()}")
        
    except FileNotFoundError as e:
        print(f"HATA: Dosya bulunamadı!")
        print("Lütfen aşağıdaki dosyaların Python script'i ile aynı klasörde olduğundan emin olun:")
        print("- contest_data.csv")
        print("- country_data.csv") 
        print("- song_data.csv")
        print(f"Hata detayı: {str(e)}")
    except Exception as e:
        print(f"HATA: {str(e)}")
        print("Veri setinizin formatını kontrol edin.")

if __name__ == "__main__":
    main()