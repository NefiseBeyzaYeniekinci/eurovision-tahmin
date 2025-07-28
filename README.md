📊 Verilerle Eurovision 2025: Yapay Zeka Tahminleri vs İnsan Sezgisi
YKS sorularını çözen yapay zekaları duymuşsunuzdur; peki ya Eurovision 2025'in kazananını tahmin ettirsek? İşte tam olarak da bu merakla yola çıktım ve üç önde gelen yapay zeka modelini karşı karşıya getirdiğim bir deney gerçekleştirdim!

🎤 Katılan Modeller ve Tahminleri
Eurovision 2025'in galibini tahmin etme görevini üstlenen yapay zeka modelleri ve öngörüleri şu şekildeydi:

Gemini (Google): İsveç
DeepSeek: Finlandiya
ChatGPT (OpenAI): Ukrayna

🏆 Gerçek Kazanan ve Şok Eden Sonuç
Peki, modeller ne kadar isabetliydi? İşte asıl çarpıcı sonuç:
Eurovision 2025'in gerçek şampiyonu Avusturya oldu!

📂 Veri Kaynağı
Bu kapsamlı deneyde, tahminlerin temelini oluşturan veriler Eurovision Song Contest Data adlı Kaggle veri setinden alındı. Bu set, yarışma sıralamaları, şarkı meta verileri, anket sonuçları ve oylama bilgileri gibi zengin içerikler barındırıyor.
🔗 İlgili Veri Kaynağı: https://www.kaggle.com/datasets/diamondsnake/eurovision-song-contest-data

⚙️ Modellerin Kullandığı Teknoloji ve Kütüphaneler
Her model, tahminde bulunurken farklı teknolojiler ve kütüphaneler kullandı:

 Model	           Kullanılan Teknolojiler                                                                            	Açıklama
ChatGPT	         pandas, scikit-learn	                                      Veri okuma-birleştirme, LabelEncoder ile kategorik kodlama, RandomForestClassifier ile modelleme.
Gemini	         pandas, numpy, scikit-learn                    	          One-Hot Encoding, RandomForest ile olasılık tahmini, np.argmax() ile en yüksek ihtimali bulma.
DeepSeek	       pandas, numpy, scikit-learn, matplotlib, seaborn          	Veri hazırlama, model eğitimi, grafiklerle çıktı görselleştirme, class_weight dengesi.

