import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings('ignore')

# Veri setini yükleme
tracks = pd.read_csv('dataset/tracks_features.csv/tracks_features.csv')

# Veri kümesinin boyutunu ve sütunlarını kontrol etme
print(tracks.shape)
print(tracks.info())

# Boş değerleri kontrol etme ve silme
tracks.dropna(subset=['name', 'album'], inplace=True)
tracks.isnull().sum().plot.bar()
plt.show()

# Kullanılmayacak sütunları kaldırma
tracks = tracks.drop(['id', 'album_id', 'artist_ids'], axis=1)

# Aynı şarkının farklı versiyonlarını kaldırma
tracks.drop_duplicates(subset=['name'], keep='first', inplace=True)

# En popüler 10.000 şarkıyı seçme (yoksa en son yıldaki şarkılar alınır)
tracks = tracks.sort_values(by=['year'], ascending=False).head(10000)

# t-SNE ile görselleştirme (sadece sayısal sütunlarla)
model = TSNE(n_components=2, random_state=0)
numeric_features = tracks.select_dtypes(include=np.number).head(500)
tsne_data = model.fit_transform(numeric_features)

plt.figure(figsize=(7, 7))
plt.scatter(tsne_data[:, 0], tsne_data[:, 1])
plt.show()

# Sanatçılar için CountVectorizer tanımlama
song_vectorizer = CountVectorizer(tokenizer=lambda x: x.split(", "))
song_vectorizer.fit(tracks['artists'])

# Benzerlikleri hesaplayan fonksiyon
def get_similarities(song_name, data):
    # Girdi şarkısının vektörleri
    text_array1 = song_vectorizer.transform(data[data['name'] == song_name]['artists']).toarray()
    num_array1 = data[data['name'] == song_name].select_dtypes(include=np.number).to_numpy()

    # Benzerlik hesaplama
    sim = []
    for idx, row in data.iterrows():
        name = row['name']
        text_array2 = song_vectorizer.transform(data[data['name'] == name]['artists']).toarray()
        num_array2 = data[data['name'] == name].select_dtypes(include=np.number).to_numpy()

        text_sim = cosine_similarity(text_array1, text_array2)[0][0]
        num_sim = cosine_similarity(num_array1, num_array2)[0][0]
        sim.append(text_sim + num_sim)
    return sim

# Şarkı önerisi yapan fonksiyon
def recommend_songs(song_name, data=tracks):
    if data[data['name'] == song_name].shape[0] == 0:
        print(f'"{song_name}" bulunamadı. İşte bazı öneriler:\n')
        for song in data.sample(n=5)['name'].values:
            print(song)
        return

    data['similarity_factor'] = get_similarities(song_name, data)
    data = data.sort_values(by='similarity_factor', ascending=False)

    # Girdi şarkısının kendisi hariç tutularak 5 öneri yapılıyor
    print(f'"{song_name}" için öneriler:\n')
    print(data[['name', 'artists']][1:6])

# Örnek çalıştırma
recommend_songs('Shape of You')
recommend_songs('Love Someone')
recommend_songs('Love me like you do')
