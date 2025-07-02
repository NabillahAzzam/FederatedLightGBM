# Federated Learning untuk Deteksi Penipuan Kartu Kredit

## Deskripsi
Implementasi federated learning menggunakan LightGBM untuk deteksi penipuan kartu kredit dengan fitur:
- **Knowledge Distillation** dengan hard labels untuk agregasi model
- **Differential Privacy** untuk privasi data
- **Enkripsi** untuk komunikasi aman
- **Dataset sintetis** yang mensimulasikan kasus penipuan kartu kredit

## Arsitektur
- 1 Server (server.py) 
- 2 Client (client1.py dan client2.py)
- Menggunakan socket TCP untuk komunikasi
- Enkripsi end-to-end dengan Fernet

## Instalasi

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verifikasi Instalasi
```python
import lightgbm as lgb
import sklearn
import cryptography
print("Semua dependencies berhasil diinstall!")
```

## Cara Menjalankan

### Urutan Eksekusi:

#### 1. Jalankan Server (Terminal 1)
```bash
python server.py
```

#### 2. Jalankan Client 1 (Terminal 2)
```bash
python client1.py
```

#### 3. Jalankan Client 2 (Terminal 3)
```bash  
python client2.py
```

## Fitur Utama

### 1. Knowledge Distillation
- Server menggunakan ensemble predictions dari semua client
- Hard labels dibuat dari rata-rata predictions (threshold 0.5)
- Global model dilatih ulang menggunakan hard labels

### 2. Differential Privacy
- Noise Laplace ditambahkan ke predictions
- Parameter epsilon berbeda untuk setiap client (1.0 dan 1.2)
- Melindungi privasi data lokal client

### 3. Enkripsi
- Menggunakan Fernet (symmetric encryption)
- Server generate key, dikirim ke client
- Semua komunikasi model terenkripsi

### 4. Dataset Sintetis
- Mensimulasikan deteksi penipuan kartu kredit
- Data tidak seimbang (95% normal, 5% penipuan)
- Setiap client memiliki distribusi data yang berbeda

## Konfigurasi

ganti localhost dengan ip address server!!!

### Server (server.py)
```python
FederatedServer(
    host='localhost',
    port=8888, 
    num_clients=2
)
```

### Client 1 & 2
```python
FederatedClient(
    client_id=1,  # atau 2
    server_host='localhost',
    server_port=8888
)
```

## Parameter yang Dapat Disesuaikan

### Model LightGBM
- `num_leaves`: 31 (client1), 25 (client2)
- `learning_rate`: 0.1 (client1), 0.12 (client2)
- `num_boost_round`: 100 (client1), 120 (client2)

### Differential Privacy
- `epsilon`: 1.0 (client1), 1.2 (client2)
- Semakin kecil epsilon = privasi lebih tinggi, akurasi lebih rendah

### Federated Learning
- `max_rounds`: 5 (default)
- `retry_attempts`: 3 per client
- `timeout`: 30 detik per koneksi

## Output yang Diharapkan

### Server Log:
```
Server initialized with 2 clients expected
=== FEDERATED LEARNING ROUND 1/5 ===
Server listening on localhost:8888
All clients connected. Starting model aggregation...
Global model updated via Knowledge Distillation
Global model accuracy: 0.8520
```

### Client Log:
```
CLIENT1 - Local training data shape: (1600, 20)
CLIENT1 - Local model trained successfully
CLIENT1 - Training accuracy: 0.8734
CLIENT1 - Validation accuracy: 0.8625
CLIENT1 - Added differential privacy noise with epsilon=1.0
CLIENT1 - Sent encrypted local model to server
```

## Troubleshooting

### 1. Connection Refused
- Pastikan server sudah running sebelum client
- Periksa port 8888 tidak digunakan aplikasi lain

### 2. Import Error
- Install semua dependencies: `pip install -r requirements.txt`
- Gunakan Python 3.8+ 

### 3. Memory Error
- Kurangi ukuran dataset di `make_classification()`
- Reduce `num_boost_round` di parameter LightGBM

### 4. Encryption Error
- Pastikan key exchange berhasil
- Restart server dan client jika terjadi error

## Struktur File
```
federated_learning/
├── global_server.py      # Server federated learning
├── client1.py           # Client pertama  
├── client2.py           # Client kedua
├── requirements.txt     # Dependencies
└── README.md           # Panduan ini
```

## Modifikasi untuk Data Real

Untuk menggunakan dataset real:

1. **Ganti fungsi `_generate_local_data()`** di client:
```python
def _generate_local_data(self):
    # Load your real credit card fraud dataset
    data = pd.read_csv('your_fraud_dataset.csv')
    X = data.drop('fraud_label', axis=1)
    y = data['fraud_label']
    return train_test_split(X, y, test_size=0.2, stratify=y)
```

2. **Update `_generate_test_data()`** di server dengan data test real

3. **Sesuaikan parameter LightGBM** berdasarkan karakteristik data real

## Keamanan dan Privasi

- ✅ Enkripsi komunikasi end-to-end
- ✅ Differential privacy untuk perlindungan data
- ✅ Data tidak pernah dibagi secara langsung
- ✅ Only model updates yang dikirim ke server
- ✅ Noise injection untuk mencegah model inversion attacks

## Limitasi

- Implementasi DP disederhanakan (untuk demonstrasi)
- Tidak ada Byzantine fault tolerance
- Single point of failure pada server
- Tidak ada mekanisme dropout client
- Komunikasi synchronous (semua client harus online)

Implementasi ini memberikan foundation yang solid untuk federated learning dengan fokus pada privacy dan security untuk kasus deteksi penipuan kartu kredit.# Federated Learning untuk Deteksi Penipuan Kartu Kredit

## Deskripsi
Implementasi federated learning menggunakan LightGBM untuk deteksi penipuan kartu kredit dengan fitur:
- **Knowledge Distillation** dengan hard labels untuk agregasi model
- **Differential Privacy** untuk privasi data
- **Enkripsi** untuk komunikasi aman
- **Dataset sintetis** yang mensimulasikan kasus penipuan kartu kredit

## Arsitektur
- 1 Server (global_server.py) 
- 2 Client (client1.py dan client2.py)
- Menggunakan socket TCP untuk komunikasi
- Enkripsi end-to-end dengan Fernet

## Instalasi

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verifikasi Instalasi
```python
import lightgbm as lgb
import sklearn
import cryptography
print("Semua dependencies berhasil diinstall!")
```

## Cara Menjalankan

### Urutan Eksekusi:

#### 1. Jalankan Server (Terminal 1)
```bash
python global_server.py
```

#### 2. Jalankan Client 1 (Terminal 2)
```bash
python client1.py
```

#### 3. Jalankan Client 2 (Terminal 3)
```bash  
python client2.py
```

## Fitur Utama

### 1. Knowledge Distillation
- Server menggunakan ensemble predictions dari semua client
- Hard labels dibuat dari rata-rata predictions (threshold 0.5)
- Global model dilatih ulang menggunakan hard labels

### 2. Differential Privacy
- Noise Laplace ditambahkan ke predictions
- Parameter epsilon berbeda untuk setiap client (1.0 dan 1.2)
- Melindungi privasi data lokal client

### 3. Enkripsi
- Menggunakan Fernet (symmetric encryption)
- Server generate key, dikirim ke client
- Semua komunikasi model terenkripsi

### 4. Dataset Sintetis
- Mensimulasikan deteksi penipuan kartu kredit
- Data tidak seimbang (95% normal, 5% penipuan)
- Setiap client memiliki distribusi data yang berbeda

## Konfigurasi

### Server (global_server.py)
```python
FederatedServer(
    host='localhost',
    port=8888, 
    num_clients=2
)
```

### Client 1 & 2
```python
FederatedClient(
    client_id=1,  # atau 2
    server_host='localhost',
    server_port=8888
)
```

## Parameter yang Dapat Disesuaikan

### Model LightGBM
- `num_leaves`: 31 (client1), 25 (client2)
- `learning_rate`: 0.1 (client1), 0.12 (client2)
- `num_boost_round`: 100 (client1), 120 (client2)

### Differential Privacy
- `epsilon`: 1.0 (client1), 1.2 (client2)
- Semakin kecil epsilon = privasi lebih tinggi, akurasi lebih rendah

### Federated Learning
- `max_rounds`: 5 (default)
- `retry_attempts`: 3 per client
- `timeout`: 30 detik per koneksi

## Output yang Diharapkan

### Server Log:
```
Server initialized with 2 clients expected
=== FEDERATED LEARNING ROUND 1/5 ===
Server listening on localhost:8888
All clients connected. Starting model aggregation...
Global model updated via Knowledge Distillation
Global model accuracy: 0.8520
```

### Client Log:
```
CLIENT1 - Local training data shape: (1600, 20)
CLIENT1 - Local model trained successfully
CLIENT1 - Training accuracy: 0.8734
CLIENT1 - Validation accuracy: 0.8625
CLIENT1 - Added differential privacy noise with epsilon=1.0
CLIENT1 - Sent encrypted local model to server
```

## Troubleshooting

### 1. Connection Refused
- Pastikan server sudah running sebelum client
- Periksa port 8888 tidak digunakan aplikasi lain

### 2. Import Error
- Install semua dependencies: `pip install -r requirements.txt`
- Gunakan Python 3.8+ 

### 3. Memory Error
- Kurangi ukuran dataset di `make_classification()`
- Reduce `num_boost_round` di parameter LightGBM

### 4. Encryption Error
- Pastikan key exchange berhasil
- Restart server dan client jika terjadi error

## Struktur File
```
federated_learning/
├── global_server.py      # Server federated learning
├── client1.py           # Client pertama  
├── client2.py           # Client kedua
├── requirements.txt     # Dependencies
└── README.md           # Panduan ini
```

## Modifikasi untuk Data Real

Untuk menggunakan dataset real:

1. **Ganti fungsi `_generate_local_data()`** di client:
```python
def _generate_local_data(self):
    # Load your real credit card fraud dataset
    data = pd.read_csv('your_fraud_dataset.csv')
    X = data.drop('fraud_label', axis=1)
    y = data['fraud_label']
    return train_test_split(X, y, test_size=0.2, stratify=y)
```

2. **Update `_generate_test_data()`** di server dengan data test real

3. **Sesuaikan parameter LightGBM** berdasarkan karakteristik data real

## Keamanan dan Privasi

- ✅ Enkripsi komunikasi end-to-end
- ✅ Differential privacy untuk perlindungan data
- ✅ Data tidak pernah dibagi secara langsung
- ✅ Only model updates yang dikirim ke server
- ✅ Noise injection untuk mencegah model inversion attacks

## Limitasi

- Implementasi DP disederhanakan (untuk demonstrasi)
- Tidak ada Byzantine fault tolerance
- Single point of failure pada server
- Tidak ada mekanisme dropout client
- Komunikasi synchronous (semua client harus online)

Implementasi ini memberikan foundation yang solid untuk federated learning dengan fokus pada privacy dan security untuk kasus deteksi penipuan kartu kredit.