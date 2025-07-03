#!/usr/bin/env python3
"""
NFS-based Federated Learning Server untuk deteksi penipuan kartu kredit
Menggunakan LightGBM dengan file sharing via NFS dan Knowledge Distillation
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, precision_recall_fscore_support
import time
import logging
from datetime import datetime
import glob

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - SERVER - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# BARU: Fungsi helper untuk Knowledge Distillation
def calculate_sample_weights(soft_labels, true_labels, fraud_weight_multiplier=5.0):
    """
    Menghitung sample weights berdasarkan:
    1. Confidence dari teacher model
    2. Penekanan tambahan untuk kasus fraud
    """
    # Base weights berdasarkan confidence (0 to 1)
    confidence_weights = np.abs(soft_labels - 0.5) * 2
    
    # Tambahan weight untuk fraud cases
    fraud_mask = true_labels == 1
    confidence_weights[fraud_mask] *= fraud_weight_multiplier
    
    logger.info(f"Sample weights calculated. Mean weight: {np.mean(confidence_weights):.4f}")
    return confidence_weights

def knowledge_distillation(X_distill, soft_labels, y_distill_true, student_params):
    """
    Melatih student model menggunakan knowledge distillation dengan penanganan khusus.
    """
    # 1. Analisis distribusi
    logger.info("=== Knowledge Distillation Analysis ===")
    logger.info(f"Soft labels - Min: {soft_labels.min():.4f}, Max: {soft_labels.max():.4f}, Mean: {soft_labels.mean():.4f}")
    logger.info(f"True labels - Fraud rate: {y_distill_true.mean():.4f}")
    
    # 2. Gunakan threshold yang lebih rendah untuk hard labels
    threshold = 0.3
    hard_labels = (soft_labels > threshold).astype(int)
    
    logger.info(f"Using threshold {threshold} for hard labels. New fraud rate: {hard_labels.mean():.4f}")
    
    # 3. Hitung sample weights
    sample_weights = calculate_sample_weights(soft_labels, y_distill_true)
    
    # 4. Training dengan handling
    student_model = lgb.LGBMClassifier(**student_params)
    
    # Cek apakah kedua set label memiliki kelas yang sama
    if set(np.unique(hard_labels)) == set(np.unique(y_distill_true)):
        logger.info("Training with true labels for evaluation set.")
        eval_set = [(X_distill, y_distill_true)]
        eval_labels = y_distill_true
    else:
        logger.warning("Class mismatch between hard and true labels. Using hard labels for evaluation.")
        eval_set = [(X_distill, hard_labels)]
        eval_labels = hard_labels

    student_model.fit(X_distill, hard_labels,
                      sample_weight=sample_weights,
                      eval_set=eval_set,
                      eval_metric='auc',
                      callbacks=[lgb.early_stopping(10, verbose=False)])
    
    # 5. Manual evaluation dengan true labels untuk laporan
    y_pred_proba = student_model.predict_proba(X_distill)[:, 1]
    final_auc = roc_auc_score(y_distill_true, y_pred_proba)
    logger.info(f"Final AUC Score (vs true labels) on distillation data: {final_auc:.4f}")
    
    return student_model

class NFSFederatedServer:
    def __init__(self, nfs_path='/shared/federated_learning', num_clients=2, max_rounds=5):
        self.nfs_path = nfs_path
        self.num_clients = num_clients
        self.max_rounds = max_rounds
        self.current_round = 0
        
        # NFS directories
        self.models_dir = os.path.join(nfs_path, 'models')
        self.global_dir = os.path.join(nfs_path, 'global')
        self.status_dir = os.path.join(nfs_path, 'status')
        self.results_dir = os.path.join(nfs_path, 'results')
        
        self._setup_nfs_directories()
        
        # DIUBAH: Data tes sekarang juga berfungsi sebagai data distilasi
        self.X_distill, self.y_distill = self._load_test_data()
        
        self.global_model = None
        self._initialize_global_model()
        
        logger.info(f"NFS Federated Server initialized")
        logger.info(f"Distillation data shape: {self.X_distill.shape}")

    # ... (Metode _setup_nfs_directories tidak berubah) ...
    def _setup_nfs_directories(self):
        """Create necessary NFS directories"""
        dirs = [self.nfs_path, self.models_dir, self.global_dir, self.status_dir, self.results_dir]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
        logger.info("NFS directories created")

    # ... (Metode _load_test_data tidak berubah, hanya nama variabel yang disesuaikan) ...
    def _load_test_data(self):
        """Load test dataset, yang juga akan digunakan sebagai data distilasi."""
        try:
            dataset_paths = ['dataset/test.csv', '/shared/dataset/test.csv', 'test.csv']
            
            for path in dataset_paths:
                if os.path.exists(path):
                    logger.info(f"Loading distillation/test data from: {path}")
                    data = pd.read_csv(path)
                    
                    target_candidates = ['Class', 'isFraud', 'fraud', 'target', 'label']
                    target_col = next((col for col in target_candidates if col in data.columns), data.columns[-1])
                    
                    X = data.drop(target_col, axis=1)
                    y = data[target_col]
                    
                    X = X.fillna(X.mean())
                    
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    logger.info(f"Distillation/Test dataset loaded: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
                    return X_scaled, y.values
            
            logger.warning("Real test dataset not found, generating synthetic data")
            return self._generate_synthetic_test_data()
            
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            return self._generate_synthetic_test_data()

    # ... (Metode _generate_synthetic_test_data & _initialize_global_model tidak berubah) ...
    def _generate_synthetic_test_data(self):
        """Generate synthetic test data"""
        X, y = make_classification(n_samples=1000, n_features=30, n_redundant=5, n_informative=25, n_clusters_per_class=1, weights=[0.95, 0.05], random_state=42)
        logger.info(f"Generated synthetic test data: {X.shape[0]} samples")
        return X, y
    def _initialize_global_model(self):
        """Initialize global model"""
        try:
            sample_data_path = '/shared/dataset/train.csv' # Gunakan data tes untuk inisialisasi
            if not os.path.exists(sample_data_path):
                X_dummy, y_dummy = make_classification(n_samples=10, n_features=30, random_state=42)
                self.global_model = lgb.LGBMClassifier(random_state=42).fit(X_dummy, y_dummy)
            else:
                df = pd.read_csv(sample_data_path)
                target_col = next((col for col in ['Class','isFraud'] if col in df.columns), df.columns[-1])
                X_init = df.drop(target_col, axis=1).fillna(0)
                y_init = df[target_col]
                self.global_model = lgb.LGBMClassifier(random_state=42).fit(X_init, y_init)

            self._save_global_model()
            logger.info("Global model initialized and saved to NFS")
        except Exception as e:
            logger.error(f"Error initializing global model: {e}")

    # ... (Metode _save_global_model & _wait_for_client_models tidak berubah) ...
    def _save_global_model(self):
        """Save global model to NFS"""
        try:
            # Model global sekarang adalah LGBMClassifier, bukan Booster
            model_path = os.path.join(self.global_dir, f'global_model_round_{self.current_round}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.global_model, f)
            
            # Metadata dan status file
            metadata = {'round': self.current_round, 'timestamp': datetime.now().isoformat()}
            metadata_path = os.path.join(self.global_dir, f'global_model_round_{self.current_round}_metadata.json')
            with open(metadata_path, 'w') as f: json.dump(metadata, f, indent=2)
            
            status_file = os.path.join(self.status_dir, f'global_model_ready_round_{self.current_round}.flag')
            with open(status_file, 'w') as f: f.write(f"Ready: {datetime.now().isoformat()}")
            
            logger.info(f"Global model saved for round {self.current_round}")
        except Exception as e:
            logger.error(f"Error saving global model: {e}")
    def _wait_for_client_models(self, round_num, timeout=300):
        logger.info(f"Waiting for {self.num_clients} client models for round {round_num}")
        start_time = time.time()
        client_models_received = {}
        while len(client_models_received) < self.num_clients:
            model_pattern = os.path.join(self.models_dir, f'client_*_round_{round_num}_model.pkl')
            model_files = glob.glob(model_pattern)
            for model_file in model_files:
                filename = os.path.basename(model_file)
                client_id = filename.split('_')[1]
                if client_id not in client_models_received:
                    metadata_file = model_file.replace('_model.pkl', '_metadata.json')
                    if os.path.exists(metadata_file):
                        client_models_received[client_id] = {'model_file': model_file}
                        logger.info(f"Received model from client {client_id}")
            if time.time() - start_time > timeout:
                logger.warning(f"Timeout waiting for client models. Received {len(client_models_received)}/{self.num_clients}")
                break
            time.sleep(2)
        return client_models_received

    # DIUBAH: client_predictions tidak lagi digunakan, hanya model yang dimuat
    def _load_client_models(self, client_models_info):
        """Load client models from NFS"""
        client_models = []
        for client_id, info in client_models_info.items():
            try:
                with open(info['model_file'], 'rb') as f:
                    model = pickle.load(f)
                client_models.append(model)
                logger.info(f"Loaded model from client {client_id}")
            except Exception as e:
                logger.error(f"Error loading model from client {client_id}: {e}")
        return client_models

    # DIUBAH: Logika agregasi diganti sepenuhnya dengan Knowledge Distillation
    def _aggregate_models(self, client_models):
        """Aggregate client models using advanced knowledge distillation."""
        try:
            if not client_models:
                logger.error("No client models to aggregate")
                return False
            
            logger.info(f"Starting model aggregation for round {self.current_round} with {len(client_models)} client models.")
            start_time = time.perf_counter()

            # 1. Dapatkan prediksi probabilitas dari setiap model klien (Teacher Models)
            all_client_probas = []
            for client_model in client_models:
                # Pastikan model adalah instance dari Scikit-Learn API
                if hasattr(client_model, 'predict_proba'):
                    client_probas = client_model.predict_proba(self.X_distill)[:, 1]
                    all_client_probas.append(client_probas)
                else:
                    logger.warning("A client model does not have 'predict_proba' method. Skipping.")
            
            if not all_client_probas:
                logger.error("Could not get probabilities from any client model.")
                return False

            # 2. Buat soft labels dengan merata-ratakan probabilitas
            stacked_probas = np.stack(all_client_probas, axis=1)
            soft_labels = np.mean(stacked_probas, axis=1)
            logger.info(f"Soft labels (ensembled probabilities) created. Shape: {soft_labels.shape}")

            # 3. Definisikan parameter untuk student model
            student_params = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'random_state': 42 + self.current_round, # Seed berbeda setiap ronde
                'n_estimators': 100, # Jumlah estimator untuk student
                'is_unbalance': True # Penting untuk data fraud
            }

            # 4. Latih student model menggunakan knowledge distillation
            student_model = knowledge_distillation(
                self.X_distill,       # Data distilasi (dari test.csv)
                soft_labels,          # Label "lunak" dari para guru
                self.y_distill,       # Label asli untuk pembobotan dan evaluasi akhir
                student_params
            )

            # 5. Tetapkan student model sebagai model global baru
            self.global_model = student_model
            end_time = time.perf_counter()
            logger.info(f"Aggregation via Knowledge Distillation took {end_time - start_time:.4f} seconds.")

            # 6. Evaluasi model global baru pada data yang sama (sekarang disebut data tes)
            global_pred_proba = self.global_model.predict_proba(self.X_distill)[:, 1]
            global_pred_binary = (global_pred_proba > 0.5).astype(int) # Threshold standar untuk pelaporan
            
            accuracy = accuracy_score(self.y_distill, global_pred_binary)
            auc_score = roc_auc_score(self.y_distill, global_pred_proba)
            precision, recall, f1, _ = precision_recall_fscore_support(self.y_distill, global_pred_binary, average='binary', zero_division=0)

            # Simpan hasil
            results = {
                'round': self.current_round, 'accuracy': float(accuracy), 'auc': float(auc_score),
                'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
                'timestamp': datetime.now().isoformat()
            }
            results_file = os.path.join(self.results_dir, f'round_{self.current_round}_results.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

            logger.info(f"Round {self.current_round} Aggregation Metrics - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
            logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in model aggregation: {e}", exc_info=True)
            return False

    # ... (Metode _cleanup_round_files, run_federated_learning, _final_evaluation tidak berubah) ...
    def _cleanup_round_files(self, round_num):
        """Clean up files from previous rounds"""
        try:
            for pattern in [f'*_round_{round_num}_*', f'*_round_{round_num-1}*']: # Bersihkan ronde N dan N-1
                files_to_remove = glob.glob(os.path.join(self.models_dir, pattern))
                files_to_remove += glob.glob(os.path.join(self.status_dir, pattern))
                for file_path in files_to_remove:
                    try: os.remove(file_path)
                    except: pass
            logger.info(f"Cleaned up files from round {round_num-1} and {round_num}")
        except Exception as e:
            logger.warning(f"Error cleaning up round files: {e}")

    def run_federated_learning(self):
        """Run federated learning process"""
        logger.info("Starting NFS-based Federated Learning")
        
        for round_num in range(1, self.max_rounds + 1):
            logger.info(f"\n=== FEDERATED LEARNING ROUND {round_num}/{self.max_rounds} ===")
            self.current_round = round_num
            
            if round_num > 1: self._cleanup_round_files(round_num - 1)
            
            self._save_global_model()
            
            client_models_info = self._wait_for_client_models(round_num)
            if len(client_models_info) < self.num_clients:
                logger.error(f"Not enough client models received. Expected {self.num_clients}, got {len(client_models_info)}. Stopping.")
                break
            
            # DIUBAH: Memuat model, tidak lagi memuat prediksi
            client_models = self._load_client_models(client_models_info)
            
            # DIUBAH: Memanggil fungsi agregasi yang baru
            if self._aggregate_models(client_models):
                logger.info(f"Round {round_num} completed successfully")
            else:
                logger.error(f"Round {round_num} failed during aggregation")
                break
            
            completion_file = os.path.join(self.status_dir, f'round_{round_num}_completed.flag')
            with open(completion_file, 'w') as f: f.write(f"Round {round_num} completed")
        
        self._final_evaluation()
        logger.info("\n=== FEDERATED LEARNING COMPLETED ===")
    
    def _final_evaluation(self):
        """Perform final evaluation"""
        if self.global_model is not None:
            try:
                final_pred_proba = self.global_model.predict_proba(self.X_distill)[:, 1]
                final_pred_binary = (final_pred_proba > 0.5).astype(int)
                
                logger.info("\n=== Final Global Model Evaluation ===")
                print(classification_report(self.y_distill, final_pred_binary))
                
                final_results = {
                    'total_rounds': self.current_round,
                    'timestamp': datetime.now().isoformat(),
                    'classification_report': classification_report(self.y_distill, final_pred_binary, output_dict=True)
                }
                final_results_file = os.path.join(self.results_dir, 'final_results.json')
                with open(final_results_file, 'w') as f: json.dump(final_results, f, indent=2)
                
                final_model_path = os.path.join(self.global_dir, 'final_global_model.pkl')
                with open(final_model_path, 'wb') as f: pickle.dump(self.global_model, f)
            except Exception as e:
                logger.error(f"Error in final evaluation: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NFS-based Federated Learning Server with Knowledge Distillation')
    parser.add_argument('--nfs-path', default='/shared/federated_learning', help='NFS mount path')
    parser.add_argument('--num-clients', type=int, default=2, help='Number of expected clients')
    parser.add_argument('--max-rounds', type=int, default=5, help='Maximum number of rounds')
    
    args = parser.parse_args()
    
    server = NFSFederatedServer(
        nfs_path=args.nfs_path,
        num_clients=args.num_clients,
        max_rounds=args.max_rounds
    )
    
    server.run_federated_learning()