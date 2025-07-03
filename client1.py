# fl_client1.py

#!/usr/bin/env python3
"""
NFS-based Federated Learning Client 1 untuk deteksi penipuan kartu kredit
"""
import os
import pickle
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import time
import logging
from datetime import datetime
import argparse

# Class NFSFederatedClient (SAMA PERSIS seperti sebelumnya)
# ... (Salin seluruh class NFSFederatedClient dari jawaban sebelumnya)
class NFSFederatedClient:
    def __init__(self, client_id, nfs_path='/shared/federated_learning', max_rounds=5):
        self.client_id = client_id
        self.nfs_path = nfs_path
        self.max_rounds = max_rounds
        self.current_round = 0
        
        self.logger = logging.getLogger(f'CLIENT_{client_id}')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(f'%(asctime)s - CLIENT {self.client_id} - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False

        self.models_dir = os.path.join(nfs_path, 'models')
        self.global_dir = os.path.join(nfs_path, 'global')
        self.status_dir = os.path.join(nfs_path, 'status')
        self.results_dir = os.path.join(nfs_path, 'results')
        
        for dir_path in [self.models_dir, self.global_dir, self.status_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        self.X_train, self.X_val, self.y_train, self.y_val = self._load_local_data()
        self.local_model = None
        self.global_model = None
        
        self.logger.info(f"Client {client_id} initialized")
        self.logger.info(f"NFS Path: {nfs_path}")
        self.logger.info(f"Training data shape: {self.X_train.shape}")
        self.logger.info(f"Validation data shape: {self.X_val.shape}")
    
    def _load_local_data(self):
        # Cari dataset spesifik untuk klien
        dataset_path = f'dataset/client{self.client_id}_data.csv'
        shared_dataset_path = f'/shared/dataset/client{self.client_id}_data.csv'

        data = None
        data_file = None

        if os.path.exists(dataset_path):
            data_file = dataset_path
        elif os.path.exists(shared_dataset_path):
            data_file = shared_dataset_path

        if data_file:
            self.logger.info(f"Loading specific dataset from: {data_file}")
            data = pd.read_csv(data_file)
        else:
            self.logger.warning(f"No specific dataset found for client {self.client_id}. Check for {dataset_path} or {shared_dataset_path}.")
            self.logger.info("Falling back to synthetic data.")
            return self._generate_synthetic_data()

        target_candidates = ['Class', 'isFraud', 'fraud', 'target', 'label']
        target_col = next((col for col in target_candidates if col in data.columns), data.columns[-1])

        X = data.drop(target_col, axis=1)
        y = data[target_col]
        X = X.fillna(X.mean())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y.values, test_size=0.2, random_state=42 + self.client_id, stratify=y)
        self.logger.info(f"Real dataset loaded. Train: {X_train.shape[0]} samples, Val: {X_val.shape[0]} samples.")
        return X_train, X_val, y_train, y_val

    # ... (Metode lainnya: _generate_synthetic_data, _wait_for_global_model, _load_global_model, _train_local_model, _save_local_model, _wait_for_round_completion) ...
    # ... (Salin semua metode ini dari jawaban sebelumnya, mereka tidak berubah) ...
    def _generate_synthetic_data(self):
        np.random.seed(42 + self.client_id)
        X, y = make_classification(n_samples=2000, n_features=30, n_redundant=5, n_informative=25, n_clusters_per_class=1, weights=[0.92, 0.08], random_state=42 + self.client_id)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.logger.info("Generated synthetic dataset")
        return X_train, X_val, y_train, y_val
    def _wait_for_global_model(self, round_num, timeout=300):
        self.logger.info(f"Waiting for global model for round {round_num}")
        start_time = time.time()
        status_file = os.path.join(self.status_dir, f'global_model_ready_round_{round_num}.flag')
        while not os.path.exists(status_file):
            if time.time() - start_time > timeout:
                self.logger.error(f"Timeout waiting for global model for round {round_num}")
                return False
            time.sleep(2)
        self.logger.info(f"Global model ready signal found for round {round_num}")
        return True
    def _load_global_model(self, round_num):
        try:
            model_path = os.path.join(self.global_dir, f'global_model_round_{round_num}.pkl')
            time.sleep(1) 
            with open(model_path, 'rb') as f: self.global_model = pickle.load(f)
            self.logger.info(f"Global model loaded for round {round_num}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading global model: {e}")
            return False
    def _train_local_model(self):
        try:
            train_data = lgb.Dataset(self.X_train, label=self.y_train)
            valid_data = lgb.Dataset(self.X_val, label=self.y_val, reference=train_data)
            params = {'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1, 'random_state': 42 + self.client_id, 'is_unbalance': True}
            self.logger.info(f"Starting local training for round {self.current_round}...")
            self.local_model = lgb.train(params, train_data, num_boost_round=50, valid_sets=[valid_data], init_model=self.global_model, callbacks=[lgb.early_stopping(10, verbose=False), lgb.log_evaluation(period=0)])
            y_pred_val = self.local_model.predict(self.X_val)
            local_auc = roc_auc_score(self.y_val, y_pred_val)
            self.logger.info(f"Local training finished. Local validation AUC: {local_auc:.4f}")
        except Exception as e:
            self.logger.error(f"Error during local training: {e}", exc_info=True)
            self.local_model = self.global_model
    def _save_local_model(self):
        try:
            model_filename = f'client_{self.client_id}_round_{self.current_round}_model.pkl'
            model_path = os.path.join(self.models_dir, model_filename)
            metadata_filename = f'client_{self.client_id}_round_{self.current_round}_metadata.json'
            metadata_path = os.path.join(self.models_dir, metadata_filename)
            with open(model_path, 'wb') as f: pickle.dump(self.local_model, f)
            metadata = {'client_id': self.client_id, 'round': self.current_round, 'timestamp': datetime.now().isoformat(), 'model_path': model_path, 'num_training_samples': len(self.X_train)}
            with open(metadata_path, 'w') as f: json.dump(metadata, f, indent=2)
            self.logger.info(f"Local model and metadata for round {self.current_round} saved to NFS.")
            return True
        except Exception as e:
            self.logger.error(f"Error saving local model: {e}")
            return False
    def _wait_for_round_completion(self, round_num, timeout=300):
        self.logger.info(f"Waiting for server to complete round {round_num}...")
        start_time = time.time()
        completion_flag = os.path.join(self.status_dir, f'round_{round_num}_completed.flag')
        while not os.path.exists(completion_flag):
            if time.time() - start_time > timeout:
                self.logger.warning(f"Timeout waiting for round {round_num} completion signal.")
                return False
            time.sleep(5)
        self.logger.info(f"Server completed aggregation for round {round_num}.")
        return True
    
    def run_federated_learning(self):
        for round_num in range(1, self.max_rounds + 1):
            self.current_round = round_num
            self.logger.info(f"\n--- STARTING ROUND {self.current_round}/{self.max_rounds} ---")
            if not self._wait_for_global_model(self.current_round): break
            if not self._load_global_model(self.current_round): break
            self._train_local_model()
            if not self._save_local_model(): break
            self.logger.info(f"Round {self.current_round} completed. Waiting for the next round.")
            if round_num < self.max_rounds: self._wait_for_round_completion(self.current_round)
        self.logger.info(f"Client {self.client_id} has finished all federated learning rounds.")

# --- BAGIAN UTAMA YANG BERBEDA ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Learning Client 1')
    parser.add_argument('--nfs-path', default='/shared/federated_learning', help='NFS mount path')
    parser.add_argument('--max-rounds', type=int, default=5, help='Maximum number of federated rounds')
    args = parser.parse_args()
    
    # ID Klien ditetapkan secara eksplisit
    CLIENT_ID = 1
    
    print(f"--- Starting Federated Learning Client {CLIENT_ID} ---")
    
    try:
        client = NFSFederatedClient(
            client_id=CLIENT_ID,
            nfs_path=args.nfs_path,
            max_rounds=args.max_rounds
        )
        client.run_federated_learning()
    except Exception as e:
        logging.error(f"Critical error in client {CLIENT_ID}: {e}", exc_info=True)

    print(f"--- Client {CLIENT_ID} has completed its tasks. ---")