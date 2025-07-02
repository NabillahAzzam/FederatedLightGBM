#!/usr/bin/env python3
"""
Federated Learning Client 2 untuk deteksi penipuan kartu kredit
Menggunakan LightGBM dengan Differential Privacy dan Encryption
"""

import socket
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from cryptography.fernet import Fernet
import time
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - CLIENT2 - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FederatedClient:
    def __init__(self, client_id=2, server_host='localhost', server_port=8888):
        self.client_id = client_id
        self.server_host = server_host
        self.server_port = server_port
        self.local_model = None
        self.global_model = None
        self.cipher = None
        self.privacy_epsilon = 1.0  # Differential privacy parameter
        
        # Generate local dataset for this client
        self.X_train, self.X_val, self.y_train, self.y_val = self._generate_local_data()
        
        logger.info(f"Client {client_id} initialized")
        logger.info(f"Local training data shape: {self.X_train.shape}")
        logger.info(f"Local validation data shape: {self.X_val.shape}")
    
    def _generate_local_data(self):
        """Load real local credit card fraud dataset"""
        try:
            import pandas as pd
            from sklearn.preprocessing import StandardScaler
            
            # Option 1: Load data khusus untuk client ini
            if self.client_id == 1:
                data_file = 'dataset/client1_data.csv'
            else:
                data_file = 'dataset/client2_data.csv'
            
            # Option 2: Split data besar menjadi bagian untuk setiap client
            if not os.path.exists(data_file):
                data_file = 'dataset/train.csv'
                data = pd.read_csv(data_file)
                
                # Split data berdasarkan client_id
                total_samples = len(data)
                client_samples = total_samples // 2  # Bagi 2 untuk 2 client
                
                start_idx = (self.client_id - 1) * client_samples
                end_idx = start_idx + client_samples
                
                if self.client_id == 2:  # Client terakhir ambil sisa data
                    end_idx = total_samples
                    
                data = data.iloc[start_idx:end_idx]
            else:
                data = pd.read_csv(data_file)
            
            # Sesuaikan nama kolom target
            target_column = 'Class'  # Ganti sesuai dataset Anda
            
            # Pisahkan features dan target
            X = data.drop(target_column, axis=1)
            y = data[target_column]
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Feature scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split train-validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y.values, 
                test_size=0.2, 
                random_state=42 + self.client_id, 
                stratify=y
            )
            
            logger.info(f"Client {self.client_id} real data loaded:")
            logger.info(f"Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            logger.info(f"Validation: {X_val.shape[0]} samples")
            logger.info(f"Fraud ratio - Train: {y_train.mean():.4f}, Val: {y_val.mean():.4f}")
            
            return X_train, X_val, y_train, y_val
            
        except Exception as e:
            logger.error(f"Error loading real dataset: {e}")
            logger.info("Falling back to synthetic data")
            return self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        """Fallback synthetic data"""
        np.random.seed(42 + self.client_id)
        
        X, y = make_classification(
            n_samples=2000,
            n_features=20,
            n_redundant=2,
            n_informative=18,
            n_clusters_per_class=1,
            weights=[0.92, 0.08],
            random_state=42 + self.client_id
        )
    
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    def _add_differential_privacy_noise(self, model_params, epsilon=1.0):
        """
        Add differential privacy noise to model parameters
        """
        try:
            # Calculate sensitivity (simplified approach)
            sensitivity = 1.0 / len(self.X_train)
            
            # Add Laplace noise to model predictions instead of parameters
            # This is a simplified DP approach for demonstration
            noise_scale = sensitivity / epsilon
            
            # Add noise to model predictions on validation set
            val_predictions = model_params.predict(self.X_val)
            noisy_predictions = val_predictions + np.random.laplace(0, noise_scale, len(val_predictions))
            
            # Clip predictions to valid range [0, 1]
            noisy_predictions = np.clip(noisy_predictions, 0, 1)
            
            logger.info(f"Added differential privacy noise with epsilon={epsilon}")
            
            return model_params, noisy_predictions
            
        except Exception as e:
            logger.error(f"Error adding DP noise: {e}")
            return model_params, model_params.predict(self.X_val)
    
    def _train_local_model(self):
        """Train local LightGBM model dengan parameter untuk dataset real"""
        try:
            # Parameter yang disesuaikan untuk fraud detection
            params = {
                'objective': 'binary',
                'metric': ['binary_logloss', 'auc'],
                'boosting_type': 'gbdt',
                'num_leaves': 31 if self.client_id == 1 else 25,
                'learning_rate': 0.05,  # Lebih kecil untuk dataset real
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42 + self.client_id,
                'is_unbalance': True,
                'min_data_in_leaf': 20,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'max_depth': 6,
                'min_gain_to_split': 0.1
            }
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(self.X_train, label=self.y_train)
            val_data = lgb.Dataset(self.X_val, label=self.y_val, reference=train_data)
            
            # Train model dengan lebih banyak iterations
            self.local_model = lgb.train(
                params,
                train_data,
                num_boost_round=200,  # Lebih banyak untuk dataset real
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=20),
                    lgb.log_evaluation(period=0)  # Silent training
                ],
                verbose_eval=False
            )
            
            # Evaluasi dengan metrik yang lebih lengkap
            train_pred = self.local_model.predict(self.X_train)
            val_pred = self.local_model.predict(self.X_val)
            
            from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
            
            train_auc = roc_auc_score(self.y_train, train_pred)
            val_auc = roc_auc_score(self.y_val, val_pred)
            
            train_pred_binary = (train_pred > 0.5).astype(int)
            val_pred_binary = (val_pred > 0.5).astype(int)
            
            train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
                self.y_train, train_pred_binary, average='binary'
            )
            val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
                self.y_val, val_pred_binary, average='binary'
            )
            
            logger.info(f"Local model trained successfully")
            logger.info(f"Training - AUC: {train_auc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
            logger.info(f"Validation - AUC: {val_auc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training local model: {e}")
            return False
    
    def _connect_to_server(self):
        """Connect to federated server and exchange models"""
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to connect to server (attempt {attempt + 1}/{max_retries})")
                
                # Create socket connection
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                client_socket.settimeout(30)  # 30 second timeout
                client_socket.connect((self.server_host, self.server_port))
                
                logger.info("Connected to server successfully")
                
                # Receive encryption key from server
                encryption_key = client_socket.recv(44)  # Fernet key is 44 bytes
                self.cipher = Fernet(encryption_key)
                logger.info("Received encryption key from server")
                
                # Receive global model from server
                model_size_data = client_socket.recv(4)
                if model_size_data:
                    model_size = int.from_bytes(model_size_data, byteorder='big')
                    
                    # Receive encrypted model
                    encrypted_model = b''
                    while len(encrypted_model) < model_size:
                        chunk = client_socket.recv(min(4096, model_size - len(encrypted_model)))
                        if not chunk:
                            break
                        encrypted_model += chunk
                    
                    # Decrypt and deserialize global model
                    decrypted_model = self.cipher.decrypt(encrypted_model)
                    self.global_model = pickle.loads(decrypted_model)
                    logger.info("Received global model from server")
                
                # Train local model
                if self._train_local_model():
                    # Add differential privacy noise
                    noisy_model, noisy_predictions = self._add_differential_privacy_noise(
                        self.local_model, 
                        self.privacy_epsilon
                    )
                    
                    # Prepare data to send to server
                    client_data = {
                        'model': noisy_model,
                        'predictions': noisy_predictions,
                        'client_id': self.client_id
                    }
                    
                    # Encrypt and send to server
                    serialized_data = pickle.dumps(client_data)
                    encrypted_data = self.cipher.encrypt(serialized_data)
                    
                    # Send data size first
                    data_size = len(encrypted_data)
                    client_socket.send(data_size.to_bytes(4, byteorder='big'))
                    
                    # Send encrypted data
                    client_socket.sendall(encrypted_data)
                    logger.info("Sent encrypted local model to server")
                    
                    client_socket.close()
                    return True
                
            except socket.timeout:
                logger.warning(f"Connection timeout on attempt {attempt + 1}")
            except ConnectionRefusedError:
                logger.warning(f"Connection refused on attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"Connection error on attempt {attempt + 1}: {e}")
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
        logger.error("Failed to connect to server after all retries")
        return False
    
    def run_federated_learning(self, max_rounds=5):
        """Run federated learning for multiple rounds"""
        logger.info(f"Starting federated learning client {self.client_id}")
        
        for round_num in range(max_rounds):
            logger.info(f"\n=== CLIENT {self.client_id} - ROUND {round_num + 1}/{max_rounds} ===")
            
            if self._connect_to_server():
                logger.info(f"Round {round_num + 1} completed successfully")
            else:
                logger.error(f"Round {round_num + 1} failed")
                break
            
            # Wait before next round
            if round_num < max_rounds - 1:
                logger.info("Waiting 15 seconds before next round...")
                time.sleep(15)
        
        logger.info(f"Client {self.client_id} federated learning completed")

if __name__ == "__main__":
    # Client 2 configuration
    client = FederatedClient(
        client_id=2,
        server_host='localhost',
        server_port=8888
    )
    
    client.run_federated_learning(max_rounds=5)