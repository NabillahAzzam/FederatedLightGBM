#!/usr/bin/env python3
"""
Federated Learning Server untuk deteksi penipuan kartu kredit
Menggunakan LightGBM dengan Knowledge Distillation dan Hard Labels
"""

import socket
import threading
import pickle
import numpy as np
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from cryptography.fernet import Fernet
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FederatedServer:
    def __init__(self, host='localhost', port=8888, num_clients=2):
        self.host = host
        self.port = port
        self.num_clients = num_clients
        self.clients = []
        self.client_models = []
        self.global_model = None
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        self.current_round = 0
        self.max_rounds = 5
        
        # Generate synthetic credit card fraud dataset
        self.X_test, self.y_test = self._generate_test_data()
        
        logger.info(f"Server initialized with {num_clients} clients expected")
        logger.info(f"Encryption key generated: {self.encryption_key.decode()[:20]}...")
    
    def _generate_test_data(self):
        """Load real credit card fraud test dataset"""
        try:
            import pandas as pd
            
            # Load test dataset
            test_data = pd.read_csv('dataset/test.csv')
            
            # Sesuaikan nama kolom target (ganti 'Class' dengan nama kolom target Anda)
            target_column = 'Class'  # Atau 'isFraud', 'fraud', dll
            
            # Pisahkan features dan target
            X = test_data.drop(target_column, axis=1)
            y = test_data[target_column]
            
            # Handle missing values jika ada
            X = X.fillna(X.mean())
            
            # Feature scaling jika diperlukan
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            logger.info(f"Test dataset loaded: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
            logger.info(f"Fraud ratio: {y.mean():.4f}")
            
            return X_scaled, y.values
            
        except Exception as e:
            logger.error(f"Error loading test dataset: {e}")
            # Fallback ke synthetic data
            return self._generate_synthetic_test_data()

    def _generate_synthetic_test_data(self):
        """Fallback synthetic data jika real data tidak tersedia"""
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_redundant=2,
            n_informative=18,
            n_clusters_per_class=1,
            weights=[0.95, 0.05],
            random_state=42
        )
        return X, y
    
    def _initialize_global_model(self):
        """Initialize global model dengan parameter untuk dataset real"""
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 63,           # Lebih banyak untuk dataset kompleks
            'learning_rate': 0.05,      # Learning rate lebih kecil
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'min_data_in_leaf': 50,     # Untuk mencegah overfitting
            'lambda_l1': 0.1,           # L1 regularization
            'lambda_l2': 0.1,           # L2 regularization
            'is_unbalance': True        # Penting untuk fraud detection
        }
        
        # Load sample data untuk inisialisasi
        try:
            import pandas as pd
            sample_data = pd.read_csv('dataset/train.csv').sample(n=1000, random_state=42)
            target_column = 'Class'
            
            X_sample = sample_data.drop(target_column, axis=1).fillna(0)
            y_sample = sample_data[target_column]
            
            # Feature scaling
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_sample_scaled = scaler.fit_transform(X_sample)
            
            train_data = lgb.Dataset(X_sample_scaled, label=y_sample)
            
        except:
            # Fallback ke synthetic data
            X_dummy, y_dummy = make_classification(n_samples=100, n_features=30, random_state=42)
            train_data = lgb.Dataset(X_dummy, label=y_dummy)
        
        # PERBAIKAN: Hapus verbose_eval dan gunakan callbacks
        self.global_model = lgb.train(
            params,
            train_data,
            num_boost_round=50,
            callbacks=[lgb.log_evaluation(period=0)]  # Silent training
        )
        logger.info("Global model initialized with real dataset parameters")
    
    def _aggregate_models_knowledge_distillation(self, client_models, client_predictions):
        """
        Agregasi model menggunakan Knowledge Distillation dengan Hard Labels
        """
        try:
            # Ensemble predictions dari semua client models
            ensemble_predictions = np.mean(client_predictions, axis=0)
            hard_labels = (ensemble_predictions > 0.5).astype(int)
            
            # Create new training dataset dengan hard labels
            X_distill = self.X_test  # Menggunakan test data untuk distillation
            y_distill = hard_labels
            
            # Train global model dengan hard labels
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42
            }
            
            train_data = lgb.Dataset(X_distill, label=y_distill)
            
            # PERBAIKAN: Hapus verbose_eval dan gunakan callbacks
            self.global_model = lgb.train(
                params,
                train_data,
                num_boost_round=50,
                callbacks=[lgb.log_evaluation(period=0)]  # Silent training
            )
            
            # Evaluate global model
            global_pred = self.global_model.predict(self.X_test)
            global_pred_binary = (global_pred > 0.5).astype(int)
            accuracy = accuracy_score(self.y_test, global_pred_binary)
            
            # Tambahan evaluasi untuk fraud detection
            from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
            
            try:
                auc_score = roc_auc_score(self.y_test, global_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    self.y_test, global_pred_binary, average='binary'
                )
                
                logger.info(f"Global model updated via Knowledge Distillation")
                logger.info(f"Global model metrics - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")
                logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                logger.warning(f"Error calculating additional metrics: {e}")
                logger.info(f"Global model updated via Knowledge Distillation")
                logger.info(f"Global model accuracy: {accuracy:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in model aggregation: {e}")
            return False
    
    def handle_client(self, client_socket, client_address):
        """Handle individual client connection"""
        try:
            logger.info(f"Client connected from {client_address}")
            
            # Send encryption key to client
            client_socket.send(self.encryption_key)
            
            # Send global model to client
            if self.global_model is not None:
                model_data = pickle.dumps(self.global_model)
                encrypted_model = self.cipher.encrypt(model_data)
                
                # Send model size first
                model_size = len(encrypted_model)
                client_socket.send(model_size.to_bytes(4, byteorder='big'))
                
                # Send encrypted model
                client_socket.sendall(encrypted_model)
                logger.info(f"Global model sent to {client_address}")
            
            # Receive client model and predictions
            # Receive data size first
            size_data = client_socket.recv(4)
            if size_data:
                data_size = int.from_bytes(size_data, byteorder='big')
                
                # Receive encrypted data
                encrypted_data = b''
                while len(encrypted_data) < data_size:
                    chunk = client_socket.recv(min(4096, data_size - len(encrypted_data)))
                    if not chunk:
                        break
                    encrypted_data += chunk
                
                # Decrypt and deserialize
                decrypted_data = self.cipher.decrypt(encrypted_data)
                client_data = pickle.loads(decrypted_data)
                
                client_model = client_data['model']
                client_predictions = client_data['predictions']
                client_id = client_data.get('client_id', 'unknown')
                
                self.client_models.append(client_model)
                
                # Store predictions for aggregation
                if not hasattr(self, 'all_predictions'):
                    self.all_predictions = []
                self.all_predictions.append(client_predictions)
                
                logger.info(f"Received model and predictions from client {client_id} at {client_address}")
            
        except Exception as e:
            logger.error(f"Error handling client {client_address}: {e}")
        finally:
            client_socket.close()
    
    def start_server(self):
        """Start federated learning server"""
        self._initialize_global_model()
        
        for round_num in range(self.max_rounds):
            logger.info(f"\n=== FEDERATED LEARNING ROUND {round_num + 1}/{self.max_rounds} ===")
            
            # Reset for new round
            self.client_models = []
            self.all_predictions = []
            
            # Create server socket
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.host, self.port))
            server_socket.listen(self.num_clients)
            
            logger.info(f"Server listening on {self.host}:{self.port}")
            logger.info(f"Waiting for {self.num_clients} clients to connect...")
            
            # Handle client connections
            client_threads = []
            for i in range(self.num_clients):
                try:
                    client_socket, client_address = server_socket.accept()
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, client_address)
                    )
                    client_thread.start()
                    client_threads.append(client_thread)
                except Exception as e:
                    logger.error(f"Error accepting client connection: {e}")
            
            # Wait for all clients to finish
            for thread in client_threads:
                thread.join()
            
            server_socket.close()
            
            # Aggregate models if we have received all client models
            if len(self.client_models) == self.num_clients:
                logger.info("All clients connected. Starting model aggregation...")
                
                if self._aggregate_models_knowledge_distillation(
                    self.client_models, 
                    self.all_predictions
                ):
                    logger.info(f"Round {round_num + 1} completed successfully")
                else:
                    logger.error(f"Round {round_num + 1} failed")
                    break
            else:
                logger.warning(f"Only {len(self.client_models)} out of {self.num_clients} clients connected")
            
            # Wait before next round
            if round_num < self.max_rounds - 1:
                logger.info("Waiting 10 seconds before next round...")
                time.sleep(10)
        
        logger.info("\n=== FEDERATED LEARNING COMPLETED ===")
        
        # Final evaluation
        if self.global_model is not None:
            final_pred = self.global_model.predict(self.X_test)
            final_pred_binary = (final_pred > 0.5).astype(int)
            final_accuracy = accuracy_score(self.y_test, final_pred_binary)
            
            logger.info(f"Final global model accuracy: {final_accuracy:.4f}")
            print(f"\nFinal Classification Report:")
            print(classification_report(self.y_test, final_pred_binary))

if __name__ == "__main__":
    server = FederatedServer(host='localhost', port=8888, num_clients=2)
    server.start_server()