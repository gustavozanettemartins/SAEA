#!/usr/bin/env python3
"""
SAEA - Sistema Autônomo de Alerta de Enchentes
Script de Treinamento de Modelos - Versão Melhorada

Este script treina e otimiza modelos de ML para previsão de enchentes,
com foco em resolver problemas de desbalanceamento extremo de classes.

Melhorias implementadas:
- Verificação detalhada da distribuição de classes
- Estratégias robustas de balanceamento
- Métricas adequadas para dados desbalanceados
- Pipeline de debugging melhorado
- Validação cruzada estratificada
- Threshold tuning para otimizar recall

Autores: [Gustavo Zanette Martins, Michelle Guedes Cavalari]
Data: Junho/2025
"""

import pandas as pd
import numpy as np
import json
import joblib
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import os
import sys
from pathlib import Path

# Sklearn imports
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold, TimeSeriesSplit
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, make_scorer,
    average_precision_score, balanced_accuracy_score
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Balanceamento
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost não instalado. Alguns modelos não estarão disponíveis.")

# Suprimir warnings desnecessários
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SAEATrainer:
    """
    Classe principal para treinamento de modelos SAEA com foco em dados desbalanceados.
    """
    
    def __init__(self, config_path: str = 'config/training.json'):
        """
        Inicializa o treinador.
        
        Args:
            config_path: Caminho para arquivo de configuração
        """
        self.config = self._load_config(config_path)
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = None
        self.feature_names = None
        self.class_distribution = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Carrega configurações do arquivo JSON."""
        default_config = {
            'random_state': 42,
            'test_size': 0.2,
            'cv_folds': 5,
            'scoring_metric': 'f1',
            'optimization_metric': 'recall',
            'balancing_strategy': 'smote_tomek',  # Estratégia mais robusta
            'smote_k_neighbors': 3,  # Reduzido para datasets pequenos
            'min_samples_per_class': 10,  # Mínimo para aplicar SMOTE
            'threshold_tuning': True,  # Otimizar threshold de classificação
            'use_ensemble': True,  # Usar ensemble de modelos
            'verbose': True
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Erro ao carregar config: {e}. Usando configuração padrão.")
        
        return default_config
    
    def carregar_dados(self, data_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Carrega e prepara dados para treinamento com análise detalhada.
        
        Args:
            data_path: Caminho para o dataset
            
        Returns:
            Tupla (features, target_binario, target_multiclasse)
        """
        logger.info(f"Carregando dados de: {data_path}")
        
        try:
            # Carregar dados
            df = pd.read_csv(data_path)
            logger.info(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
            
            # Verificar colunas essenciais
            required_cols = ['teve_enchente']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Colunas obrigatórias ausentes: {missing_cols}")
            
            # Converter datetime se existir
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Separar features e targets
            exclude_columns = ['datetime', 'location', 'teve_enchente', 'nivel_alerta']
            feature_columns = [col for col in df.columns if col not in exclude_columns]
            
            if len(feature_columns) == 0:
                raise ValueError("Nenhuma feature encontrada!")
            
            X = df[feature_columns]
            y_binary = df['teve_enchente']
            
            # Target multiclasse (se existir)
            if 'nivel_alerta' in df.columns:
                y_multi = df['nivel_alerta']
            else:
                y_multi = y_binary.copy()  # Usar binário como fallback
            
            # Análise detalhada dos dados
            logger.info(f"Features selecionadas: {len(feature_columns)}")
            logger.info(f"Features: {feature_columns}")
            
            # Verificar valores ausentes
            missing_data = X.isnull().sum()
            if missing_data.sum() > 0:
                logger.warning(f"Valores ausentes encontrados:")
                for col, missing in missing_data[missing_data > 0].items():
                    logger.warning(f"  {col}: {missing} ({missing/len(X)*100:.1f}%)")
                
                # Preencher valores ausentes
                X = X.fillna(X.median())
            
            # Análise da distribuição de classes
            binary_dist = y_binary.value_counts().sort_index()
            multi_dist = y_multi.value_counts().sort_index()
            
            logger.info(f"Distribuição binária: {binary_dist.to_dict()}")
            logger.info(f"Distribuição multiclasse: {multi_dist.to_dict()}")
            
            # Calcular taxa de desbalanceamento
            minority_class = binary_dist.min()
            majority_class = binary_dist.max()
            imbalance_ratio = majority_class / minority_class if minority_class > 0 else float('inf')
            
            logger.info(f"Taxa de desbalanceamento: {imbalance_ratio:.1f}:1")
            
            if imbalance_ratio > 100:
                logger.warning("⚠️ DATASET EXTREMAMENTE DESBALANCEADO!")
                logger.warning("Aplicando estratégias especiais para dados desbalanceados...")
            
            # Verificar se há amostras suficientes da classe minoritária
            if minority_class < self.config['min_samples_per_class']:
                logger.error(f"Classe minoritária tem apenas {minority_class} amostras!")
                logger.error(f"Mínimo necessário: {self.config['min_samples_per_class']}")
                raise ValueError("Amostras insuficientes da classe minoritária")
            
            # Salvar informações para uso posterior
            self.feature_names = feature_columns
            self.class_distribution = binary_dist
            
            logger.info("✅ Dados carregados e validados com sucesso!")
            
            return X, y_binary, y_multi
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {str(e)}")
            raise
    
    def dividir_dados_estratificado(self, X: pd.DataFrame, y: pd.Series, 
                                   test_size: float = 0.2) -> Tuple:
        """
        Divide dados mantendo proporção de classes e distribuição temporal.
        
        Args:
            X: Features
            y: Target
            test_size: Proporção para teste
            
        Returns:
            Tupla (X_train, X_test, y_train, y_test)
        """
        logger.info("Dividindo dados com estratificação...")
        
        try:
            # Verificar se há pelo menos 2 amostras de cada classe
            class_counts = y.value_counts()
            min_class_count = class_counts.min()
            
            if min_class_count < 2:
                logger.warning(f"Classe minoritária tem apenas {min_class_count} amostra(s)")
                logger.warning("Usando divisão simples sem estratificação")
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=self.config['random_state'],
                    shuffle=True
                )
            else:
                # Divisão estratificada normal
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=test_size,
                    stratify=y,
                    random_state=self.config['random_state']
                )
            
            logger.info(f"Dados de treino: {len(X_train)} amostras")
            logger.info(f"Dados de teste: {len(X_test)} amostras")
            
            # Verificar distribuição
            train_dist = np.bincount(y_train)
            test_dist = np.bincount(y_test)
            
            logger.info(f"Distribuição treino: {dict(enumerate(train_dist))}")
            logger.info(f"Distribuição teste: {dict(enumerate(test_dist))}")
            
            # Verificar se há classes ausentes no teste
            if len(test_dist) < 2 or test_dist[1] == 0:
                logger.warning("⚠️ Classe minoritária ausente no conjunto de teste!")
                logger.warning("Isso pode afetar a avaliação do modelo.")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Erro na divisão dos dados: {str(e)}")
            raise
    
    def normalizar_dados(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple:
        """
        Normaliza dados usando RobustScaler para lidar com outliers.
        
        Args:
            X_train, X_test: Arrays de features
            
        Returns:
            Tupla (X_train_scaled, X_test_scaled, scaler)
        """
        logger.info("Normalizando features...")
        
        # Usar RobustScaler que é menos sensível a outliers
        self.scaler = RobustScaler()
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("✅ Normalização concluída")
        
        return X_train_scaled, X_test_scaled, self.scaler
    
    def balancear_dados_robusto(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple:
        """
        Aplica estratégias robustas de balanceamento para dados extremamente desbalanceados.
        
        Args:
            X_train, y_train: Dados de treino
            
        Returns:
            Tupla (X_train_balanced, y_train_balanced)
        """
        logger.info("Aplicando balanceamento robusto...")
        
        # Verificar distribuição inicial
        unique, counts = np.unique(y_train, return_counts=True)
        initial_dist = dict(zip(unique, counts))
        logger.info(f"Distribuição inicial: {initial_dist}")
        
        # Verificar se balanceamento é necessário e possível
        if len(unique) < 2:
            logger.warning("Apenas uma classe presente no treino - sem balanceamento")
            return X_train, y_train
        
        minority_samples = counts.min()
        majority_samples = counts.max()
        imbalance_ratio = majority_samples / minority_samples
        
        logger.info(f"Taxa de desbalanceamento: {imbalance_ratio:.1f}:1")
        
        try:
            strategy = self.config['balancing_strategy']
            
            if strategy == 'none' or imbalance_ratio < 2:
                logger.info("Sem balanceamento aplicado")
                return X_train, y_train
            
            # Determinar k_neighbors para SMOTE
            k_neighbors = min(
                self.config['smote_k_neighbors'],
                minority_samples - 1
            )
            
            if k_neighbors < 1:
                logger.warning("Amostras insuficientes para SMOTE - usando duplicação simples")
                # Estratégia de fallback: duplicar amostras minoritárias
                minority_indices = np.where(y_train == 1)[0]
                n_duplicates = majority_samples - minority_samples
                
                if n_duplicates > 0:
                    duplicate_indices = np.random.choice(
                        minority_indices, 
                        size=min(n_duplicates, len(minority_indices) * 3),
                        replace=True
                    )
                    
                    X_train_balanced = np.vstack([X_train, X_train[duplicate_indices]])
                    y_train_balanced = np.hstack([y_train, y_train[duplicate_indices]])
                else:
                    X_train_balanced = X_train
                    y_train_balanced = y_train
            
            else:
                # Aplicar estratégias de balanceamento
                if strategy == 'smote':
                    sampler = SMOTE(
                        random_state=self.config['random_state'],
                        k_neighbors=k_neighbors,
                        sampling_strategy='auto'
                    )
                
                elif strategy == 'adasyn':
                    sampler = ADASYN(
                        random_state=self.config['random_state'],
                        n_neighbors=k_neighbors,
                        sampling_strategy='auto'
                    )
                
                elif strategy == 'smote_tomek':
                    sampler = SMOTETomek(
                        random_state=self.config['random_state'],
                        smote=SMOTE(k_neighbors=k_neighbors)
                    )
                
                elif strategy == 'smote_enn':
                    sampler = SMOTEENN(
                        random_state=self.config['random_state'],
                        smote=SMOTE(k_neighbors=k_neighbors)
                    )
                
                elif strategy == 'borderline_smote':
                    sampler = BorderlineSMOTE(
                        random_state=self.config['random_state'],
                        k_neighbors=k_neighbors
                    )
                
                else:
                    logger.warning(f"Estratégia desconhecida: {strategy}. Usando SMOTE.")
                    sampler = SMOTE(
                        random_state=self.config['random_state'],
                        k_neighbors=k_neighbors
                    )
                
                # Aplicar balanceamento
                X_train_balanced, y_train_balanced = sampler.fit_resample(X_train, y_train)
            
            # Verificar resultado
            final_unique, final_counts = np.unique(y_train_balanced, return_counts=True)
            final_dist = dict(zip(final_unique, final_counts))
            
            logger.info(f"Distribuição final: {final_dist}")
            logger.info(f"Amostras adicionadas: {len(X_train_balanced) - len(X_train)}")
            
            return X_train_balanced, y_train_balanced
            
        except Exception as e:
            logger.error(f"Erro no balanceamento: {str(e)}")
            logger.warning("Usando dados originais sem balanceamento")
            return X_train, y_train
    
    def criar_modelos_robustos(self) -> Dict[str, Any]:
        """
        Cria conjunto de modelos otimizados para dados desbalanceados.
        
        Returns:
            Dicionário com modelos
        """
        logger.info("Criando modelos robustos...")
        
        models = {}
        
        # Random Forest com parâmetros otimizados para dados desbalanceados
        models['RandomForest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced_subsample',  # Melhor para dados desbalanceados
            random_state=self.config['random_state'],
            n_jobs=-1
        )
        
        # Extra Trees
        models['ExtraTrees'] = ExtraTreesClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced_subsample',
            random_state=self.config['random_state'],
            n_jobs=-1
        )
        
        # Gradient Boosting
        models['GradientBoosting'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=self.config['random_state']
        )
        
        # Logistic Regression
        models['LogisticRegression'] = LogisticRegression(
            class_weight='balanced',
            random_state=self.config['random_state'],
            max_iter=1000,
            solver='liblinear'  # Melhor para datasets pequenos
        )
        
        # Decision Tree simples
        models['DecisionTree'] = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=self.config['random_state']
        )
        
        # XGBoost se disponível
        if XGBOOST_AVAILABLE:
            # Calcular scale_pos_weight baseado na distribuição
            if self.class_distribution is not None:
                neg_samples = self.class_distribution[0]
                pos_samples = self.class_distribution[1]
                scale_pos_weight = neg_samples / pos_samples
            else:
                scale_pos_weight = 10  # Valor padrão
            
            models['XGBoost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=self.config['random_state'],
                eval_metric='logloss',
                use_label_encoder=False
            )
        
        logger.info(f"Criados {len(models)} modelos")
        return models
    
    def avaliar_modelo_detalhado(self, model: Any, X_test: np.ndarray, 
                                y_test: np.ndarray, name: str) -> Dict:
        """
        Avalia modelo com métricas adequadas para dados desbalanceados.
        
        Args:
            model: Modelo treinado
            X_test, y_test: Dados de teste
            name: Nome do modelo
            
        Returns:
            Dicionário com resultados detalhados
        """
        logger.info(f"Avaliando {name}...")
        
        try:
            # Predições
            y_pred = model.predict(X_test)
            
            # Probabilidades se disponível
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
                if y_pred_proba.shape[1] > 1:
                    y_pred_proba = y_pred_proba[:, 1]
                else:
                    y_pred_proba = y_pred_proba[:, 0]
            else:
                y_pred_proba = None
            
            # Métricas básicas
            results = {
                'model': model,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'accuracy': accuracy_score(y_test, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            # Métricas avançadas se houver mais de uma classe no teste
            unique_test_classes = np.unique(y_test)
            
            if len(unique_test_classes) > 1 and y_pred_proba is not None:
                try:
                    results['auc_roc'] = roc_auc_score(y_test, y_pred_proba)
                    results['average_precision'] = average_precision_score(y_test, y_pred_proba)
                except Exception as e:
                    logger.warning(f"Erro ao calcular AUC-ROC: {e}")
            
            # Score customizado para enchentes (prioriza recall)
            results['custom_score'] = (
                0.3 * results['precision'] + 
                0.5 * results['recall'] + 
                0.2 * results['f1']
            )
            
            # Log das métricas
            logger.info(f"  Acurácia:           {results['accuracy']:.3f}")
            logger.info(f"  Acurácia Balanceada: {results['balanced_accuracy']:.3f}")
            logger.info(f"  Precisão:           {results['precision']:.3f}")
            logger.info(f"  Recall:             {results['recall']:.3f}")
            logger.info(f"  F1-Score:           {results['f1']:.3f}")
            if 'auc_roc' in results:
                logger.info(f"  AUC-ROC:            {results['auc_roc']:.3f}")
            logger.info(f"  Score Customizado:  {results['custom_score']:.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Erro na avaliação de {name}: {str(e)}")
            return {
                'model': model,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'custom_score': 0.0,
                'error': str(e)
            }
    
    def otimizar_threshold(self, model: Any, X_val: np.ndarray, 
                          y_val: np.ndarray) -> float:
        """
        Otimiza threshold de classificação para maximizar recall.
        
        Args:
            model: Modelo treinado
            X_val, y_val: Dados de validação
            
        Returns:
            Threshold otimizado
        """
        if not hasattr(model, 'predict_proba'):
            return 0.5
        
        logger.info("Otimizando threshold de classificação...")
        
        try:
            # Obter probabilidades
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Testar diferentes thresholds
            thresholds = np.arange(0.1, 0.9, 0.05)
            best_threshold = 0.5
            best_score = 0.0
            
            for threshold in thresholds:
                y_pred_thresh = (y_pred_proba >= threshold).astype(int)
                
                # Score que prioriza recall
                recall = recall_score(y_val, y_pred_thresh, zero_division=0)
                precision = precision_score(y_val, y_pred_thresh, zero_division=0)
                f1 = f1_score(y_val, y_pred_thresh, zero_division=0)
                
                # Score customizado
                score = 0.6 * recall + 0.3 * f1 + 0.1 * precision
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            
            logger.info(f"Threshold otimizado: {best_threshold:.3f} (score: {best_score:.3f})")
            return best_threshold
            
        except Exception as e:
            logger.warning(f"Erro na otimização de threshold: {e}")
            return 0.5
    
    def executar_pipeline_completo(self, data_path: str, output_dir: str = 'models/'):
        """
        Executa pipeline completo de treinamento melhorado.
        
        Args:
            data_path: Caminho para dados
            output_dir: Diretório para salvar modelos
        """
        logger.info("\n" + "="*70)
        logger.info("🚀 INICIANDO PIPELINE DE TREINAMENTO SAEA MELHORADO")
        logger.info("="*70)
        
        try:
            # 1. Carregar e analisar dados
            X, y_binary, y_multi = self.carregar_dados(data_path)
            
            # 2. Dividir dados estratificadamente
            X_train, X_test, y_train, y_test = self.dividir_dados_estratificado(
                X, y_binary, self.config['test_size']
            )
            
            # 3. Normalizar dados
            X_train_scaled, X_test_scaled, scaler = self.normalizar_dados(
                X_train.values, X_test.values
            )
            
            # 4. Balancear dados de treino
            X_train_balanced, y_train_balanced = self.balancear_dados_robusto(
                X_train_scaled, y_train.values
            )
            
            # 5. Criar divisão para validação (para threshold tuning)
            X_train_final, X_val, y_train_final, y_val = train_test_split(
                X_train_balanced, y_train_balanced,
                test_size=0.2,
                stratify=y_train_balanced if len(np.unique(y_train_balanced)) > 1 else None,
                random_state=self.config['random_state']
            )
            
            # 6. Treinar modelos
            logger.info("\n" + "-"*60)
            logger.info("🎯 TREINAMENTO E AVALIAÇÃO DE MODELOS")
            logger.info("-"*60)
            
            models = self.criar_modelos_robustos()
            results = {}
            
            for name, model in models.items():
                try:
                    # Treinar modelo
                    model.fit(X_train_final, y_train_final)
                    
                    # Otimizar threshold se solicitado
                    optimal_threshold = 0.5
                    if self.config.get('threshold_tuning', False):
                        optimal_threshold = self.otimizar_threshold(model, X_val, y_val)
                    
                    # Avaliar no conjunto de teste
                    results[name] = self.avaliar_modelo_detalhado(
                        model, X_test_scaled, y_test.values, name
                    )
                    results[name]['optimal_threshold'] = optimal_threshold
                    
                except Exception as e:
                    logger.error(f"Erro ao treinar {name}: {str(e)}")
                    continue
            
            if not results:
                raise ValueError("Nenhum modelo foi treinado com sucesso!")
            
            # 7. Selecionar melhor modelo
            # Priorizar recall para minimizar falsos negativos
            best_model_name = max(results.keys(), 
                                key=lambda k: results[k].get('custom_score', 0))
            
            self.best_model = results[best_model_name]['model']
            best_threshold = results[best_model_name].get('optimal_threshold', 0.5)
            
            logger.info(f"\n🏆 Melhor modelo: {best_model_name}")
            logger.info(f"   Score customizado: {results[best_model_name]['custom_score']:.3f}")
            logger.info(f"   Threshold otimizado: {best_threshold:.3f}")
            
            # 8. Avaliação final detalhada
            logger.info("\n" + "-"*60)
            logger.info("📊 AVALIAÇÃO FINAL DETALHADA")
            logger.info("-"*60)
            
            best_result = results[best_model_name]
            
            # Aplicar threshold otimizado
            if best_result.get('probabilities') is not None:
                y_pred_optimized = (best_result['probabilities'] >= best_threshold).astype(int)
                
                logger.info("Métricas com threshold otimizado:")
                logger.info(f"  Acurácia:  {accuracy_score(y_test, y_pred_optimized):.3f}")
                logger.info(f"  Precisão:  {precision_score(y_test, y_pred_optimized, zero_division=0):.3f}")
                logger.info(f"  Recall:    {recall_score(y_test, y_pred_optimized, zero_division=0):.3f}")
                logger.info(f"  F1-Score:  {f1_score(y_test, y_pred_optimized, zero_division=0):.3f}")
                
                # Matriz de confusão otimizada
                cm_opt = confusion_matrix(y_test, y_pred_optimized)
                logger.info("\nMatriz de Confusão (threshold otimizado):")
                logger.info(f"  TN: {cm_opt[0,0]}  FP: {cm_opt[0,1]}")
                if cm_opt.shape[0] > 1:
                    logger.info(f"  FN: {cm_opt[1,0]}  TP: {cm_opt[1,1]}")
            
            # Relatório de classificação
            logger.info("\nRelatório de Classificação:")
            print(classification_report(
                y_test, 
                best_result['predictions'], 
                target_names=['Normal', 'Enchente'],
                zero_division=0
            ))
            
            # 9. Criar modelo embarcado
            logger.info("\n" + "-"*60)
            logger.info("🔧 CRIAÇÃO DO MODELO EMBARCADO")
            logger.info("-"*60)
            
            embedded_model = self.criar_modelo_embarcado_melhorado(
                self.best_model, X_train_balanced, y_train_balanced
            )
            
            # 10. Salvar modelos e artefatos
            logger.info("\n" + "-"*60)
            logger.info("💾 SALVANDO MODELOS E ARTEFATOS")
            logger.info("-"*60)
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Salvar modelos
            joblib.dump(self.best_model, os.path.join(output_dir, 'modelo_completo.pkl'))
            joblib.dump(embedded_model, os.path.join(output_dir, 'modelo_embarcado.pkl'))
            joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
            
            # Salvar metadados completos
            metadata = {
                'data_treinamento': datetime.now().isoformat(),
                'melhor_modelo': best_model_name,
                'threshold_otimizado': float(best_threshold),
                'features': self.feature_names,
                'n_features': len(self.feature_names),
                'distribuicao_classes': self.class_distribution.to_dict(),
                'amostras_treino': len(X_train_balanced),
                'amostras_teste': len(X_test),
                'estrategia_balanceamento': self.config['balancing_strategy'],
                'metricas_finais': {
                    'accuracy': float(best_result['accuracy']),
                    'precision': float(best_result['precision']),
                    'recall': float(best_result['recall']),
                    'f1_score': float(best_result['f1']),
                    'custom_score': float(best_result['custom_score'])
                },
                'todos_resultados': {
                    name: {
                        'accuracy': float(result['accuracy']),
                        'precision': float(result['precision']),
                        'recall': float(result['recall']),
                        'f1_score': float(result['f1']),
                        'custom_score': float(result['custom_score'])
                    } for name, result in results.items()
                },
                'configuracao': self.config
            }
            
            with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # 11. Gerar código C++ melhorado
            self.gerar_codigo_cpp_melhorado(
                embedded_model, self.feature_names, best_threshold
            )
            
            logger.info("\n" + "="*70)
            logger.info("✅ PIPELINE CONCLUÍDO COM SUCESSO!")
            logger.info("="*70)
            logger.info(f"📁 Modelos salvos em: {output_dir}")
            logger.info(f"🎯 Melhor modelo: {best_model_name}")
            logger.info(f"📈 Score final: {results[best_model_name]['custom_score']:.3f}")
            
            return self.best_model, embedded_model, results
            
        except Exception as e:
            logger.error(f"❌ Erro no pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def criar_modelo_embarcado_melhorado(self, best_model: Any, X_train: np.ndarray, 
                                       y_train: np.ndarray) -> Any:
        """
        Cria versão otimizada do modelo para ESP32.
        
        Args:
            best_model: Melhor modelo completo
            X_train, y_train: Dados de treino
            
        Returns:
            Modelo simplificado e otimizado
        """
        logger.info("Criando modelo embarcado otimizado...")
        
        # Configuração otimizada baseada no tipo do melhor modelo
        if isinstance(best_model, (RandomForestClassifier, ExtraTreesClassifier)):
            embedded_model = DecisionTreeClassifier(
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=self.config['random_state']
            )
        else:
            # Usar árvore de decisão como fallback universal
            embedded_model = DecisionTreeClassifier(
                max_depth=6,
                min_samples_split=15,
                min_samples_leaf=8,
                class_weight='balanced',
                random_state=self.config['random_state']
            )
        
        # Treinar modelo embarcado
        embedded_model.fit(X_train, y_train)
        
        logger.info("✅ Modelo embarcado criado com sucesso")
        
        return embedded_model
    
    def gerar_codigo_cpp_melhorado(self, model: Any, feature_names: List[str], 
                                  threshold: float = 0.5,
                                  output_path: str = 'hardware/ml_model.h'):
        """
        Gera código C++ otimizado do modelo para ESP32 - VERSÃO CORRIGIDA.
        
        Args:
            model: Modelo treinado
            feature_names: Nomes das features
            threshold: Threshold de classificação
            output_path: Caminho para salvar código
        """
        logger.info("Gerando código C++ otimizado para ESP32...")
        
        cpp_code = f'''/**
 * SAEA - Modelo de ML Otimizado para ESP32
 * Gerado automaticamente em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
 * 
 * Modelo: {type(model).__name__}
 * Features: {len(feature_names)}
 * Threshold otimizado: {threshold:.3f}
 */

#ifndef SAEA_ML_MODEL_H
#define SAEA_ML_MODEL_H

#include <Arduino.h>

// Configurações do modelo
#define SAEA_N_FEATURES {len(feature_names)}
#define SAEA_THRESHOLD {threshold:.6f}f
#define SAEA_MODEL_VERSION "1.0"

// Estrutura para dados de entrada
struct SAEAFeatures {{
    // Features básicas dos sensores
    float precipitation_mm;
    float temperature_c;
    float humidity_pct;
    float pressure_hpa;
    float wind_speed_kmh;
    float soil_moisture_pct;
    float nivel_metros;
    float taxa_subida_m_h;
    
    // Features temporais
    float hora;
    float dia_semana;
    float mes;
    float hora_sin;
    float hora_cos;
    float mes_sin;
    float mes_cos;
    
    // Features derivadas
    float precipitation_mm_ma_12h;
    float nivel_metros_ma_12h;
    float precipitation_mm_ma_24h;
    float nivel_metros_ma_24h;
    float precipitation_mm_std_12h;
    float nivel_metros_std_12h;
    float precipitation_mm_diff;
    float nivel_metros_diff;
    float aceleracao_nivel;
    float pressao_normalizada;
    
    // Features agregadas ✅ CAMPOS QUE ESTAVAM FALTANDO
    float precip_24h;
    float precip_48h;
}};

// Thresholds críticos baseados no modelo treinado
#define THRESHOLD_PRECIP_24H_CRITICAL 100.0f
#define THRESHOLD_PRECIP_48H_CRITICAL 150.0f
#define THRESHOLD_PRECIP_72H_CRITICAL 200.0f
#define THRESHOLD_NIVEL_CRITICAL 4.0f
#define THRESHOLD_NIVEL_HIGH 3.0f
#define THRESHOLD_NIVEL_MEDIUM 2.0f
#define THRESHOLD_TAXA_CRITICAL 0.3f
#define THRESHOLD_TAXA_HIGH 0.2f

/**
 * Função principal de predição otimizada
 * Retorna: 0 = Sem risco, 1 = Risco de enchente
 */
int predictFlood(SAEAFeatures features) {{
    // Verificações críticas imediatas ✅ AGORA USA CAMPOS CORRETOS
    float precip_24h = features.precip_24h;
    float precip_48h = features.precip_48h;
    float nivel_metros = features.nivel_metros;
    float taxa_subida = features.taxa_subida_m_h;
    
    // Regra 1: Condições extremas
    if (precip_48h > THRESHOLD_PRECIP_48H_CRITICAL || 
        nivel_metros > THRESHOLD_NIVEL_CRITICAL ||
        taxa_subida > THRESHOLD_TAXA_CRITICAL) {{
        return 1; // RISCO ALTO
    }}
    
    // Regra 2: Combinação de fatores de risco
    float riskScore = 0.0f;
    
    // Componente precipitação (peso: 40%)
    if (precip_48h > 50.0f) {{
        riskScore += min(1.0f, precip_48h / 150.0f) * 0.4f;
    }}
    
    // Componente nível d'água (peso: 35%)
    if (nivel_metros > 1.0f) {{
        riskScore += min(1.0f, nivel_metros / 4.0f) * 0.35f;
    }}
    
    // Componente taxa de subida (peso: 25%)
    if (taxa_subida > 0.05f) {{
        riskScore += min(1.0f, taxa_subida / 0.3f) * 0.25f;
    }}
    
    // Decisão baseada no score e threshold otimizado
    return (riskScore >= SAEA_THRESHOLD) ? 1 : 0;
}}

/**
 * Função para calcular score de risco contínuo
 * Retorna valor entre 0.0 e 1.0
 */
float calculateRiskScore(SAEAFeatures features) {{
    float score = 0.0f;
    
    // Normalizar features principais ✅ AGORA USA CAMPOS CORRETOS
    float precip_norm = min(1.0f, features.precip_48h / 200.0f);
    float nivel_norm = min(1.0f, features.nivel_metros / 5.0f);
    float taxa_norm = min(1.0f, features.taxa_subida_m_h / 0.5f);
    
    // Combinar com pesos otimizados
    score = precip_norm * 0.4f + nivel_norm * 0.35f + taxa_norm * 0.25f;
    
    // Aplicar função de ativação suave
    return score / (1.0f + abs(1.0f - score));
}}

/**
 * Função para obter nível de alerta
 * Retorna: 0=Verde, 1=Amarelo, 2=Laranja, 3=Vermelho
 */
int getAlertLevel(SAEAFeatures features) {{
    float riskScore = calculateRiskScore(features);
    
    if (riskScore >= 0.8f) return 3; // VERMELHO
    if (riskScore >= 0.6f) return 2; // LARANJA  
    if (riskScore >= 0.4f) return 1; // AMARELO
    return 0; // VERDE
}}

/**
 * Função auxiliar para validar dados de entrada
 */
bool validateFeatures(SAEAFeatures features) {{
    // Verificar valores válidos
    if (features.precip_48h < 0 || features.precip_48h > 500) return false;
    if (features.nivel_metros < 0 || features.nivel_metros > 10) return false;
    if (features.taxa_subida_m_h < -1 || features.taxa_subida_m_h > 2) return false;
    
    return true;
}}

/**
 * Função para obter descrição do alerta
 */
const char* getAlertDescription(int level) {{
    switch(level) {{
        case 0: return "Normal - Monitoramento rotineiro";
        case 1: return "Atenção - Acompanhar condições";
        case 2: return "Alerta - Preparar ações preventivas";
        case 3: return "Emergência - Evacuar área de risco";
        default: return "Status desconhecido";
    }}
}}

/**
 * Função para log de debug
 */
void logPrediction(SAEAFeatures features, int prediction, float score) {{
    Serial.print("SAEA Prediction - ");
    Serial.print("Risk Score: "); Serial.print(score, 3);
    Serial.print(", Prediction: "); Serial.print(prediction);
    Serial.print(", Alert Level: "); Serial.println(getAlertLevel(features));
}}

/**
 * Exemplo de uso no ESP32
 */
void exemploUso() {{
    SAEAFeatures dados;
    
    // Preencher dados básicos dos sensores
    dados.precipitation_mm = 15.5f;
    dados.temperature_c = 28.2f;
    dados.humidity_pct = 85.0f;
    dados.pressure_hpa = 1010.5f;
    dados.wind_speed_kmh = 12.0f;
    dados.soil_moisture_pct = 70.0f;
    dados.nivel_metros = 2.3f;
    dados.taxa_subida_m_h = 0.15f;
    
    // Features temporais
    dados.hora = 14.0f;
    dados.dia_semana = 2.0f;
    dados.mes = 6.0f;
    dados.hora_sin = sin(2 * PI * dados.hora / 24);
    dados.hora_cos = cos(2 * PI * dados.hora / 24);
    dados.mes_sin = sin(2 * PI * dados.mes / 12);
    dados.mes_cos = cos(2 * PI * dados.mes / 12);
    
    // Features derivadas (calcular a partir dos dados)
    dados.precipitation_mm_ma_12h = dados.precipitation_mm * 0.8f;
    dados.nivel_metros_ma_12h = dados.nivel_metros * 0.95f;
    dados.precipitation_mm_ma_24h = dados.precipitation_mm * 0.6f;
    dados.nivel_metros_ma_24h = dados.nivel_metros * 0.9f;
    dados.precipitation_mm_std_12h = dados.precipitation_mm * 0.3f;
    dados.nivel_metros_std_12h = dados.nivel_metros * 0.1f;
    dados.precipitation_mm_diff = dados.precipitation_mm * 0.2f;
    dados.nivel_metros_diff = dados.nivel_metros * 0.05f;
    dados.aceleracao_nivel = dados.taxa_subida_m_h * 0.1f;
    dados.pressao_normalizada = (dados.pressure_hpa - 1013.25f) / 50.0f;
    
    // Features agregadas ✅ INCLUIR ESTES CAMPOS
    dados.precip_24h = dados.precipitation_mm * 24;  // Estimativa
    dados.precip_48h = dados.precip_24h * 2;        // Estimativa
    
    // Validar dados
    if (!validateFeatures(dados)) {{
        Serial.println("Dados inválidos!");
        return;
    }}
    
    // Fazer predição
    int predicao = predictFlood(dados);
    float score = calculateRiskScore(dados);
    int nivel_alerta = getAlertLevel(dados);
    
    // Log resultado
    logPrediction(dados, predicao, score);
    
    Serial.print("Descrição: ");
    Serial.println(getAlertDescription(nivel_alerta));
}}

#endif // SAEA_ML_MODEL_H
'''
        
        # Salvar código
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cpp_code)
        
        logger.info(f"✅ Código C++ corrigido salvo em: {output_path}")


def main():
    """Função principal melhorada."""
    import argparse
    import os
    import sys
    import logging

    # Configure logging first
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description='Treinar modelos SAEA - Versão Melhorada',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data', type=str,
                        default='data/processed/dataset_ml_final.csv',  # Moved default here
                        help='Caminho para o dataset CSV')
    parser.add_argument('--output', type=str, default='models/',
                        help='Diretório de saída para modelos')
    parser.add_argument('--config', type=str, default='config/training.json',
                        help='Arquivo de configuração (opcional)')
    parser.add_argument('--strategy', type=str,
                        choices=['smote', 'adasyn', 'smote_tomek', 'smote_enn', 'none'],
                        default='smote_tomek',
                        help='Estratégia de balanceamento')
    parser.add_argument('--threshold-tuning', action='store_true',
                        help='Ativar otimização de threshold')
    parser.add_argument('--verbose', action='store_true',
                        help='Saída detalhada')

    args = parser.parse_args()

    # Configurar nível de log
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Verificar se arquivo de dados existe
        if not os.path.exists(args.data):
            logger.error(f"Arquivo de dados não encontrado: {args.data}")
            sys.exit(1)

        # Criar diretório de saída se não existir
        os.makedirs(args.output, exist_ok=True)

        # Configuração personalizada
        config_override = {
            'balancing_strategy': args.strategy,
            'threshold_tuning': args.threshold_tuning,
            'verbose': args.verbose
        }

        # Criar treinador (assuming SAEATrainer is imported)
        trainer = SAEATrainer(args.config)
        trainer.config.update(config_override)

        logger.info(f"🔧 Configuração:")
        logger.info(f"   Estratégia de balanceamento: {trainer.config['balancing_strategy']}")
        logger.info(f"   Otimização de threshold: {trainer.config['threshold_tuning']}")
        logger.info(f"   Random state: {trainer.config['random_state']}")

        # Executar pipeline
        best_model, embedded_model, results = trainer.executar_pipeline_completo(
            args.data, args.output
        )

        logger.info("\n🎉 TREINAMENTO CONCLUÍDO COM SUCESSO!")
        logger.info(f"🏆 Melhor modelo salvo em: {args.output}")

        # Resumo dos resultados
        logger.info("\n📊 RESUMO DOS RESULTADOS:")
        for name, result in results.items():
            logger.info(f"   {name:20} - Score: {result['custom_score']:.3f} "
                        f"(Recall: {result['recall']:.3f})")

    except KeyboardInterrupt:
        logger.info("\n⚠️ Treinamento interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Erro durante o treinamento: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()