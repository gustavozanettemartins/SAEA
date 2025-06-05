#!/usr/bin/env python3
"""
SAEA - Sistema Autônomo de Alerta de Enchentes
Script de Preprocessamento de Dados

Este script processa dados brutos de múltiplas fontes e prepara
o dataset para treinamento do modelo de Machine Learning.

Autores: [Gustavo Zanette Martins, Michelle Guedes Cavalari]
Data: Junho/2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple, Optional
import os
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SAEAPreprocessor:
    """
    Classe para preprocessamento de dados do SAEA.
    """
    
    def __init__(self, config_path: str = 'config/preprocessing.json'):
        """
        Inicializa o preprocessador.
        
        Args:
            config_path: Caminho para arquivo de configuração
        """
        self.config = self._load_config(config_path)
        self.scaler = None
        self.feature_names = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Carrega configurações do arquivo JSON."""
        default_config = {
            'features_numericas': [
                'precipitation_mm', 'temperature_c', 'humidity_pct',
                'pressure_hpa', 'wind_speed_kmh', 'soil_moisture_pct',
                'nivel_metros', 'taxa_subida_m_h'
            ],
            'janelas_temporais': [24, 48, 72],
            'limites_outliers': {
                'precipitation_mm': [0, 500],
                'temperature_c': [-10, 50],
                'humidity_pct': [0, 100],
                'nivel_metros': [0, 20]
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        
        return default_config
    
    def carregar_dados_brutos(self, 
                             weather_path: str, 
                             river_path: str,
                             disasters_path: str) -> pd.DataFrame:
        """
        Carrega e combina dados de múltiplas fontes.
        
        Args:
            weather_path: Caminho para dados meteorológicos
            river_path: Caminho para dados de níveis de rios
            disasters_path: Caminho para dados do DisasterCharter
            
        Returns:
            DataFrame com dados combinados
        """
        logger.info("Carregando dados brutos...")
        
        # Carregar datasets
        df_weather = pd.read_csv(weather_path)
        df_rivers = pd.read_csv(river_path)
        df_disasters = pd.read_csv(disasters_path)
        
        # Converter datas
        df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
        df_rivers['datetime'] = pd.to_datetime(df_rivers['datetime'])
        df_disasters['date'] = pd.to_datetime(df_disasters['date'])
        
        # Merge datasets
        df_combined = pd.merge(
            df_weather,
            df_rivers[['datetime', 'location', 'nivel_metros', 'taxa_subida_m_h', 'status']],
            on=['datetime', 'location'],
            how='left'
        )
        
        logger.info(f"Dados carregados: {len(df_combined)} registros")
        
        return df_combined, df_disasters
    
    def criar_features_temporais(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features baseadas em padrões temporais.
        
        Args:
            df: DataFrame com dados
            
        Returns:
            DataFrame com novas features
        """
        logger.info("Criando features temporais...")
        
        # Features de tempo
        df['hora'] = df['datetime'].dt.hour
        df['dia_semana'] = df['datetime'].dt.dayofweek
        df['mes'] = df['datetime'].dt.month
        df['dia_mes'] = df['datetime'].dt.day
        df['trimestre'] = df['datetime'].dt.quarter
        
        # Features cíclicas
        df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24)
        df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24)
        df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
        df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
        df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana'] / 7)
        df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana'] / 7)
        
        return df
    
    def criar_features_acumuladas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de valores acumulados e médias móveis.
        
        Args:
            df: DataFrame com dados
            
        Returns:
            DataFrame com features acumuladas
        """
        logger.info("Criando features acumuladas...")
        
        # Agrupar por localização
        for location in df['location'].unique():
            mask = df['location'] == location
            
            # Precipitação acumulada
            for janela in self.config['janelas_temporais']:
                col_name = f'precip_{janela}h'
                if col_name not in df.columns:
                    df.loc[mask, col_name] = df.loc[mask, 'precipitation_mm'].rolling(
                        janela, min_periods=1
                    ).sum()
            
            # Médias móveis
            for col in ['precipitation_mm', 'nivel_metros', 'temperatura_c']:
                if col in df.columns:
                    df.loc[mask, f'{col}_ma_6h'] = df.loc[mask, col].rolling(6).mean()
                    df.loc[mask, f'{col}_ma_12h'] = df.loc[mask, col].rolling(12).mean()
                    df.loc[mask, f'{col}_ma_24h'] = df.loc[mask, col].rolling(24).mean()
            
            # Desvio padrão móvel (volatilidade)
            for col in ['precipitation_mm', 'nivel_metros']:
                if col in df.columns:
                    df.loc[mask, f'{col}_std_12h'] = df.loc[mask, col].rolling(12).std()
                    df.loc[mask, f'{col}_std_24h'] = df.loc[mask, col].rolling(24).std()
            
            # Diferenças (tendências)
            for col in ['precipitation_mm', 'nivel_metros', 'temperature_c']:
                if col in df.columns:
                    df.loc[mask, f'{col}_diff'] = df.loc[mask, col].diff()
                    df.loc[mask, f'{col}_diff_6h'] = df.loc[mask, col].diff(6)
        
        return df
    
    def criar_features_derivadas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features derivadas e índices compostos.
        
        Args:
            df: DataFrame com dados
            
        Returns:
            DataFrame com features derivadas
        """
        logger.info("Criando features derivadas...")
        
        # Taxa de mudança do nível
        df['aceleracao_nivel'] = df.groupby('location')['taxa_subida_m_h'].diff()
        
        # Índice de saturação do solo
        if 'soil_moisture_pct' in df.columns and 'precip_72h' in df.columns:
            df['indice_saturacao'] = (
                df['soil_moisture_pct'] * 0.5 + 
                (df['precip_72h'] / df['precip_72h'].max()) * 50
            )
        
        # Pressão atmosférica normalizada
        if 'pressure_hpa' in df.columns:
            df['pressao_normalizada'] = (
                (df['pressure_hpa'] - df['pressure_hpa'].mean()) / 
                df['pressure_hpa'].std()
            )
        
        # Índice de risco composto
        risk_features = []
        weights = []
        
        if 'precip_48h' in df.columns:
            risk_features.append(df['precip_48h'] / df['precip_48h'].max())
            weights.append(0.3)
        
        if 'nivel_metros' in df.columns:
            risk_features.append(df['nivel_metros'] / df['nivel_metros'].max())
            weights.append(0.3)
        
        if 'taxa_subida_m_h' in df.columns:
            risk_features.append(df['taxa_subida_m_h'].clip(lower=0) / df['taxa_subida_m_h'].max())
            weights.append(0.2)
        
        if 'indice_saturacao' in df.columns:
            risk_features.append(df['indice_saturacao'] / 100)
            weights.append(0.2)
        
        if risk_features:
            # Normalizar pesos
            weights = np.array(weights) / sum(weights)
            df['risco_composto'] = sum(f * w for f, w in zip(risk_features, weights))
        
        # Interações entre features
        if 'precip_48h' in df.columns and 'nivel_metros' in df.columns:
            df['precip_nivel_interaction'] = df['precip_48h'] * df['nivel_metros']
        
        if 'humidity_pct' in df.columns and 'temperature_c' in df.columns:
            df['humidade_temp_interaction'] = df['humidity_pct'] * df['temperature_c']
        
        return df
    
    def marcar_eventos_enchente(self, 
                               df: pd.DataFrame, 
                               df_disasters: pd.DataFrame) -> pd.DataFrame:
        """
        Marca períodos de enchente baseado em eventos reais.
        
        Args:
            df: DataFrame com dados
            df_disasters: DataFrame com eventos de enchente
            
        Returns:
            DataFrame com marcações de enchente
        """
        logger.info("Marcando eventos de enchente...")
        
        # Inicializar colunas
        df['teve_enchente'] = 0
        df['nivel_alerta'] = 0
        df['tempo_ate_enchente'] = np.nan
        
        # Marcar períodos de enchente
        eventos_marcados = 0
        
        for _, evento in df_disasters.iterrows():
            # Período do evento (72h antes até o fim)
            inicio_alerta = evento['date'] - timedelta(hours=72)
            fim_evento = evento['date'] + timedelta(days=evento.get('duration_days', 3))
            
            # Encontrar localização correspondente
            location_match = None
            for loc in df['location'].unique():
                if loc.lower() in evento['location'].lower():
                    location_match = loc
                    break
            
            if location_match:
                mask = (
                    (df['datetime'] >= inicio_alerta) & 
                    (df['datetime'] <= fim_evento) &
                    (df['location'] == location_match)
                )
                
                df.loc[mask, 'teve_enchente'] = 1
                eventos_marcados += 1
                
                # Calcular tempo até o evento
                df.loc[mask, 'tempo_ate_enchente'] = (
                    evento['date'] - df.loc[mask, 'datetime']
                ).dt.total_seconds() / 3600
                
                # Definir níveis de alerta
                df.loc[mask & (df['tempo_ate_enchente'] > 48), 'nivel_alerta'] = 1  # Amarelo
                df.loc[mask & (df['tempo_ate_enchente'] <= 48) & 
                       (df['tempo_ate_enchente'] > 24), 'nivel_alerta'] = 2  # Laranja
                df.loc[mask & (df['tempo_ate_enchente'] <= 24), 'nivel_alerta'] = 3  # Vermelho
        
        logger.info(f"Eventos marcados: {eventos_marcados}")
        logger.info(f"Registros com enchente: {df['teve_enchente'].sum()}")
        
        return df
    
    def remover_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers baseado em limites definidos.
        
        Args:
            df: DataFrame com dados
            
        Returns:
            DataFrame sem outliers extremos
        """
        logger.info("Removendo outliers...")
        
        n_original = len(df)
        
        for col, (min_val, max_val) in self.config['limites_outliers'].items():
            if col in df.columns:
                mask = (df[col] >= min_val) & (df[col] <= max_val)
                df = df[mask]
        
        n_removidos = n_original - len(df)
        logger.info(f"Outliers removidos: {n_removidos} ({n_removidos/n_original*100:.2f}%)")
        
        return df
    
    def normalizar_dados(self, 
                        df: pd.DataFrame, 
                        metodo: str = 'robust') -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Normaliza features numéricas.
        
        Args:
            df: DataFrame com dados
            metodo: 'standard' ou 'robust'
            
        Returns:
            Tupla (DataFrame normalizado, scaler)
        """
        logger.info(f"Normalizando dados com método: {metodo}")
        
        # Selecionar features numéricas
        features_num = [col for col in self.config['features_numericas'] 
                       if col in df.columns]
        
        # Escolher scaler
        if metodo == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        # Normalizar
        df_scaled = df.copy()
        df_scaled[features_num] = self.scaler.fit_transform(df[features_num])
        
        return df_scaled, self.scaler
    
    def preparar_dataset_final(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara dataset final removendo NaNs e selecionando features.
        
        Args:
            df: DataFrame processado
            
        Returns:
            DataFrame pronto para ML
        """
        logger.info("Preparando dataset final...")
        
        # Remover colunas com muitos NaNs
        threshold = 0.3  # 30% de NaNs
        n_rows = len(df)
        
        for col in df.columns:
            if df[col].isna().sum() / n_rows > threshold:
                logger.warning(f"Removendo coluna {col} (>{threshold*100}% NaNs)")
                df = df.drop(columns=[col])
        
        # Preencher NaNs restantes
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Selecionar features finais
        features_manter = [
            # Identificadores
            'datetime', 'location',
            
            # Features originais
            'precipitation_mm', 'temperature_c', 'humidity_pct', 'pressure_hpa',
            'wind_speed_kmh', 'soil_moisture_pct', 'nivel_metros', 'taxa_subida_m_h',
            
            # Features temporais
            'hora', 'dia_semana', 'mes', 'hora_sin', 'hora_cos', 'mes_sin', 'mes_cos',
            
            # Features acumuladas
            'precip_24h', 'precip_48h', 'precip_72h',
            
            # Médias móveis
            'precipitation_mm_ma_12h', 'nivel_metros_ma_12h',
            'precipitation_mm_ma_24h', 'nivel_metros_ma_24h',
            
            # Volatilidade
            'precipitation_mm_std_12h', 'nivel_metros_std_12h',
            
            # Diferenças
            'precipitation_mm_diff', 'nivel_metros_diff',
            
            # Features derivadas
            'aceleracao_nivel', 'indice_saturacao', 'risco_composto',
            'pressao_normalizada',
            
            # Targets
            'teve_enchente', 'nivel_alerta'
        ]
        
        # Manter apenas colunas que existem
        features_finais = [f for f in features_manter if f in df.columns]
        df_final = df[features_finais]
        
        self.feature_names = [f for f in features_finais 
                             if f not in ['datetime', 'location', 'teve_enchente', 'nivel_alerta']]
        
        logger.info(f"Dataset final: {len(df_final)} registros, {len(features_finais)} colunas")
        
        return df_final
    
    def processar_pipeline_completo(self,
                                   weather_path: str,
                                   river_path: str,
                                   disasters_path: str,
                                   output_path: str = 'data/processed/dataset_final.csv'):
        """
        Executa pipeline completo de preprocessamento.
        
        Args:
            weather_path: Caminho dados meteorológicos
            river_path: Caminho dados de rios
            disasters_path: Caminho dados DisasterCharter
            output_path: Caminho para salvar dataset final
        """
        logger.info("=== Iniciando pipeline de preprocessamento ===")
        
        # 1. Carregar dados
        df, df_disasters = self.carregar_dados_brutos(
            weather_path, river_path, disasters_path
        )
        
        # 2. Criar features temporais
        df = self.criar_features_temporais(df)
        
        # 3. Criar features acumuladas
        df = self.criar_features_acumuladas(df)
        
        # 4. Criar features derivadas
        df = self.criar_features_derivadas(df)
        
        # 5. Marcar eventos de enchente
        df = self.marcar_eventos_enchente(df, df_disasters)
        
        # 6. Remover outliers
        df = self.remover_outliers(df)
        
        # 7. Preparar dataset final
        df_final = self.preparar_dataset_final(df)
        
        # 8. Salvar dataset
        df_final.to_csv(output_path, index=False)
        logger.info(f"Dataset salvo em: {output_path}")
        
        # 9. Salvar metadados
        metadata = {
            'data_processamento': datetime.now().isoformat(),
            'n_registros': len(df_final),
            'n_features': len(self.feature_names),
            'features': self.feature_names,
            'targets': ['teve_enchente', 'nivel_alerta'],
            'localizacoes': df_final['location'].unique().tolist(),
            'periodo': {
                'inicio': str(df_final['datetime'].min()),
                'fim': str(df_final['datetime'].max())
            },
            'estatisticas': {
                'enchentes_registradas': int(df_final['teve_enchente'].sum()),
                'taxa_positivos': float(df_final['teve_enchente'].mean()),
                'distribuicao_alertas': df_final['nivel_alerta'].value_counts().to_dict()
            }
        }
        
        metadata_path = output_path.replace('.csv', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadados salvos em: {metadata_path}")
        logger.info("=== Pipeline concluído com sucesso! ===")
        
        return df_final


def main():
    """Função principal para execução do script."""
    # Configurar caminhos
    weather_path = 'data/raw/weather_data.csv'
    river_path = 'data/raw/river_levels.csv'
    disasters_path = 'data/disaster_charter/flood_events.csv'
    output_path = 'data/processed/dataset_ml_final.csv'
    
    # Criar preprocessador
    preprocessor = SAEAPreprocessor()
    
    # Executar pipeline
    df_final = preprocessor.processar_pipeline_completo(
        weather_path,
        river_path,
        disasters_path,
        output_path
    )
    
    # Exibir resumo
    print("\n📊 RESUMO DO PROCESSAMENTO:")
    print(f"Total de registros: {len(df_final):,}")
    print(f"Features criadas: {len(preprocessor.feature_names)}")
    print(f"Taxa de eventos positivos: {df_final['teve_enchente'].mean():.2%}")
    print("\nDistribuição de alertas:")
    print(df_final['nivel_alerta'].value_counts().sort_index())


if __name__ == "__main__":
    main()