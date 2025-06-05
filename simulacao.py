#!/usr/bin/env python3
"""
SAEA - Sistema de Simulação Completa
Simula diferentes cenários de enchentes para testar o modelo treinado

Este script permite:
1. Simular dados de sensores em tempo real
2. Testar diferentes cenários de enchente
3. Validar alertas e thresholds
4. Simular comportamento do ESP32
5. Gerar relatórios de performance

Autores: [Gustavo Zanette Martins, Michelle Guedes Cavalari]
Data: Junho/2025
"""

import pandas as pd
import numpy as np
import joblib
import json
import time
import random
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SAEASimulator:
    """
    Simulador completo do sistema SAEA para testes e validação.
    """
    
    def __init__(self, models_dir='models/'):
        """
        Inicializa o simulador.
        
        Args:
            models_dir: Diretório dos modelos treinados
        """
        self.models_dir = models_dir
        self.modelo = None
        self.scaler = None
        self.metadata = None
        self.feature_names = None
        
        # Configurações de simulação
        self.thresholds = {
            'emergencia': 0.05,    # Máxima sensibilidade
            'urbano': 0.50,        # Balanceado
            'conservador': 0.80    # Mínimos alarmes falsos
        }
        
        # Estado atual da simulação
        self.current_state = {
            'datetime': datetime.now(),
            'location': 'Simulação',
            'sensors': {},
            'predictions': [],
            'alerts': []
        }
        
        self.carregar_modelos()
    
    def carregar_modelos(self):
        """Carrega modelos e metadados."""
        try:
            print("🔄 Carregando modelos...")
            self.modelo = joblib.load(f'{self.models_dir}/modelo_completo.pkl')
            self.scaler = joblib.load(f'{self.models_dir}/scaler.pkl')
            
            with open(f'{self.models_dir}/metadata.json', 'r') as f:
                self.metadata = json.load(f)
            
            self.feature_names = self.metadata.get('features', [])
            print(f"✅ Modelos carregados: {len(self.feature_names)} features")
            
        except Exception as e:
            print(f"❌ Erro ao carregar modelos: {e}")
            raise
    
    def gerar_dados_baseline(self) -> Dict:
        """
        Gera dados baseline (condições normais) baseados no dataset real.
        
        Returns:
            Dicionário com valores baseline dos sensores
        """
        # Valores típicos baseados na análise do dataset
        baseline = {
            'precipitation_mm': np.random.exponential(2.0),  # Precipitação baixa
            'temperature_c': np.random.normal(22.0, 5.0),   # Temperatura média
            'humidity_pct': np.random.normal(65.0, 15.0),   # Umidade média
            'pressure_hpa': np.random.normal(1013.0, 10.0), # Pressão atmosférica
            'wind_speed_kmh': np.random.exponential(8.0),   # Vento baixo
            'soil_moisture_pct': np.random.normal(40.0, 10.0), # Umidade solo
            'nivel_metros': np.random.normal(1.5, 0.3),     # Nível normal do rio
            'taxa_subida_m_h': np.random.normal(0.0, 0.05)  # Taxa de subida baixa
        }
        
        # Garantir valores realistas
        baseline['humidity_pct'] = np.clip(baseline['humidity_pct'], 0, 100)
        baseline['soil_moisture_pct'] = np.clip(baseline['soil_moisture_pct'], 0, 100)
        baseline['nivel_metros'] = np.clip(baseline['nivel_metros'], 0.5, 10.0)
        baseline['temperature_c'] = np.clip(baseline['temperature_c'], -5, 45)
        
        return baseline
    
    def simular_cenario_enchente(self, intensidade='moderada') -> Dict:
        """
        Simula cenário de enchente com diferentes intensidades.
        
        Args:
            intensidade: 'leve', 'moderada', 'severa', 'extrema'
            
        Returns:
            Dicionário com dados de sensores simulando enchente
        """
        # Começar com baseline
        dados = self.gerar_dados_baseline()
        
        # Configurações por intensidade
        configs = {
            'leve': {
                'precip_mult': (2, 4),        # 2-4x a precipitação normal
                'nivel_add': (0.5, 1.0),     # +0.5-1.0m no nível
                'taxa_mult': (5, 10),        # 5-10x taxa de subida
                'humidity_add': (10, 20),    # +10-20% umidade
                'pressure_sub': (5, 15)      # -5-15 hPa pressão
            },
            'moderada': {
                'precip_mult': (4, 8),
                'nivel_add': (1.0, 2.0),
                'taxa_mult': (10, 20),
                'humidity_add': (15, 25),
                'pressure_sub': (10, 25)
            },
            'severa': {
                'precip_mult': (8, 15),
                'nivel_add': (2.0, 3.5),
                'taxa_mult': (20, 40),
                'humidity_add': (20, 35),
                'pressure_sub': (15, 35)
            },
            'extrema': {
                'precip_mult': (15, 30),
                'nivel_add': (3.5, 6.0),
                'taxa_mult': (40, 80),
                'humidity_add': (25, 40),
                'pressure_sub': (25, 50)
            }
        }
        
        config = configs.get(intensidade, configs['moderada'])
        
        # Aplicar modificações
        dados['precipitation_mm'] *= np.random.uniform(*config['precip_mult'])
        dados['nivel_metros'] += np.random.uniform(*config['nivel_add'])
        dados['taxa_subida_m_h'] *= np.random.uniform(*config['taxa_mult'])
        dados['humidity_pct'] += np.random.uniform(*config['humidity_add'])
        dados['pressure_hpa'] -= np.random.uniform(*config['pressure_sub'])
        
        # Efeitos secundários
        dados['wind_speed_kmh'] *= np.random.uniform(1.5, 3.0)  # Vento aumenta
        dados['soil_moisture_pct'] += np.random.uniform(20, 40)  # Solo satura
        
        # Garantir limites físicos
        dados['humidity_pct'] = np.clip(dados['humidity_pct'], 0, 100)
        dados['soil_moisture_pct'] = np.clip(dados['soil_moisture_pct'], 0, 100)
        dados['nivel_metros'] = np.clip(dados['nivel_metros'], 0.1, 15.0)
        dados['taxa_subida_m_h'] = np.clip(dados['taxa_subida_m_h'], 0, 2.0)
        
        return dados
    
    def calcular_features_temporais(self, dados_base: Dict, timestamp: datetime) -> Dict:
        """
        Calcula features temporais e derived features.
        
        Args:
            dados_base: Dados básicos dos sensores
            timestamp: Timestamp atual
            
        Returns:
            Dicionário com todas as features necessárias
        """
        features = dados_base.copy()
        
        # Features temporais
        features['hora'] = timestamp.hour
        features['dia_semana'] = timestamp.weekday()
        features['mes'] = timestamp.month
        
        # Features trigonométricas
        features['hora_sin'] = np.sin(2 * np.pi * features['hora'] / 24)
        features['hora_cos'] = np.cos(2 * np.pi * features['hora'] / 24)
        features['mes_sin'] = np.sin(2 * np.pi * features['mes'] / 12)
        features['mes_cos'] = np.cos(2 * np.pi * features['mes'] / 12)
        
        # Simular médias móveis (usando valores anteriores ou estimativas)
        features['precipitation_mm_ma_12h'] = features['precipitation_mm'] * 0.8
        features['nivel_metros_ma_12h'] = features['nivel_metros'] * 0.95
        features['precipitation_mm_ma_24h'] = features['precipitation_mm'] * 0.6
        features['nivel_metros_ma_24h'] = features['nivel_metros'] * 0.9
        
        # Simular desvios padrão
        features['precipitation_mm_std_12h'] = features['precipitation_mm'] * 0.3
        features['nivel_metros_std_12h'] = features['nivel_metros'] * 0.1
        
        # Simular diferenças
        features['precipitation_mm_diff'] = features['precipitation_mm'] * 0.2
        features['nivel_metros_diff'] = features['nivel_metros'] * 0.05
        
        # Features derivadas
        features['aceleracao_nivel'] = features['taxa_subida_m_h'] * 0.1
        features['pressao_normalizada'] = (features['pressure_hpa'] - 1013.25) / 50.0
        
        return features
    
    def fazer_predicao(self, features: Dict, threshold_mode: str = 'urbano') -> Dict:
        """
        Faz predição usando o modelo treinado.
        
        Args:
            features: Dicionário com features
            threshold_mode: Modo de threshold ('emergencia', 'urbano', 'conservador')
            
        Returns:
            Dicionário com resultado da predição
        """
        try:
            # Organizar features na ordem correta
            feature_vector = []
            for feature_name in self.feature_names:
                if feature_name in features:
                    feature_vector.append(features[feature_name])
                else:
                    feature_vector.append(0.0)  # Valor padrão
            
            # Converter para array e normalizar
            X = np.array(feature_vector).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Fazer predição
            probability = self.modelo.predict_proba(X_scaled)[0, 1]
            threshold = self.thresholds[threshold_mode]
            prediction = int(probability >= threshold)
            
            # Calcular nível de alerta
            if probability >= threshold * 1.5:
                alert_level = 3  # VERMELHO
            elif probability >= threshold * 1.2:
                alert_level = 2  # LARANJA
            elif probability >= threshold:
                alert_level = 1  # AMARELO
            else:
                alert_level = 0  # VERDE
            
            return {
                'probability': float(probability),
                'prediction': prediction,
                'threshold': threshold,
                'threshold_mode': threshold_mode,
                'alert_level': alert_level,
                'confidence': float(abs(probability - 0.5) * 2)  # Confiança da predição
            }
            
        except Exception as e:
            print(f"❌ Erro na predição: {e}")
            return {
                'probability': 0.0,
                'prediction': 0,
                'threshold': 0.5,
                'threshold_mode': threshold_mode,
                'alert_level': 0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def simular_sequencia_temporal(self, cenario: str, duracao_horas: int = 24, 
                                  intervalo_minutos: int = 30) -> pd.DataFrame:
        """
        Simula uma sequência temporal de dados.
        
        Args:
            cenario: Tipo de cenário ('normal', 'enchente_leve', 'enchente_moderada', etc.)
            duracao_horas: Duração da simulação em horas
            intervalo_minutos: Intervalo entre medições
            
        Returns:
            DataFrame com histórico da simulação
        """
        print(f"🌊 Simulando cenário '{cenario}' por {duracao_horas}h...")
        
        # Calcular número de pontos
        total_points = int((duracao_horas * 60) / intervalo_minutos)
        start_time = datetime.now()
        
        resultados = []
        
        for i in range(total_points):
            # Calcular timestamp atual
            current_time = start_time + timedelta(minutes=i * intervalo_minutos)
            
            # Gerar dados baseado no cenário
            if cenario == 'normal':
                dados_sensores = self.gerar_dados_baseline()
            elif cenario.startswith('enchente_'):
                intensidade = cenario.split('_')[1]  # leve, moderada, severa, extrema
                dados_sensores = self.simular_cenario_enchente(intensidade)
            elif cenario == 'progressiva':
                # Simular enchente que piora progressivamente
                intensidade_map = {0: 'leve', 25: 'leve', 50: 'moderada', 75: 'severa', 90: 'extrema'}
                progress = int((i / total_points) * 100)
                intensidade = 'leve'
                for threshold, intens in intensidade_map.items():
                    if progress >= threshold:
                        intensidade = intens
                dados_sensores = self.simular_cenario_enchente(intensidade)
            else:
                dados_sensores = self.gerar_dados_baseline()
            
            # Calcular features completas
            features = self.calcular_features_temporais(dados_sensores, current_time)
            
            # Fazer predições com diferentes thresholds
            pred_emerg = self.fazer_predicao(features, 'emergencia')
            pred_urbano = self.fazer_predicao(features, 'urbano')
            pred_conserv = self.fazer_predicao(features, 'conservador')
            
            # Compilar resultado
            resultado = {
                'timestamp': current_time,
                'cenario': cenario,
                'hora_simulacao': i * intervalo_minutos / 60,
                
                # Dados dos sensores
                **{f'sensor_{k}': v for k, v in dados_sensores.items()},
                
                # Predições
                'prob_enchente': pred_urbano['probability'],
                'pred_emergencia': pred_emerg['prediction'],
                'pred_urbano': pred_urbano['prediction'],
                'pred_conservador': pred_conserv['prediction'],
                'alert_level_emerg': pred_emerg['alert_level'],
                'alert_level_urbano': pred_urbano['alert_level'],
                'alert_level_conserv': pred_conserv['alert_level'],
                
                # Metadados
                'confidence': pred_urbano['confidence']
            }
            
            resultados.append(resultado)
            
            # Progress indicator
            if i % (total_points // 10) == 0:
                progress = int((i / total_points) * 100)
                print(f"   {progress}% concluído...")
        
        df_resultado = pd.DataFrame(resultados)
        print(f"✅ Simulação concluída: {len(df_resultado)} pontos gerados")
        
        return df_resultado
    
    def avaliar_performance_simulacao(self, df_simulacao: pd.DataFrame, 
                                     verdade_esperada: str) -> Dict:
        """
        Avalia a performance do modelo na simulação.
        
        Args:
            df_simulacao: DataFrame com resultados da simulação
            verdade_esperada: 'normal' ou 'enchente'
            
        Returns:
            Dicionário com métricas de avaliação
        """
        # Definir ground truth
        y_true = 1 if verdade_esperada == 'enchente' else 0
        ground_truth = [y_true] * len(df_simulacao)
        
        # Calcular métricas para cada threshold
        resultados = {}
        
        for mode in ['emergencia', 'urbano', 'conservador']:
            predictions = df_simulacao[f'pred_{mode}'].values
            
            # Métricas básicas
            accuracy = np.mean(predictions == ground_truth)
            
            if y_true == 1:  # Cenário de enchente
                recall = np.mean(predictions == 1)  # Quantas foram detectadas
                false_negatives = np.sum(predictions == 0)
                detection_rate = recall
            else:  # Cenário normal
                specificity = np.mean(predictions == 0)  # Quantas foram corretamente rejeitadas
                false_positives = np.sum(predictions == 1)
                detection_rate = specificity
            
            resultados[mode] = {
                'accuracy': accuracy,
                'detection_rate': detection_rate,
                'false_positives': np.sum((predictions == 1) & (np.array(ground_truth) == 0)),
                'false_negatives': np.sum((predictions == 0) & (np.array(ground_truth) == 1)),
                'total_alerts': np.sum(predictions == 1),
                'avg_probability': df_simulacao['prob_enchente'].mean(),
                'max_probability': df_simulacao['prob_enchente'].max(),
                'min_probability': df_simulacao['prob_enchente'].min()
            }
        
        return resultados
    
    def gerar_relatorio_simulacao(self, df_simulacao: pd.DataFrame, 
                                 performance: Dict, cenario: str):
        """
        Gera relatório detalhado da simulação.
        
        Args:
            df_simulacao: DataFrame com resultados
            performance: Métricas de performance
            cenario: Nome do cenário simulado
        """
        print(f"\n📊 RELATÓRIO DE SIMULAÇÃO - {cenario.upper()}")
        print("="*60)
        
        # Estatísticas gerais
        print(f"🕐 Duração: {df_simulacao['hora_simulacao'].max():.1f} horas")
        print(f"📈 Pontos simulados: {len(df_simulacao)}")
        print(f"🌡️ Temperatura média: {df_simulacao['sensor_temperature_c'].mean():.1f}°C")
        print(f"🌧️ Precipitação média: {df_simulacao['sensor_precipitation_mm'].mean():.1f}mm")
        print(f"📏 Nível médio: {df_simulacao['sensor_nivel_metros'].mean():.2f}m")
        
        # Performance por threshold
        print(f"\n🎯 PERFORMANCE POR THRESHOLD:")
        print("-"*50)
        
        for mode, metrics in performance.items():
            print(f"\n{mode.upper()}:")
            print(f"  Acurácia: {metrics['accuracy']:.1%}")
            print(f"  Taxa de detecção: {metrics['detection_rate']:.1%}")
            print(f"  Total de alertas: {metrics['total_alerts']}")
            print(f"  Falsos positivos: {metrics['false_positives']}")
            print(f"  Falsos negativos: {metrics['false_negatives']}")
            print(f"  Prob. média: {metrics['avg_probability']:.3f}")
            print(f"  Prob. máxima: {metrics['max_probability']:.3f}")
        
        # Momentos críticos
        prob_high = df_simulacao[df_simulacao['prob_enchente'] > 0.8]
        if len(prob_high) > 0:
            print(f"\n⚠️ MOMENTOS DE ALTA PROBABILIDADE (>80%):")
            print(f"  Ocorrências: {len(prob_high)}")
            print(f"  Duração: {len(prob_high) * 0.5:.1f} horas")
            print(f"  Probabilidade máxima: {prob_high['prob_enchente'].max():.3f}")
        
        # Alertas por nível
        print(f"\n🚨 DISTRIBUIÇÃO DE ALERTAS:")
        for level in [0, 1, 2, 3]:
            count = np.sum(df_simulacao['alert_level_urbano'] == level)
            percentage = count / len(df_simulacao) * 100
            level_names = ['VERDE', 'AMARELO', 'LARANJA', 'VERMELHO']
            print(f"  {level_names[level]}: {count} ({percentage:.1f}%)")
    
    def visualizar_simulacao(self, df_simulacao: pd.DataFrame, cenario: str):
        """
        Cria visualizações da simulação.
        
        Args:
            df_simulacao: DataFrame com resultados
            cenario: Nome do cenário
        """
        try:
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            fig.suptitle(f'Simulação SAEA - Cenário: {cenario}', fontsize=16)
            
            # 1. Probabilidade ao longo do tempo
            axes[0, 0].plot(df_simulacao['hora_simulacao'], df_simulacao['prob_enchente'], 
                           'b-', linewidth=2, label='Probabilidade')
            axes[0, 0].axhline(y=0.05, color='g', linestyle='--', label='Threshold Emergência')
            axes[0, 0].axhline(y=0.50, color='orange', linestyle='--', label='Threshold Urbano')
            axes[0, 0].axhline(y=0.80, color='r', linestyle='--', label='Threshold Conservador')
            axes[0, 0].set_title('Probabilidade de Enchente')
            axes[0, 0].set_xlabel('Horas')
            axes[0, 0].set_ylabel('Probabilidade')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Sensores principais
            axes[0, 1].plot(df_simulacao['hora_simulacao'], df_simulacao['sensor_precipitation_mm'], 
                           'b-', label='Precipitação (mm)')
            axes2 = axes[0, 1].twinx()
            axes2.plot(df_simulacao['hora_simulacao'], df_simulacao['sensor_nivel_metros'], 
                      'r-', label='Nível (m)')
            axes[0, 1].set_title('Sensores Principais')
            axes[0, 1].set_xlabel('Horas')
            axes[0, 1].set_ylabel('Precipitação (mm)', color='b')
            axes2.set_ylabel('Nível (m)', color='r')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Níveis de alerta
            alert_colors = ['green', 'yellow', 'orange', 'red']
            for i, alert_level in enumerate([0, 1, 2, 3]):
                mask = df_simulacao['alert_level_urbano'] == alert_level
                if mask.any():
                    axes[1, 0].scatter(df_simulacao.loc[mask, 'hora_simulacao'], 
                                     [alert_level] * mask.sum(),
                                     c=alert_colors[i], s=20, alpha=0.7,
                                     label=f'Nível {alert_level}')
            axes[1, 0].set_title('Níveis de Alerta ao Longo do Tempo')
            axes[1, 0].set_xlabel('Horas')
            axes[1, 0].set_ylabel('Nível de Alerta')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Comparação de thresholds
            thresholds = ['emergencia', 'urbano', 'conservador']
            colors = ['green', 'orange', 'red']
            for i, threshold in enumerate(thresholds):
                alerts = df_simulacao[f'pred_{threshold}']
                alert_times = df_simulacao.loc[alerts == 1, 'hora_simulacao']
                if len(alert_times) > 0:
                    axes[1, 1].scatter(alert_times, [i] * len(alert_times), 
                                     c=colors[i], s=30, alpha=0.8, label=threshold)
            axes[1, 1].set_title('Comparação de Alertas por Threshold')
            axes[1, 1].set_xlabel('Horas')
            axes[1, 1].set_ylabel('Modo de Threshold')
            axes[1, 1].set_yticks([0, 1, 2])
            axes[1, 1].set_yticklabels(['Emergência', 'Urbano', 'Conservador'])
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # 5. Distribuição de probabilidades
            axes[2, 0].hist(df_simulacao['prob_enchente'], bins=30, alpha=0.7, 
                           color='skyblue', edgecolor='black')
            axes[2, 0].axvline(x=0.05, color='g', linestyle='--', label='Emergência')
            axes[2, 0].axvline(x=0.50, color='orange', linestyle='--', label='Urbano')
            axes[2, 0].axvline(x=0.80, color='r', linestyle='--', label='Conservador')
            axes[2, 0].set_title('Distribuição de Probabilidades')
            axes[2, 0].set_xlabel('Probabilidade')
            axes[2, 0].set_ylabel('Frequência')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
            
            # 6. Condições atmosféricas
            axes[2, 1].plot(df_simulacao['hora_simulacao'], df_simulacao['sensor_humidity_pct'], 
                           'b-', label='Umidade (%)')
            axes[2, 1].plot(df_simulacao['hora_simulacao'], df_simulacao['sensor_pressure_hpa'] - 1000, 
                           'r-', label='Pressão (hPa-1000)')
            axes[2, 1].set_title('Condições Atmosféricas')
            axes[2, 1].set_xlabel('Horas')
            axes[2, 1].set_ylabel('Valor')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Salvar gráfico
            filename = f'simulacao_{cenario}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📊 Gráficos salvos em: {filename}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Erro ao criar visualizações: {e}")
    
    def executar_suite_testes(self):
        """
        Executa uma suíte completa de testes.
        """
        print("🧪 EXECUTANDO SUÍTE DE TESTES SAEA")
        print("="*50)
        
        cenarios = [
            ('normal', 'normal'),
            ('enchente_leve', 'enchente'),
            ('enchente_moderada', 'enchente'),
            ('enchente_severa', 'enchente'),
            ('progressiva', 'enchente')
        ]
        
        resultados_suite = {}
        
        for cenario, verdade in cenarios:
            print(f"\n🔄 Testando cenário: {cenario}")
            
            # Simular
            df_sim = self.simular_sequencia_temporal(cenario, duracao_horas=12, 
                                                    intervalo_minutos=15)
            
            # Avaliar
            performance = self.avaliar_performance_simulacao(df_sim, verdade)
            
            # Gerar relatório
            self.gerar_relatorio_simulacao(df_sim, performance, cenario)
            
            # Salvar resultados
            resultados_suite[cenario] = {
                'simulacao': df_sim,
                'performance': performance,
                'verdade_esperada': verdade
            }
        
        print(f"\n✅ SUÍTE DE TESTES CONCLUÍDA!")
        print(f"📁 {len(cenarios)} cenários testados")
        
        return resultados_suite


def main():
    """Função principal para executar simulações."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simulador SAEA')
    parser.add_argument('--models-dir', default='models/', 
                       help='Diretório dos modelos')
    parser.add_argument('--cenario', default='enchente_moderada',
                       choices=['normal', 'enchente_leve', 'enchente_moderada', 
                               'enchente_severa', 'enchente_extrema', 'progressiva'],
                       help='Cenário a simular')
    parser.add_argument('--duracao', type=int, default=24,
                       help='Duração em horas')
    parser.add_argument('--intervalo', type=int, default=30,
                       help='Intervalo em minutos')
    parser.add_argument('--suite', action='store_true',
                       help='Executar suíte completa de testes')
    parser.add_argument('--visualizar', action='store_true',
                       help='Gerar visualizações')
    
    args = parser.parse_args()
    
    try:
        # Criar simulador
        simulador = SAEASimulator(args.models_dir)
        
        if args.suite:
            # Executar suíte completa
            resultados = simulador.executar_suite_testes()
        else:
            # Simular cenário específico
            verdade = 'enchente' if 'enchente' in args.cenario else 'normal'
            
            df_simulacao = simulador.simular_sequencia_temporal(
                args.cenario, args.duracao, args.intervalo
            )
            
            performance = simulador.avaliar_performance_simulacao(df_simulacao, verdade)
            simulador.gerar_relatorio_simulacao(df_simulacao, performance, args.cenario)
            
            if args.visualizar:
                simulador.visualizar_simulacao(df_simulacao, args.cenario)
            
            # Salvar resultados
            filename = f'simulacao_{args.cenario}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            df_simulacao.to_csv(filename, index=False)
            print(f"💾 Dados salvos em: {filename}")
        
        print(f"\n🎉 Simulação concluída com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro na simulação: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()