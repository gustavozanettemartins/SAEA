#!/usr/bin/env python3
"""
Script de correção para configurar a estrutura do projeto SAEA
"""

import os
from pathlib import Path

def criar_estrutura_pastas():
    """Cria toda a estrutura de pastas necessária"""
    print("🔧 Criando estrutura de pastas...")
    
    pastas = [
        'data/raw',
        'data/processed', 
        'data/disaster_charter',
        'data/raw/inmet',
        'models',
        'config',
        'hardware/firmware_esp32',
        'logs',
        'results'
    ]
    
    for pasta in pastas:
        Path(pasta).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {pasta}")
    
    print("📁 Estrutura de pastas criada!")

def criar_config_basico():
    """Cria arquivos de configuração básicos"""
    print("\n⚙️ Criando configurações básicas...")
    
    # Config de treinamento
    config_training = {
        "random_state": 42,
        "test_size": 0.2,
        "cv_folds": 5,
        "scoring_metric": "f1",
        "optimization_metric": "recall",
        "balancing_strategy": "smote_tomek",
        "smote_k_neighbors": 3,
        "min_samples_per_class": 10,
        "threshold_tuning": True,
        "use_ensemble": True,
        "verbose": True
    }
    
    # Config de preprocessamento
    config_preprocessing = {
        "features_numericas": [
            "precipitation_mm", "temperature_c", "humidity_pct",
            "pressure_hpa", "wind_speed_kmh", "soil_moisture_pct", 
            "nivel_metros", "taxa_subida_m_h"
        ],
        "janelas_temporais": [24, 48, 72],
        "limites_outliers": {
            "precipitation_mm": [0, 500],
            "temperature_c": [-10, 50], 
            "humidity_pct": [0, 100],
            "nivel_metros": [0, 20]
        }
    }
    
    import json
    
    with open('config/training.json', 'w') as f:
        json.dump(config_training, f, indent=2)
    
    with open('config/preprocessing.json', 'w') as f:
        json.dump(config_preprocessing, f, indent=2)
    
    print("   ✅ config/training.json")
    print("   ✅ config/preprocessing.json")

def verificar_dados():
    """Verifica se os dados necessários existem"""
    print("\n📊 Verificando dados...")
    
    arquivos_necessarios = [
        'data/raw/weather_data.csv',
        'data/raw/river_levels.csv',
        'data/disaster_charter/flood_events.csv'
    ]
    
    dados_ok = True
    for arquivo in arquivos_necessarios:
        if os.path.exists(arquivo):
            print(f"   ✅ {arquivo}")
        else:
            print(f"   ❌ {arquivo} - NÃO ENCONTRADO")
            dados_ok = False
    
    if not dados_ok:
        print("\n💡 Para criar os dados, execute:")
        print("   python script_processar_csv_inmet.py")

def main():
    """Função principal"""
    print("🚀 CONFIGURAÇÃO AUTOMÁTICA DO PROJETO SAEA")
    print("=" * 50)
    
    # 1. Criar pastas
    criar_estrutura_pastas()
    
    # 2. Criar configs
    criar_config_basico()
    
    # 3. Verificar dados
    verificar_dados()
    
    print("\n" + "=" * 50)
    print("✅ CONFIGURAÇÃO CONCLUÍDA!")
    print("\n🚀 Próximos passos:")
    print("1. Se ainda não fez, execute: python script_processar_csv_inmet.py")
    print("2. Execute: python src/preprocessamento.py")
    print("3. Execute: python src/treinamento.py --data data/processed/dataset_ml_final.csv")

if __name__ == "__main__":
    main()