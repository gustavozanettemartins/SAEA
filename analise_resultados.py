#!/usr/bin/env python3
"""
Análise corrigida de threshold para o modelo SAEA
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def analisar_threshold_detalhado(models_dir='models/', data_path='data/processed/dataset_ml_final.csv'):
    """Análise detalhada e corrigida de threshold."""
    
    print("🔍 ANÁLISE DETALHADA DE THRESHOLD - SAEA")
    print("="*60)
    
    # Carregar modelo e scaler
    modelo = joblib.load(f'{models_dir}/modelo_completo.pkl')
    scaler = joblib.load(f'{models_dir}/scaler.pkl')
    
    # Carregar dados
    df = pd.read_csv(data_path)
    
    # Preparar dados de teste (mesma divisão do treinamento)
    feature_columns = [col for col in df.columns 
                     if col not in ['datetime', 'location', 'teve_enchente', 'nivel_alerta']]
    
    # Usar divisão estratificada idêntica ao treinamento
    from sklearn.model_selection import train_test_split
    
    X = df[feature_columns]
    y = df['teve_enchente']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Normalizar dados de teste
    X_test_scaled = scaler.transform(X_test)
    
    print(f"📊 Dados de teste: {len(X_test)} amostras")
    print(f"   Classe 0 (Normal): {sum(y_test == 0)}")
    print(f"   Classe 1 (Enchente): {sum(y_test == 1)}")
    
    # Obter probabilidades
    y_proba = modelo.predict_proba(X_test_scaled)
    
    print(f"\n🔢 Probabilidades shape: {y_proba.shape}")
    print(f"   Probabilidades classe 1 - Min: {y_proba[:, 1].min():.3f}, Max: {y_proba[:, 1].max():.3f}")
    
    # Análise de threshold com range mais amplo
    print(f"\n🎯 ANÁLISE DE THRESHOLD DETALHADA:")
    print("="*70)
    print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'TP':<5} {'FP':<5} {'FN':<5} {'TN':<5}")
    print("-" * 70)
    
    thresholds = np.arange(0.05, 0.95, 0.05)
    results = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_proba[:, 1] >= threshold).astype(int)
        
        # Calcular métricas
        precision = precision_score(y_test, y_pred_thresh, zero_division=0)
        recall = recall_score(y_test, y_pred_thresh, zero_division=0)
        f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
        
        # Matriz de confusão
        cm = confusion_matrix(y_test, y_pred_thresh)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Caso especial: apenas uma classe predita
            if len(np.unique(y_pred_thresh)) == 1:
                if y_pred_thresh[0] == 0:  # Tudo predito como 0
                    tn = sum(y_test == 0)
                    fp = 0
                    fn = sum(y_test == 1)
                    tp = 0
                else:  # Tudo predito como 1
                    tn = 0
                    fp = sum(y_test == 0)
                    fn = 0
                    tp = sum(y_test == 1)
            else:
                tn = fp = fn = tp = 0
        
        print(f"{threshold:<10.2f} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {tp:<5} {fp:<5} {fn:<5} {tn:<5}")
        
        # Salvar resultados
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,  # Taxa de falsos negativos
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0   # Taxa de falsos positivos
        })
    
    # Converter para DataFrame
    results_df = pd.DataFrame(results)
    
    # Encontrar melhor threshold baseado em diferentes critérios
    print(f"\n🏆 MELHORES THRESHOLDS POR CRITÉRIO:")
    print("="*50)
    
    # Melhor F1-Score
    best_f1_idx = results_df['f1'].idxmax()
    best_f1_threshold = results_df.loc[best_f1_idx, 'threshold']
    best_f1_score = results_df.loc[best_f1_idx, 'f1']
    print(f"Melhor F1-Score: {best_f1_threshold:.2f} (F1: {best_f1_score:.3f})")
    
    # Melhor Recall (minimizar falsos negativos)
    best_recall_idx = results_df['recall'].idxmax()
    best_recall_threshold = results_df.loc[best_recall_idx, 'threshold']
    best_recall_score = results_df.loc[best_recall_idx, 'recall']
    print(f"Melhor Recall: {best_recall_threshold:.2f} (Recall: {best_recall_score:.3f})")
    
    # Melhor balanceamento (Score customizado)
    results_df['custom_score'] = 0.3 * results_df['precision'] + 0.5 * results_df['recall'] + 0.2 * results_df['f1']
    best_custom_idx = results_df['custom_score'].idxmax()
    best_custom_threshold = results_df.loc[best_custom_idx, 'threshold']
    best_custom_score = results_df.loc[best_custom_idx, 'custom_score']
    print(f"Melhor Score Customizado: {best_custom_threshold:.2f} (Score: {best_custom_score:.3f})")
    
    # Threshold que minimiza falsos negativos (crítico para enchentes)
    min_fn_mask = results_df['fn'] == results_df['fn'].min()
    min_fn_candidates = results_df[min_fn_mask]
    # Entre os que minimizam FN, escolher o com melhor precisão
    best_safety_idx = min_fn_candidates['precision'].idxmax()
    best_safety_threshold = min_fn_candidates.loc[best_safety_idx, 'threshold']
    best_safety_fn = min_fn_candidates.loc[best_safety_idx, 'fn']
    print(f"Mínimos Falsos Negativos: {best_safety_threshold:.2f} (FN: {best_safety_fn})")
    
    # Análise específica para o threshold atual (0.5)
    current_result = results_df[results_df['threshold'] == 0.5]
    if not current_result.empty:
        print(f"\n📊 ANÁLISE DO THRESHOLD ATUAL (0.5):")
        current = current_result.iloc[0]
        print(f"   Precision: {current['precision']:.3f}")
        print(f"   Recall: {current['recall']:.3f}")
        print(f"   F1-Score: {current['f1']:.3f}")
        print(f"   Falsos Negativos: {current['fn']} (Taxa: {current['fnr']:.1%})")
        print(f"   Falsos Positivos: {current['fp']} (Taxa: {current['fpr']:.1%})")
    
    # Recomendações
    print(f"\n💡 RECOMENDAÇÕES:")
    print("="*40)
    
    if best_safety_fn == 0:
        print(f"✅ EXCELENTE: Existem thresholds que eliminam falsos negativos!")
        print(f"   Recomendado: {best_safety_threshold:.2f} (zero enchentes perdidas)")
    else:
        print(f"⚠️ Mínimo de falsos negativos: {best_safety_fn}")
        print(f"   Threshold recomendado: {best_safety_threshold:.2f}")
    
    print(f"\n🎯 Para uso em produção:")
    print(f"   • Segurança máxima: {best_safety_threshold:.2f} (mínimo FN)")
    print(f"   • Balanceado: {best_f1_threshold:.2f} (melhor F1)")
    print(f"   • Personalizado: {best_custom_threshold:.2f} (prioriza recall)")
    
    return results_df

def visualizar_threshold_analysis(results_df):
    """Cria visualização da análise de threshold."""
    try:
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Métricas principais
        plt.subplot(2, 3, 1)
        plt.plot(results_df['threshold'], results_df['precision'], 'b-', label='Precision', linewidth=2)
        plt.plot(results_df['threshold'], results_df['recall'], 'r-', label='Recall', linewidth=2)
        plt.plot(results_df['threshold'], results_df['f1'], 'g-', label='F1-Score', linewidth=2)
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Métricas vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Falsos Negativos e Positivos
        plt.subplot(2, 3, 2)
        plt.plot(results_df['threshold'], results_df['fn'], 'r-', label='Falsos Negativos', linewidth=2)
        plt.plot(results_df['threshold'], results_df['fp'], 'orange', label='Falsos Positivos', linewidth=2)
        plt.xlabel('Threshold')
        plt.ylabel('Contagem')
        plt.title('Erros vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 3: Taxa de erro
        plt.subplot(2, 3, 3)
        plt.plot(results_df['threshold'], results_df['fnr'], 'r-', label='Taxa FN', linewidth=2)
        plt.plot(results_df['threshold'], results_df['fpr'], 'orange', label='Taxa FP', linewidth=2)
        plt.xlabel('Threshold')
        plt.ylabel('Taxa de Erro')
        plt.title('Taxa de Erro vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 4: Score customizado
        plt.subplot(2, 3, 4)
        plt.plot(results_df['threshold'], results_df['custom_score'], 'purple', linewidth=2)
        plt.xlabel('Threshold')
        plt.ylabel('Score Customizado')
        plt.title('Score Customizado vs Threshold')
        plt.grid(True, alpha=0.3)
        
        # Subplot 5: Confusion Matrix components
        plt.subplot(2, 3, 5)
        plt.plot(results_df['threshold'], results_df['tp'], 'g-', label='True Positives', linewidth=2)
        plt.plot(results_df['threshold'], results_df['tn'], 'b-', label='True Negatives', linewidth=2)
        plt.xlabel('Threshold')
        plt.ylabel('Contagem')
        plt.title('Predições Corretas vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 6: Resumo
        plt.subplot(2, 3, 6)
        best_f1_idx = results_df['f1'].idxmax()
        best_custom_idx = results_df['custom_score'].idxmax()
        
        plt.axvline(x=results_df.loc[best_f1_idx, 'threshold'], color='green', 
                   linestyle='--', label=f'Melhor F1 ({results_df.loc[best_f1_idx, "threshold"]:.2f})')
        plt.axvline(x=results_df.loc[best_custom_idx, 'threshold'], color='purple', 
                   linestyle='--', label=f'Melhor Custom ({results_df.loc[best_custom_idx, "threshold"]:.2f})')
        plt.axvline(x=0.5, color='red', linestyle='-', label='Atual (0.5)')
        
        plt.plot(results_df['threshold'], results_df['f1'], 'g-', alpha=0.7)
        plt.plot(results_df['threshold'], results_df['custom_score'], 'purple', alpha=0.7)
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Comparação de Thresholds')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Gráfico salvo como 'threshold_analysis.png'")
        
    except Exception as e:
        print(f"Erro ao criar visualização: {e}")

if __name__ == "__main__":
    # Executar análise
    results_df = analisar_threshold_detalhado()
    
    # Criar visualização
    print(f"\n📊 Criando visualizações...")
    try:
        visualizar_threshold_analysis(results_df)
    except ImportError:
        print("Matplotlib não disponível para visualizações")
    
    print(f"\n🎉 Análise de threshold concluída!")