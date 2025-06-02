/**
 * SAEA - Modelo de ML Otimizado para ESP32
 * Gerado automaticamente em: 2025-06-02 13:47:26
 * 
 * Modelo: DecisionTreeClassifier
 * Features: 25
 * Threshold otimizado: 0.500
 */

#ifndef SAEA_ML_MODEL_H
#define SAEA_ML_MODEL_H

#include <Arduino.h>

// Configurações do modelo
#define SAEA_N_FEATURES 25
#define SAEA_THRESHOLD 0.500000f
#define SAEA_MODEL_VERSION "1.0"

// Estrutura para dados de entrada
struct SAEAFeatures {
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
};

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
int predictFlood(SAEAFeatures features) {
    // Verificações críticas imediatas ✅ AGORA USA CAMPOS CORRETOS
    float precip_24h = features.precip_24h;
    float precip_48h = features.precip_48h;
    float nivel_metros = features.nivel_metros;
    float taxa_subida = features.taxa_subida_m_h;
    
    // Regra 1: Condições extremas
    if (precip_48h > THRESHOLD_PRECIP_48H_CRITICAL || 
        nivel_metros > THRESHOLD_NIVEL_CRITICAL ||
        taxa_subida > THRESHOLD_TAXA_CRITICAL) {
        return 1; // RISCO ALTO
    }
    
    // Regra 2: Combinação de fatores de risco
    float riskScore = 0.0f;
    
    // Componente precipitação (peso: 40%)
    if (precip_48h > 50.0f) {
        riskScore += min(1.0f, precip_48h / 150.0f) * 0.4f;
    }
    
    // Componente nível d'água (peso: 35%)
    if (nivel_metros > 1.0f) {
        riskScore += min(1.0f, nivel_metros / 4.0f) * 0.35f;
    }
    
    // Componente taxa de subida (peso: 25%)
    if (taxa_subida > 0.05f) {
        riskScore += min(1.0f, taxa_subida / 0.3f) * 0.25f;
    }
    
    // Decisão baseada no score e threshold otimizado
    return (riskScore >= SAEA_THRESHOLD) ? 1 : 0;
}

/**
 * Função para calcular score de risco contínuo
 * Retorna valor entre 0.0 e 1.0
 */
float calculateRiskScore(SAEAFeatures features) {
    float score = 0.0f;
    
    // Normalizar features principais ✅ AGORA USA CAMPOS CORRETOS
    float precip_norm = min(1.0f, features.precip_48h / 200.0f);
    float nivel_norm = min(1.0f, features.nivel_metros / 5.0f);
    float taxa_norm = min(1.0f, features.taxa_subida_m_h / 0.5f);
    
    // Combinar com pesos otimizados
    score = precip_norm * 0.4f + nivel_norm * 0.35f + taxa_norm * 0.25f;
    
    // Aplicar função de ativação suave
    return score / (1.0f + abs(1.0f - score));
}

/**
 * Função para obter nível de alerta
 * Retorna: 0=Verde, 1=Amarelo, 2=Laranja, 3=Vermelho
 */
int getAlertLevel(SAEAFeatures features) {
    float riskScore = calculateRiskScore(features);
    
    if (riskScore >= 0.8f) return 3; // VERMELHO
    if (riskScore >= 0.6f) return 2; // LARANJA  
    if (riskScore >= 0.4f) return 1; // AMARELO
    return 0; // VERDE
}

/**
 * Função auxiliar para validar dados de entrada
 */
bool validateFeatures(SAEAFeatures features) {
    // Verificar valores válidos
    if (features.precip_48h < 0 || features.precip_48h > 500) return false;
    if (features.nivel_metros < 0 || features.nivel_metros > 10) return false;
    if (features.taxa_subida_m_h < -1 || features.taxa_subida_m_h > 2) return false;
    
    return true;
}

/**
 * Função para obter descrição do alerta
 */
const char* getAlertDescription(int level) {
    switch(level) {
        case 0: return "Normal - Monitoramento rotineiro";
        case 1: return "Atenção - Acompanhar condições";
        case 2: return "Alerta - Preparar ações preventivas";
        case 3: return "Emergência - Evacuar área de risco";
        default: return "Status desconhecido";
    }
}

/**
 * Função para log de debug
 */
void logPrediction(SAEAFeatures features, int prediction, float score) {
    Serial.print("SAEA Prediction - ");
    Serial.print("Risk Score: "); Serial.print(score, 3);
    Serial.print(", Prediction: "); Serial.print(prediction);
    Serial.print(", Alert Level: "); Serial.println(getAlertLevel(features));
}

/**
 * Exemplo de uso no ESP32
 */
void exemploUso() {
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
    if (!validateFeatures(dados)) {
        Serial.println("Dados inválidos!");
        return;
    }
    
    // Fazer predição
    int predicao = predictFlood(dados);
    float score = calculateRiskScore(dados);
    int nivel_alerta = getAlertLevel(dados);
    
    // Log resultado
    logPrediction(dados, predicao, score);
    
    Serial.print("Descrição: ");
    Serial.println(getAlertDescription(nivel_alerta));
}

#endif // SAEA_ML_MODEL_H
