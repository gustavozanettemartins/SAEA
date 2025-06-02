/*
 * SAEA_ESP32_Solar.ino
 * Sistema Autônomo de Alerta de Enchentes - Com Energia Solar
 * 
 * Autor: Sistema SAEA
 * Data: 2025-06-02
 * Versão: 2.0 - Solar Edition
 * 
 * Hardware necessário:
 * - ESP32 DevKit
 * - Sensor de chuva (analógico)
 * - Sensor DHT22 (temperatura/umidade)
 * - Sensor BMP280 (pressão)
 * - Sensor ultrassônico HC-SR04 (nível de água)
 * - Display OLED 128x64 (opcional)
 * - LED RGB para alertas visuais
 * - Painel Solar 6V 2W + Bateria Li-ion 18650 + TP4056
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <DHT.h>
#include <Wire.h>
#include <Adafruit_BMP280.h>
#include <SSD1306Display.h>
#include "ml_model.h"  // Seu arquivo de ML

// ============================
// CONFIGURAÇÕES DO HARDWARE
// ============================

// Pinos dos sensores
#define DHT_PIN 4
#define DHT_TYPE DHT22
#define RAIN_SENSOR_PIN A0
#define ULTRASONIC_TRIG_PIN 5
#define ULTRASONIC_ECHO_PIN 18
#define SOIL_MOISTURE_PIN A3

// Pinos dos LEDs de alerta
#define LED_VERDE_PIN 25
#define LED_AMARELO_PIN 26  
#define LED_LARANJA_PIN 27
#define LED_VERMELHO_PIN 14

// Pinos do sistema de energia solar
#define BATTERY_VOLTAGE_PIN A1     // Monitoramento da bateria
#define SOLAR_VOLTAGE_PIN A2       // Monitoramento do painel solar
#define POWER_LED_PIN 13           // LED indicador de energia
#define BUZZER_PIN 12              // Buzzer para alertas sonoros

// Display OLED (opcional)
#define OLED_SDA 21
#define OLED_SCL 22

// ============================
// CONFIGURAÇÕES DE ENERGIA SOLAR
// ============================

// Thresholds de bateria (em Volts)
#define BATTERY_FULL 4.2f          // Bateria carregada
#define BATTERY_GOOD 3.8f          // Bateria boa
#define BATTERY_LOW 3.6f           // Bateria baixa
#define BATTERY_CRITICAL 3.3f      // Bateria crítica
#define BATTERY_EMERGENCY 3.0f     // Emergência

// Estados de energia
enum PowerMode {
    POWER_NORMAL,      // Operação normal (30s)
    POWER_ECONOMY,     // Economia (2 min)
    POWER_CRITICAL,    // Crítico (5 min)
    POWER_EMERGENCY    // Emergência (10 min, só alertas)
};

// ============================
// CONFIGURAÇÕES DE REDE
// ============================
const char* ssid = "SEU_WIFI_SSID";
const char* password = "SEU_WIFI_PASSWORD";
const char* serverURL = "http://seu-servidor.com/api/saea";

// ============================
// OBJETOS DOS SENSORES
// ============================
DHT dht(DHT_PIN, DHT_TYPE);
Adafruit_BMP280 bmp;
SSD1306Display display(0x3c, OLED_SDA, OLED_SCL);

// ============================
// VARIÁVEIS GLOBAIS
// ============================
struct SensorData {
    float temperature;
    float humidity;
    float pressure;
    float precipitation;
    float windSpeed;
    float soilMoisture;
    float waterLevel;
    float waterRiseRate;
    unsigned long timestamp;
};

// Histórico para cálculos de médias móveis
const int HISTORY_SIZE = 48;
SensorData sensorHistory[HISTORY_SIZE];
int historyIndex = 0;
int historyCount = 0;

// Variáveis de energia
PowerMode currentPowerMode = POWER_NORMAL;
float batteryVoltage = 0.0f;
float solarVoltage = 0.0f;
bool isCharging = false;
bool displayEnabled = true;

// Configurações de timing
unsigned long lastMeasurement = 0;
unsigned long lastWifiRetry = 0;
unsigned long lastPowerCheck = 0;

// Estado atual
int currentAlertLevel = 0;
float currentRiskScore = 0.0;
bool wifiConnected = false;

// ============================
// SETUP
// ============================
void setup() {
    Serial.begin(115200);
    Serial.println("🌊☀️ SAEA - Sistema Autônomo de Alerta de Enchentes - Solar Edition");
    Serial.println("Inicializando...");
    
    // Configurar pinos
    pinMode(LED_VERDE_PIN, OUTPUT);
    pinMode(LED_AMARELO_PIN, OUTPUT);
    pinMode(LED_LARANJA_PIN, OUTPUT);
    pinMode(LED_VERMELHO_PIN, OUTPUT);
    pinMode(POWER_LED_PIN, OUTPUT);
    pinMode(BUZZER_PIN, OUTPUT);
    pinMode(ULTRASONIC_TRIG_PIN, OUTPUT);
    pinMode(ULTRASONIC_ECHO_PIN, INPUT);
    
    // Teste inicial dos LEDs
    testAllLEDs();
    
    // Inicializar sensores
    dht.begin();
    
    if (!bmp.begin()) {
        Serial.println("❌ Erro: Sensor BMP280 não encontrado!");
    }
    
    // Inicializar display (se disponível)
    if (!display.init()) {
        Serial.println("⚠️ Display OLED não encontrado - continuando sem display");
        displayEnabled = false;
    } else {
        displayEnabled = true;
        updateDisplayWithPower("SAEA Solar", "Inicializando...");
    }
    
    // Verificar status inicial da energia
    readPowerStatus();
    managePowerMode();
    
    // Conectar WiFi (se energia permitir)
    if (currentPowerMode != POWER_EMERGENCY) {
        connectWiFi();
    }
    
    // Teste inicial do modelo ML
    Serial.println("🧠 Testando modelo de ML...");
    testMLModel();
    
    Serial.println("✅ Sistema inicializado com sucesso!");
    if (displayEnabled) {
        updateDisplayWithPower("SAEA Pronto", "Sistema OK");
    }
    
    // Som de inicialização
    playStartupSound();
}

// ============================
// LOOP PRINCIPAL
// ============================
void loop() {
    unsigned long currentTime = millis();
    
    // Gerenciar energia a cada 10 segundos
    if (currentTime - lastPowerCheck >= 10000) {
        managePowerMode();
        lastPowerCheck = currentTime;
    }
    
    // Atualizar LED de energia
    updatePowerLED();
    
    // Intervalo baseado no modo de energia
    unsigned long measurementInterval = getMeasurementInterval();
    
    // Verificar se é hora de fazer uma medição
    if (currentTime - lastMeasurement >= measurementInterval) {
        
        if (currentPowerMode == POWER_EMERGENCY) {
            // Modo emergência: só alertas críticos
            emergencyMode();
        } else {
            // Operação normal (com adaptações baseadas na energia)
            normalOperation();
        }
        
        lastMeasurement = currentTime;
    }
    
    // Tentar reconectar WiFi se necessário (exceto em emergência)
    if (!wifiConnected && currentPowerMode != POWER_EMERGENCY && 
        currentTime - lastWifiRetry >= 60000) {
        connectWiFi();
        lastWifiRetry = currentTime;
    }
    
    // Sleep inteligente baseado no modo de energia
    applyPowerSaving();
}

// ============================
// GERENCIAMENTO DE ENERGIA SOLAR
// ============================
void readPowerStatus() {
    // Ler tensão da bateria (divisor de tensão 2:1)
    int batteryRaw = analogRead(BATTERY_VOLTAGE_PIN);
    batteryVoltage = (batteryRaw * 3.3 / 4095.0) * 2.0;
    
    // Ler tensão do painel solar (divisor de tensão 3:1)
    int solarRaw = analogRead(SOLAR_VOLTAGE_PIN);
    solarVoltage = (solarRaw * 3.3 / 4095.0) * 3.0;
    
    // Verificar se está carregando
    isCharging = (solarVoltage > 4.5f && batteryVoltage < BATTERY_FULL);
    
    // Log de energia (só no modo normal para economizar)
    if (currentPowerMode == POWER_NORMAL && Serial) {
        Serial.printf("🔋 Bat: %.2fV | ☀️ Solar: %.2fV | %s\n", 
                     batteryVoltage, solarVoltage, 
                     isCharging ? "Carregando" : "Uso");
    }
}

void managePowerMode() {
    readPowerStatus();
    
    PowerMode newMode = currentPowerMode;
    
    // Determinar modo baseado na bateria
    if (batteryVoltage >= BATTERY_GOOD) {
        newMode = POWER_NORMAL;
    } else if (batteryVoltage >= BATTERY_LOW) {
        newMode = POWER_ECONOMY;
    } else if (batteryVoltage >= BATTERY_CRITICAL) {
        newMode = POWER_CRITICAL;
    } else {
        newMode = POWER_EMERGENCY;
    }
    
    // Se está carregando e sol forte, forçar modo normal
    if (isCharging && solarVoltage > 5.0f) {
        newMode = POWER_NORMAL;
    }
    
    // Mudança de modo
    if (newMode != currentPowerMode) {
        PowerMode oldMode = currentPowerMode;
        currentPowerMode = newMode;
        
        Serial.printf("⚡ Energia: %s → %s\n", 
                     getPowerModeDescription(oldMode),
                     getPowerModeDescription(currentPowerMode));
        
        // Ajustar display baseado no novo modo
        if (currentPowerMode == POWER_EMERGENCY) {
            if (displayEnabled) display.displayOff();
        } else {
            if (displayEnabled) display.displayOn();
        }
        
        // Som de mudança de modo
        playModeChangeSound(currentPowerMode);
    }
}

const char* getPowerModeDescription(PowerMode mode) {
    switch(mode) {
        case POWER_NORMAL:    return "Normal";
        case POWER_ECONOMY:   return "Economia";
        case POWER_CRITICAL:  return "Crítico";
        case POWER_EMERGENCY: return "Emergência";
        default:              return "Desconhecido";
    }
}

unsigned long getMeasurementInterval() {
    switch(currentPowerMode) {
        case POWER_NORMAL:    return 30000;   // 30 segundos
        case POWER_ECONOMY:   return 120000;  // 2 minutos
        case POWER_CRITICAL:  return 300000;  // 5 minutos
        case POWER_EMERGENCY: return 600000;  // 10 minutos
        default:              return 30000;
    }
}

const char* getBatteryStatus() {
    if (batteryVoltage >= BATTERY_FULL) return "Cheia";
    if (batteryVoltage >= BATTERY_GOOD) return "Boa";
    if (batteryVoltage >= BATTERY_LOW) return "Baixa";
    if (batteryVoltage >= BATTERY_CRITICAL) return "Crítica";
    return "Emergência";
}

// ============================
// MODOS DE OPERAÇÃO
// ============================
void normalOperation() {
    // 1. Ler todos os sensores
    SensorData currentData = readAllSensors();
    
    // 2. Adicionar ao histórico
    addToHistory(currentData);
    
    // 3. Preparar dados para o modelo ML
    SAEAFeatures features = prepareMLFeatures(currentData);
    
    // 4. Fazer predição com modelo ML
    int prediction = predictFlood(features);
    currentRiskScore = calculateRiskScore(features);
    currentAlertLevel = getAlertLevel(features);
    
    // 5. Atualizar alertas visuais
    updateAlerts(currentAlertLevel);
    
    // 6. Som de alerta se necessário
    if (currentAlertLevel >= 2) {
        soundAlert(currentAlertLevel);
    }
    
    // 7. Atualizar display (se habilitado e energia permitir)
    if (displayEnabled && currentPowerMode != POWER_CRITICAL) {
        String statusMsg = "Risco: " + String(currentRiskScore, 2);
        updateDisplayWithPower("SAEA Monitor", statusMsg);
    }
    
    // 8. Enviar dados para servidor (se conectado e energia permitir)
    if (wifiConnected && currentPowerMode == POWER_NORMAL) {
        sendDataToServer(currentData, prediction, currentRiskScore, currentAlertLevel);
    }
    
    // 9. Log no Serial (adaptado ao modo de energia)
    if (currentPowerMode <= POWER_ECONOMY) {
        logCurrentStatus(currentData, prediction, currentRiskScore, currentAlertLevel);
    }
}

void emergencyMode() {
    Serial.println("🚨 MODO EMERGÊNCIA - Bateria crítica!");
    
    // Ler apenas sensores críticos para economia
    float waterLevel = readWaterLevel();
    float precipitation = readRainSensor();
    
    // Atualizar timestamp
    unsigned long currentTime = millis();
    
    // Predição simplificada (sem ML complexo)
    bool criticalFlood = (waterLevel > THRESHOLD_NIVEL_CRITICAL || 
                         precipitation > 50.0f);  // 50mm/h é crítico
    
    // Calcular score simplificado
    float emergencyScore = 0.0f;
    if (waterLevel > THRESHOLD_NIVEL_HIGH) emergencyScore += 0.4f;
    if (precipitation > 30.0f) emergencyScore += 0.4f;
    if (waterLevel > THRESHOLD_NIVEL_CRITICAL) emergencyScore += 0.2f;
    
    currentRiskScore = min(1.0f, emergencyScore);
    
    if (criticalFlood) {
        currentAlertLevel = 3;  // Emergência
        
        // Alerta máximo mesmo com bateria baixa
        digitalWrite(LED_VERMELHO_PIN, HIGH);
        digitalWrite(LED_VERDE_PIN, LOW);
        digitalWrite(LED_AMARELO_PIN, LOW);
        digitalWrite(LED_LARANJA_PIN, LOW);
        
        // Piscar todos os LEDs 3 vezes para chamar atenção
        for(int i = 0; i < 3; i++) {
            digitalWrite(LED_VERDE_PIN, HIGH);
            digitalWrite(LED_AMARELO_PIN, HIGH);
            digitalWrite(LED_LARANJA_PIN, HIGH);
            delay(300);
            digitalWrite(LED_VERDE_PIN, LOW);
            digitalWrite(LED_AMARELO_PIN, LOW);
            digitalWrite(LED_LARANJA_PIN, LOW);
            delay(300);
        }
        digitalWrite(LED_VERMELHO_PIN, HIGH); // Manter vermelho aceso
        
        // Som de emergência
        emergencySound();
        
        Serial.printf("⚠️ ENCHENTE DETECTADA! Nível: %.2fm | Chuva: %.1fmm/h\n", 
                     waterLevel, precipitation);
    } else {
        currentAlertLevel = 0;
        // LED verde fraco para indicar funcionamento
        digitalWrite(LED_VERDE_PIN, HIGH);
        digitalWrite(LED_AMARELO_PIN, LOW);
        digitalWrite(LED_LARANJA_PIN, LOW);
        digitalWrite(LED_VERMELHO_PIN, LOW);
    }
    
    // Log mínimo
    Serial.printf("🚨 Emergência - Bat: %.1fV | Nível: %.2fm | Score: %.2f\n",
                 batteryVoltage, waterLevel, currentRiskScore);
}

// ============================
// FUNÇÕES DE LEITURA DOS SENSORES
// ============================
SensorData readAllSensors() {
    SensorData data;
    data.timestamp = millis();
    
    // Ler DHT22 (temperatura e umidade)
    data.temperature = dht.readTemperature();
    data.humidity = dht.readHumidity();
    
    // Verificar se leituras são válidas
    if (isnan(data.temperature)) data.temperature = 25.0; // Valor padrão
    if (isnan(data.humidity)) data.humidity = 60.0;
    
    // Ler BMP280 (pressão)
    data.pressure = bmp.readPressure() / 100.0F;  // Converter para hPa
    if (data.pressure < 800 || data.pressure > 1200) data.pressure = 1013.25; // Valor padrão
    
    // Ler sensor de chuva
    data.precipitation = readRainSensor();
    
    // Ler nível de água (ultrassônico)
    data.waterLevel = readWaterLevel();
    
    // Calcular taxa de subida da água
    data.waterRiseRate = calculateWaterRiseRate(data.waterLevel);
    
    // Ler umidade do solo
    int soilRaw = analogRead(SOIL_MOISTURE_PIN);
    data.soilMoisture = map(soilRaw, 0, 4095, 0, 100);
    
    // Simular velocidade do vento (ou adicionar sensor real)
    data.windSpeed = random(0, 20);
    
    return data;
}

float readRainSensor() {
    int rainRaw = analogRead(RAIN_SENSOR_PIN);
    // Converter leitura analógica para mm/h
    float precipitation = map(rainRaw, 0, 4095, 0, 100) / 10.0;
    return max(0.0f, precipitation);
}

float readWaterLevel() {
    // Função para ler sensor ultrassônico HC-SR04
    digitalWrite(ULTRASONIC_TRIG_PIN, LOW);
    delayMicroseconds(2);
    digitalWrite(ULTRASONIC_TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(ULTRASONIC_TRIG_PIN, LOW);
    
    long duration = pulseIn(ULTRASONIC_ECHO_PIN, HIGH, 30000); // Timeout 30ms
    
    if (duration == 0) {
        Serial.println("⚠️ Sensor ultrassônico: Timeout");
        return 1.0; // Valor padrão seguro
    }
    
    float distance = duration * 0.034 / 2;  // Converter para cm
    
    // Converter distância para nível de água (ajustar conforme instalação)
    float maxHeight = 500;  // Altura máxima do sensor em cm - AJUSTAR CONFORME INSTALAÇÃO
    float waterLevel = (maxHeight - distance) / 100.0;  // Converter para metros
    
    return max(0.0, min(10.0, waterLevel)); // Limitar entre 0 e 10 metros
}

float calculateWaterRiseRate(float currentLevel) {
    if (historyCount < 2) return 0.0;
    
    // Pegar nível anterior
    int prevIndex = (historyIndex - 1 + HISTORY_SIZE) % HISTORY_SIZE;
    float prevLevel = sensorHistory[prevIndex].waterLevel;
    
    // Calcular taxa de subida em m/h
    float timeDiff = getMeasurementInterval() / 1000.0; // Segundos
    float ratePerSecond = (currentLevel - prevLevel) / timeDiff;
    return ratePerSecond * 3600.0;  // Converter para m/h
}

// ============================
// PREPARAÇÃO DOS DADOS PARA ML
// ============================
SAEAFeatures prepareMLFeatures(SensorData currentData) {
    SAEAFeatures features;
    
    // Features básicas dos sensores
    features.precipitation_mm = currentData.precipitation;
    features.temperature_c = currentData.temperature;
    features.humidity_pct = currentData.humidity;
    features.pressure_hpa = currentData.pressure;
    features.wind_speed_kmh = currentData.windSpeed;
    features.soil_moisture_pct = currentData.soilMoisture;
    features.nivel_metros = currentData.waterLevel;
    features.taxa_subida_m_h = currentData.waterRiseRate;
    
    // Features temporais
    struct tm timeinfo;
    time_t now = currentData.timestamp / 1000;
    localtime_r(&now, &timeinfo);
    
    features.hora = timeinfo.tm_hour;
    features.dia_semana = timeinfo.tm_wday;
    features.mes = timeinfo.tm_mon + 1;
    features.hora_sin = sin(2 * PI * features.hora / 24);
    features.hora_cos = cos(2 * PI * features.hora / 24);
    features.mes_sin = sin(2 * PI * features.mes / 12);
    features.mes_cos = cos(2 * PI * features.mes / 12);
    
    // Calcular médias móveis e features derivadas
    calculateMovingAverages(features);
    
    return features;
}

void calculateMovingAverages(SAEAFeatures& features) {
    if (historyCount == 0) {
        // Valores padrão se não há histórico
        features.precipitation_mm_ma_12h = features.precipitation_mm;
        features.nivel_metros_ma_12h = features.nivel_metros;
        features.precipitation_mm_ma_24h = features.precipitation_mm;
        features.nivel_metros_ma_24h = features.nivel_metros;
        features.precip_24h = features.precipitation_mm * 24;
        features.precip_48h = features.precip_24h * 2;
        return;
    }
    
    // Calcular médias das últimas medições
    float sum_precip_12h = 0, sum_nivel_12h = 0;
    float sum_precip_24h = 0, sum_nivel_24h = 0;
    
    int count_12h = min(12, historyCount);
    int count_24h = min(24, historyCount);
    
    for (int i = 0; i < count_12h; i++) {
        int idx = (historyIndex - 1 - i + HISTORY_SIZE) % HISTORY_SIZE;
        sum_precip_12h += sensorHistory[idx].precipitation;
        sum_nivel_12h += sensorHistory[idx].waterLevel;
    }
    
    for (int i = 0; i < count_24h; i++) {
        int idx = (historyIndex - 1 - i + HISTORY_SIZE) % HISTORY_SIZE;
        sum_precip_24h += sensorHistory[idx].precipitation;
        sum_nivel_24h += sensorHistory[idx].waterLevel;
    }
    
    features.precipitation_mm_ma_12h = sum_precip_12h / count_12h;
    features.nivel_metros_ma_12h = sum_nivel_12h / count_12h;
    features.precipitation_mm_ma_24h = sum_precip_24h / count_24h;
    features.nivel_metros_ma_24h = sum_nivel_24h / count_24h;
    
    // Features agregadas
    features.precip_24h = sum_precip_24h;
    features.precip_48h = features.precip_24h * 2;  // Estimativa
    
    // Features derivadas simples
    features.precipitation_mm_std_12h = features.precipitation_mm * 0.3;
    features.nivel_metros_std_12h = features.nivel_metros * 0.1;
    features.precipitation_mm_diff = features.precipitation_mm * 0.2;
    features.nivel_metros_diff = features.nivel_metros * 0.05;
    features.aceleracao_nivel = features.taxa_subida_m_h * 0.1;
    features.pressao_normalizada = (features.pressure_hpa - 1013.25) / 50.0;
}

// ============================
// GERENCIAMENTO DE HISTÓRICO
// ============================
void addToHistory(SensorData data) {
    sensorHistory[historyIndex] = data;
    historyIndex = (historyIndex + 1) % HISTORY_SIZE;
    if (historyCount < HISTORY_SIZE) historyCount++;
}

// ============================
// ALERTAS VISUAIS E SONOROS
// ============================
void updateAlerts(int alertLevel) {
    static unsigned long lastBlink = 0;
    static bool blinkState = false;
    unsigned long currentTime = millis();
    
    // Desligar todos os LEDs primeiro
    digitalWrite(LED_VERDE_PIN, LOW);
    digitalWrite(LED_AMARELO_PIN, LOW);
    digitalWrite(LED_LARANJA_PIN, LOW);
    digitalWrite(LED_VERMELHO_PIN, LOW);
    
    switch (alertLevel) {
        case 0: // 🟢 VERDE - Normal
            digitalWrite(LED_VERDE_PIN, HIGH);
            break;
            
        case 1: // 🟡 AMARELO - Atenção
            digitalWrite(LED_AMARELO_PIN, HIGH);
            break;
            
        case 2: // 🟠 LARANJA - Alerta
            // Piscar laranja a cada 1 segundo
            if (currentTime - lastBlink > 1000) {
                blinkState = !blinkState;
                lastBlink = currentTime;
            }
            digitalWrite(LED_LARANJA_PIN, blinkState ? HIGH : LOW);
            break;
            
        case 3: // 🔴 VERMELHO - Emergência
            // Piscar vermelho rápido (0.3s)
            if (currentTime - lastBlink > 300) {
                blinkState = !blinkState;
                lastBlink = currentTime;
            }
            digitalWrite(LED_VERMELHO_PIN, blinkState ? HIGH : LOW);
            
            // Se risco muito alto, piscar outros LEDs também
            if (currentRiskScore > 0.9) {
                digitalWrite(LED_AMARELO_PIN, blinkState ? HIGH : LOW);
                digitalWrite(LED_LARANJA_PIN, blinkState ? HIGH : LOW);
            }
            break;
    }
}

void updatePowerLED() {
    static unsigned long lastBlink = 0;
    static bool blinkState = false;
    unsigned long currentTime = millis();
    
    if (isCharging) {
        // Piscar verde quando carregando
        if (currentTime - lastBlink > 500) {
            blinkState = !blinkState;
            digitalWrite(POWER_LED_PIN, blinkState);
            lastBlink = currentTime;
        }
    } else {
        // LED baseado no nível da bateria
        if (batteryVoltage >= BATTERY_GOOD) {
            digitalWrite(POWER_LED_PIN, HIGH);  // Verde fixo
        } else if (batteryVoltage >= BATTERY_LOW) {
            // Piscar lento (bateria baixa)
            if (currentTime - lastBlink > 2000) {
                blinkState = !blinkState;
                digitalWrite(POWER_LED_PIN, blinkState);
                lastBlink = currentTime;
            }
        } else {
            // Piscar rápido (crítico)
            if (currentTime - lastBlink > 300) {
                blinkState = !blinkState;
                digitalWrite(POWER_LED_PIN, blinkState);
                lastBlink = currentTime;
            }
        }
    }
}

void soundAlert(int alertLevel) {
    if (currentPowerMode == POWER_EMERGENCY) return; // Economia na emergência
    
    switch(alertLevel) {
        case 0: // Silêncio
            break;
        case 1: // 1 beep curto
            tone(BUZZER_PIN, 1000, 200);
            break;
        case 2: // 2 beeps
            tone(BUZZER_PIN, 1500, 200);
            delay(300);
            tone(BUZZER_PIN, 1500, 200);
            break;
        case 3: // Sirene contínua
            for(int i = 0; i < 3; i++) {
                tone(BUZZER_PIN, 2000, 150);
                delay(200);
                tone(BUZZER_PIN, 1000, 150);
                delay(200);
            }
            break;
    }
}

void emergencySound() {
    // Som de emergência especial
    for(int i = 0; i < 5; i++) {
        tone(BUZZER_PIN, 2500, 100);
        delay(120);
        tone(BUZZER_PIN, 1500, 100);
        delay(120);
    }
}

void playStartupSound() {
    tone(BUZZER_PIN, 1000, 100);
    delay(150);
    tone(BUZZER_PIN, 1500, 100);
    delay(150);
    tone(BUZZER_PIN, 2000, 200);
}

void playModeChangeSound(PowerMode mode) {
    switch(mode) {
        case POWER_NORMAL:
            tone(BUZZER_PIN, 1500, 200);
            break;
        case POWER_ECONOMY:
            tone(BUZZER_PIN, 1200, 200);
            break;
        case POWER_CRITICAL:
            tone(BUZZER_PIN, 800, 300);
            break;
        case POWER_EMERGENCY:
            tone(BUZZER_PIN, 500, 500);
            break;
    }
}

// ============================
// TESTE DE HARDWARE
// ============================
void testAllLEDs() {
    Serial.println("🔧 Testando LEDs...");
    
    // Teste sequencial
    digitalWrite(LED_VERDE_PIN, HIGH);
    delay(500);
    digitalWrite(LED_VERDE_PIN, LOW);
    
    digitalWrite(LED_AMARELO_PIN, HIGH);
    delay(500);
    digitalWrite(LED_AMARELO_PIN, LOW);
    
    digitalWrite(LED_LARANJA_PIN, HIGH);
    delay(500);
    digitalWrite(LED_LARANJA_PIN, LOW);
    
    digitalWrite(LED_VERMELHO_PIN, HIGH);
    delay(500);
    digitalWrite(LED_VERMELHO_PIN, LOW);
    
    digitalWrite(POWER_LED_PIN, HIGH);
    delay(500);
    digitalWrite(POWER_LED_PIN, LOW);
    
    Serial.println("✅ Teste de LEDs concluído");
}

// ============================
// CONECTIVIDADE
// ============================
void connectWiFi() {
    if (currentPowerMode == POWER_EMERGENCY) return; // Não tentar WiFi na emergência
    
    Serial.print("Conectando ao WiFi...");
    WiFi.begin(ssid, password);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        wifiConnected = true;
        Serial.println(" ✅ Conectado!");
        Serial.print("IP: ");
        Serial.println(WiFi.localIP());
    } else {
        wifiConnected = false;
        Serial.println(" ❌ Falha na conexão");
    }
}

void sendDataToServer(SensorData data, int prediction, float riskScore, int alertLevel) {
    if (!wifiConnected || currentPowerMode >= POWER_CRITICAL) return;
    
    HTTPClient http;
    http.begin(serverURL);
    http.addHeader("Content-Type", "application/json");
    
    // Criar JSON com dados
    StaticJsonDocument<1024> doc;
    doc["device_id"] = "SAEA_ESP32_Solar_001";
    doc["timestamp"] = data.timestamp;
    doc["sensors"]["temperature"] = data.temperature;
    doc["sensors"]["humidity"] = data.humidity;
    doc["sensors"]["pressure"] = data.pressure;
    doc["sensors"]["precipitation"] = data.precipitation;
    doc["sensors"]["water_level"] = data.waterLevel;
    doc["sensors"]["water_rise_rate"] = data.waterRiseRate;
    doc["prediction"]["risk_score"] = riskScore;
    doc["prediction"]["alert_level"] = alertLevel;
    doc["prediction"]["flood_predicted"] = prediction;
    
    // Dados de energia
    doc["power"]["battery_voltage"] = batteryVoltage;
    doc["power"]["solar_voltage"] = solarVoltage;
    doc["power"]["is_charging"] = isCharging;
    doc["power"]["mode"] = getPowerModeDescription(currentPowerMode);
    
    String jsonString;
    serializeJson(doc, jsonString);
    
    int httpResponseCode = http.POST(jsonString);
    
    if (httpResponseCode > 0) {
        Serial.printf("✅ Dados enviados: %d\n", httpResponseCode);
    } else {
        Serial.printf("❌ Erro no envio: %d\n", httpResponseCode);
    }
    
    http.end();
}

// ============================
// DISPLAY
// ============================
void updateDisplayWithPower(String title, String message) {
    if (!displayEnabled || currentPowerMode == POWER_EMERGENCY) return;
    
    display.clear();
    
    // Título
    display.setTextAlignment(TEXT_ALIGN_CENTER);
    display.setFont(ArialMT_Plain_16);
    display.drawString(64, 0, title);
    
    // Mensagem principal
    display.setFont(ArialMT_Plain_12);
    display.drawString(64, 18, message);
    
    // Status de energia
    display.setTextAlignment(TEXT_ALIGN_LEFT);
    display.setFont(ArialMT_Plain_10);
    
    // Linha 1: Bateria e carregamento
    String batteryStr = "Bat: " + String(batteryVoltage, 1) + "V";
    if (isCharging) batteryStr += " ⚡";
    display.drawString(0, 35, batteryStr);
    
    // Barra visual da bateria
    int batteryPercent = map(batteryVoltage * 10, 30, 42, 0, 100);
    batteryPercent = constrain(batteryPercent, 0, 100);
    
    display.drawRect(70, 35, 52, 8);
    int fillWidth = map(batteryPercent, 0, 100, 0, 50);
    display.fillRect(71, 36, fillWidth, 6);
    
    // Linha 2: Modo de energia e alerta
    String modeStr = getPowerModeDescription(currentPowerMode);
    display.drawString(0, 45, modeStr);
    
    // Status do alerta
    display.setTextAlignment(TEXT_ALIGN_RIGHT);
    String alertStr = "A" + String(currentAlertLevel);
    display.drawString(128, 45, alertStr);
    
    // Linha 3: WiFi e Solar
    display.setTextAlignment(TEXT_ALIGN_LEFT);
    String statusStr = wifiConnected ? "WiFi ✓" : "Offline";
    if (solarVoltage > 3.0) statusStr += " ☀️";
    display.drawString(0, 55, statusStr);
    
    display.display();
}

// ============================
// ECONOMIA DE ENERGIA
// ============================
void applyPowerSaving() {
    switch(currentPowerMode) {
        case POWER_NORMAL:
            delay(1000);  // 1 segundo normal
            break;
            
        case POWER_ECONOMY:
            delay(3000);  // 3 segundos economia
            break;
            
        case POWER_CRITICAL:
            delay(5000);  // 5 segundos crítico
            break;
            
        case POWER_EMERGENCY:
            // Deep sleep curto entre operações
            delay(10000); // 10 segundos
            break;
    }
}

// ============================
// LOGGING E DEBUG
// ============================
void logCurrentStatus(SensorData data, int prediction, float riskScore, int alertLevel) {
    // Log completo apenas no modo normal, resumido nos outros
    if (currentPowerMode == POWER_NORMAL) {
        Serial.println("\n" + String("=").substring(0, 50));
        Serial.println("🌊☀️ SAEA Solar - Status Completo");
        Serial.println(String("=").substring(0, 50));
        
        Serial.printf("⏰ Tempo: %lu ms\n", data.timestamp);
        Serial.printf("🌡️ Temperatura: %.1f°C\n", data.temperature);
        Serial.printf("💧 Umidade: %.1f%%\n", data.humidity);
        Serial.printf("📊 Pressão: %.1f hPa\n", data.pressure);
        Serial.printf("🌧️ Chuva: %.1f mm/h\n", data.precipitation);
        Serial.printf("📏 Nível: %.2f m\n", data.waterLevel);
        Serial.printf("📈 Taxa subida: %.3f m/h\n", data.waterRiseRate);
        
        Serial.printf("\n🧠 PREDIÇÃO ML:\n");
        Serial.printf("   Score: %.3f\n", riskScore);
        Serial.printf("   Alerta: %d (%s)\n", alertLevel, getAlertDescription(alertLevel));
        Serial.printf("   Enchente: %s\n", prediction ? "SIM" : "NÃO");
        
        Serial.printf("\n⚡ ENERGIA:\n");
        Serial.printf("   Bateria: %.2fV (%s)\n", batteryVoltage, getBatteryStatus());
        Serial.printf("   Solar: %.2fV (%s)\n", solarVoltage, isCharging ? "Carregando" : "Inativo");
        Serial.printf("   Modo: %s\n", getPowerModeDescription(currentPowerMode));
        
        Serial.printf("\n📶 Conectividade: %s\n", wifiConnected ? "WiFi ✓" : "Offline");
        Serial.println(String("=").substring(0, 50));
        
    } else {
        // Log resumido para outros modos
        Serial.printf("🌊 Nível: %.2fm | Risco: %.2f | Bat: %.1fV (%s)\n",
                     data.waterLevel, riskScore, batteryVoltage, 
                     getPowerModeDescription(currentPowerMode));
    }
}

void testMLModel() {
    Serial.println("🧪 Testando modelo ML...");
    
    // Criar dados de teste
    SAEAFeatures testData;
    memset(&testData, 0, sizeof(SAEAFeatures));
    
    testData.precipitation_mm = 25.0;
    testData.temperature_c = 28.0;
    testData.humidity_pct = 85.0;
    testData.pressure_hpa = 1008.0;
    testData.wind_speed_kmh = 15.0;
    testData.soil_moisture_pct = 75.0;
    testData.nivel_metros = 2.5;
    testData.taxa_subida_m_h = 0.15;
    testData.precip_24h = 80.0;
    testData.precip_48h = 120.0;
    
    if (validateFeatures(testData)) {
        int prediction = predictFlood(testData);
        float score = calculateRiskScore(testData);
        int alert = getAlertLevel(testData);
        
        Serial.printf("✅ Teste ML: Score=%.3f, Alerta=%d, Predição=%d\n", 
                     score, alert, prediction);
    } else {
        Serial.println("❌ Dados de teste inválidos");
    }
}