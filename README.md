# 🌊 SAEA - Sistema Autônomo de Alerta de Enchentes

Um sistema inteligente de previsão e alerta de enchentes baseado em Machine Learning, IoT e dados meteorológicos em tempo real, desenvolvido especificamente para o Rio Grande do Sul.

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Características](#características)
- [Arquitetura](#arquitetura)
- [Instalação](#instalação)
- [Uso Rápido](#uso-rápido)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Contribuição](#contribuição)
- [Licença](#licença)

## 🎯 Sobre o Projeto

O SAEA é um sistema completo que combina:
- **Machine Learning** para previsão de enchentes
- **Dispositivos IoT** (ESP32) para coleta de dados
- **Dados oficiais do INMET** para treinamento
- **Alertas em tempo real** com diferentes níveis de criticidade

### 🏆 Diferenciais

- ✅ **Integração direta com dados do INMET**
- ✅ **Otimizado para o Rio Grande do Sul**
- ✅ **Modelo embarcado para ESP32**
- ✅ **Três modos de threshold adaptativos**
- ✅ **Simulações completas de hardware**
- ✅ **Pipeline automatizado end-to-end**

## 🚀 Características

### 📊 Machine Learning Avançado
- Modelos otimizados para dados desbalanceados
- Ensemble de algoritmos (Random Forest, XGBoost, Gradient Boosting)
- Feature engineering temporal sofisticado
- Validação rigorosa com foco em minimizar falsos negativos

### 🌡️ Sensores e IoT
- Suporte para múltiplos sensores meteorológicos
- Comunicação WiFi/LoRaWAN
- Baixo consumo energético
- Tolerância a falhas de conectividade

### 🚨 Sistema de Alertas
- **Verde**: Condições normais
- **Amarelo**: Atenção - acompanhar evolução
- **Laranja**: Alerta - preparar ações preventivas  
- **Vermelho**: Emergência - evacuar área de risco

### 🔧 Modos de Operação
- **Emergência** (threshold: 0.05): Máxima sensibilidade
- **Urbano** (threshold: 0.50): Balanceado
- **Conservador** (threshold: 0.80): Mínimos alarmes falsos

## 🏗️ Arquitetura

O sistema é composto por três componentes principais:

1. **Coleta de Dados**
   - Scripts de processamento de dados INMET
   - Simulação de sensores ESP32
   - Integração com dados hidrológicos

2. **Processamento e ML**
   - Pipeline de preprocessamento
   - Treinamento de modelos
   - Otimização de thresholds

3. **Simulação e Testes**
   - Simulador de cenários
   - Testes de hardware
   - Análise de resultados

## 💻 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/saea.git
cd saea
```

2. Crie um ambiente virtual Python:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Execute o setup inicial:
```bash
python src/initial_setup.py
```

## 📥 Dados Históricos do INMET

Para obter os dados históricos do INMET necessários para o treinamento do sistema:

1. Acesse o [Portal do INMET](https://portal.inmet.gov.br/dadoshistoricos)


## 🚀 Uso Rápido

1. Processar dados INMET:
```bash
python src/script_processar_csv_inmet.py
```

2. Executar preprocessamento:
```bash
python src/preprocessamento.py
```

3. Treinar modelos:
```bash
python src/treinamento.py --data data/processed/dataset_ml_final.csv
```

4. Simular cenários:
```bash
python src/simulacao.py --cenario enchente_moderada --duracao 24
```

## 📁 Estrutura do Projeto

```
saea/
├── config/             # Arquivos de configuração
├── data/              # Dados brutos e processados
│   ├── raw/          # Dados INMET e hidrológicos
│   ├── processed/    # Datasets processados
│   └── disaster_charter/  # Eventos de enchente
├── hardware/         # Código para ESP32
├── logs/            # Logs do sistema
├── models/          # Modelos treinados
├── results/         # Resultados de simulações
└── src/             # Código fonte
    ├── initial_setup.py
    ├── preprocessamento.py
    ├── script_processar_csv_inmet.py
    ├── simulacao.py
    └── treinamento.py
```

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 📧 Contato

Para mais informações, entre em contato através de [gzanettemartins@gmail.com](mailto:gzanettemartins@gmail.com)