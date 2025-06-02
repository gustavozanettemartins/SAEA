# 🌊 SAEA - Sistema Autônomo de Alerta de Enchentes

> **Tecnologia que fala a língua da comunidade - seja por luz, som ou rádio FM - salvando vidas onde a modernidade não chega.**

[![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow)](https://github.com/gustavozanettemartins/SAEA)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![ESP32](https://img.shields.io/badge/ESP32-Compatible-green)](https://www.espressif.com/en/products/socs/esp32)
[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)

## 📋 Sobre o Projeto

O SAEA é um sistema de previsão e alerta de enchentes desenvolvido para comunidades isoladas sem infraestrutura tecnológica. Combina Machine Learning com dados reais do [DisasterCharter.org](https://disasterscharter.org/) e IoT com ESP32 para criar uma solução autônoma, acessível e adaptativa.

### 🎯 Problema

Comunidades isoladas sofrem com enchentes sem aviso prévio, não tendo acesso a:
- Sistemas de alerta modernos
- Internet confiável
- Eletricidade estável
- Meios de comunicação tradicionais

### 💡 Solução

Sistema autônomo que:
- **Detecta** níveis de água com sensor ultrassônico
- **Processa** dados localmente com ML embarcado
- **Alerta** via sirene, luzes LED e transmissor FM
- **Aprende** com feedback da comunidade
- **Funciona** com energia solar

## 🚀 Características Principais

### 🤖 Machine Learning
- Modelo treinado com dados históricos de enchentes
- Random Forest otimizado para dispositivos embarcados
- Previsão em 4 níveis: Normal, Amarelo, Laranja, Vermelho
- Taxa de precisão superior a 85%

### 📡 Sistema IoT
- **ESP32** como unidade de processamento
- **Sensor ultrassônico HC-SR04** para medição de nível
- **LEDs coloridos** para alerta visual local
- **Sirene de alta potência** para alerta sonoro
- **Transmissor FM** (opcional) para alcance ampliado

### 🔋 Autonomia Energética
- Painel solar 6V 3W
- Bateria 18650 recarregável
- Modo deep sleep para economia
- Autonomia de 7+ dias sem sol

### 🔄 Sistema Adaptativo
- 3 botões para feedback comunitário
- Ajuste automático de sensibilidade
- Redução de falsos alarmes
- Melhoria contínua com uso

## 📊 Arquitetura do Sistema

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Coleta de Dados │────▶│ Processamento ML │────▶│ Sistema de      │
│ - DisasterCharter│     │ - Random Forest  │     │ Alertas         │
│ - Meteorológicos │     │ - Validação      │     │ - Visual (LED)  │
│ - Hidrológicos   │     │ - Otimização     │     │ - Sonoro        │
└─────────────────┘     └──────────────────┘     │ - FM (opcional) │
                                                  └─────────────────┘
                               ▲                            │
                               │                            ▼
                    ┌──────────┴──────────┐     ┌─────────────────┐
                    │ ESP32 + Sensores    │     │ Feedback        │
                    │ - Ultrassônico      │────▶│ Comunitário     │
                    │ - Processamento     │     │ - Recalibração  │
                    │ - Deep Sleep        │     │ - Aprendizado   │
                    └─────────────────────┘     └─────────────────┘
```

## 🛠️ Tecnologias Utilizadas

### Software
- **Python 3.8+** - Machine Learning e análise de dados
- **Arduino IDE** - Programação do ESP32
- **Jupyter Notebook** - Desenvolvimento e documentação
- **Pandas, NumPy, Scikit-learn** - Processamento de dados
- **XGBoost/Random Forest** - Modelos de ML

### Hardware
- **ESP32** - Microcontrolador principal
- **HC-SR04** - Sensor ultrassônico
- **LEDs de alta luminosidade** - Sistema visual
- **Buzzer + Sirene 12V** - Sistema sonoro
- **KT0803L** - Transmissor FM (opcional)
- **TP4056** - Carregador de bateria
- **Painel Solar 6V** - Alimentação autônoma

## 📁 Estrutura do Projeto

```
SAEA-Sistema-Alerta-Enchentes/
├── 📄 README.md                    # Este arquivo
├── 📄 LICENSE                      # Licença MIT
├── 📁 hardware/                    # Arquivos de hardware
│   ├── 📁 firmware_esp32/         
│   │   └── 📄 saea_main.ino       # Código principal ESP32
│   ├── 📁 esquematicos/           
│   │   └── 📄 circuito_saea.pdf   # Esquema elétrico
│   └── 📄 lista_componentes.txt    # BOM completo
├── 📁 machine_learning/            # Código ML
│   ├── 📁 notebooks/              
│   │   ├── 📄 01_coleta_dados.ipynb
│   │   ├── 📄 02_analise_exploratoria.ipynb
│   │   ├── 📄 03_treinamento_modelo.ipynb
│   │   └── 📄 04_validacao_deploy.ipynb
│   ├── 📁 src/                    
│   │   ├── 📄 preprocessamento.py
│   │   ├── 📄 treinamento.py
│   │   └── 📄 api_predicao.py
│   ├── 📁 models/                 
│   │   └── 📄 modelo_final.pkl
│   └── 📄 requirements.txt
├── 📁 docs/                        # Documentação
│   ├── 📄 relatorio_final.pdf
│   ├── 📄 manual_instalacao.md
│   └── 📄 protocolo_alertas.md
└── 📁 tests/                       # Testes
    └── 📄 test_sistema.py
```

## 🚀 Como Começar

### Pré-requisitos

#### Hardware
- ESP32
- Sensor HC-SR04
- LEDs (vermelho, amarelo, laranja, verde)
- Buzzer + resistores
- Protoboard e jumpers

#### Software
```bash
# Python e bibliotecas
Python 3.8+
pip install -r machine_learning/requirements.txt

# Arduino IDE com suporte ESP32
https://randomnerdtutorials.com/installing-the-esp32-board-in-arduino-ide-windows-instructions/
```

### Instalação

1. **Clone o repositório**
```bash
git clone https://github.com/gustavozanettemartins/SAEA.git
cd SAEA-Sistema-Alerta-Enchentes
```

2. **Configure o ambiente Python**
```bash
cd machine_learning
pip install -r requirements.txt
```

3. **Execute os notebooks de ML**
```bash
jupyter notebook notebooks/01_coleta_dados.ipynb
```

4. **Programe o ESP32**
- Abra `hardware/firmware_esp32/saea_main.ino` no Arduino IDE
- Selecione a placa "ESP32 Dev Module"
- Configure a porta COM correta
- Faça o upload do código

### Montagem do Hardware

1. **Conecte os componentes** conforme esquemático em `hardware/esquematicos/`
2. **Alimente o sistema** via USB (desenvolvimento) ou bateria (produção)
3. **Execute o teste inicial** - LEDs devem piscar em sequência

## 📊 Dados e Resultados

### Dataset
- **Fonte principal**: DisasterCharter.org
- **Período**: 2020-2025
- **Registros**: 100.000+
- **Features**: 25 variáveis meteorológicas e hidrológicas

### Performance do Modelo
- **Precisão geral**: 87.3%
- **Recall (enchentes)**: 92.1%
- **F1-Score**: 0.89
- **Falsos negativos**: < 8%

### Autonomia do Sistema
- **Modo normal**: 10 dias sem recarga
- **Modo alerta**: 3 dias contínuos
- **Tempo de resposta**: < 1 segundo

## 🎯 Protocolo de Alertas

### Níveis de Alerta

| Nível | Cor | Som | Ação Recomendada |
|-------|-----|-----|------------------|
| **Normal** | 🟢 Verde | Silencioso | Monitoramento padrão |
| **Amarelo** | 🟡 Amarelo | 2 bipes | Preparar comunidade |
| **Laranja** | 🟠 Laranja | 3 bipes + sirene 2s | Mobilizar recursos |
| **Vermelho** | 🔴 Vermelho | 5 bipes + sirene 5s | Evacuar imediatamente |

### Alcance dos Alertas
- **Visual (LED)**: 50 metros
- **Sonoro (sirene)**: 500 metros
- **FM (opcional)**: 2-5 km

## 🔧 Manutenção

### Rotina Mensal
- [ ] Limpar sensor ultrassônico
- [ ] Verificar conexões
- [ ] Testar botões de feedback
- [ ] Limpar painel solar

### Calibração
- Pressione o botão de ajuste para ciclar entre sensibilidades
- Use botões de feedback para reportar falsos alarmes
- Sistema se ajusta automaticamente após 3 feedbacks

## 📈 Impacto Social

### Números Esperados
- **Custo por unidade**: < R$ 200
- **Pessoas impactadas**: 500-1000 por dispositivo
- **Redução de perdas**: Estimada em 70%
- **Tempo de aviso**: 2-6 horas antes do evento

### Comunidades Alvo
- Regiões ribeirinhas isoladas
- Comunidades sem infraestrutura
- Áreas de difícil acesso
- Populações vulneráveis

## 👥 Equipe

- **[Gustavo Zanette Martins]** - Machine Learning

## 📝 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🤝 Como Contribuir

1. Faça um Fork do projeto
2. Crie sua Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a Branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📞 Contato

- **Email**: gzanettemartins@gmail.com
- **GitHub**: [SAEA-Sistema-Alerta-Enchentes](https://github.com/gustavozanettemartins/SAEA/)

## 🏆 Reconhecimentos

- **FIAP** - Global Solutions 2025
- **DisasterCharter.org** - Dados de desastres
- **Comunidades parceiras** - Feedback e validação

---

<div align="center">
  
**💙 Desenvolvido com amor para salvar vidas 💙**

*"A tecnologia só faz sentido quando chega a quem mais precisa"*

</div>
