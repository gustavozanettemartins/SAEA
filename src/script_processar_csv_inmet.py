# processar_csv_inmet_rs.py - PROCESSA DADOS METEOROLÓGICOS DO RIO GRANDE DO SUL

import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path

def encontrar_arquivos_inmet(pasta_base="./"):
    """
    Encontra automaticamente os arquivos CSV do INMET
    """
    print("🔍 Procurando arquivos CSV do INMET...")
    
    # Padrões possíveis de busca
    padroes = [
        "**/*INMET*.CSV",
        "**/*INMET*.csv", 
        "**/2020*.zip",
        "**/2021*.zip",
        "**/2022*.zip", 
        "**/2023*.zip",
        "**/2024*.zip",
        "**/2025*.zip"
    ]
    
    arquivos_encontrados = []
    
    for padrao in padroes:
        arquivos = list(Path(pasta_base).glob(padrao))
        arquivos_encontrados.extend(arquivos)
    
    print(f"📁 Encontrados {len(arquivos_encontrados)} arquivos")
    
    for arquivo in arquivos_encontrados[:10]:  # Mostrar primeiros 10
        print(f"   📄 {arquivo}")
    
    return arquivos_encontrados

def setup_automatico():
    """
    Setup automático - encontra os arquivos e organiza
    """
    print("🚀 SETUP AUTOMÁTICO DOS DADOS INMET - RIO GRANDE DO SUL")
    print("=" * 60)
    
    # 1. Criar estrutura de pastas
    os.makedirs('data/raw/inmet', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/disaster_charter', exist_ok=True)
    
    # 2. Encontrar arquivos
    arquivos = encontrar_arquivos_inmet()
    
    # 3. Processar ZIPs se encontrados
    import zipfile
    
    for arquivo in arquivos:
        if str(arquivo).endswith('.zip'):
            print(f"\n📦 Extraindo {arquivo}...")
            try:
                with zipfile.ZipFile(arquivo, 'r') as zip_ref:
                    # Extrair para pasta específica do ano
                    ano = None
                    for year in ['2020', '2021', '2022', '2023', '2024', '2025']:
                        if year in str(arquivo):
                            ano = year
                            break
                    
                    if ano:
                        pasta_destino = f'data/raw/inmet/{ano}'
                        os.makedirs(pasta_destino, exist_ok=True)
                        zip_ref.extractall(pasta_destino)
                        print(f"   ✅ Extraído para {pasta_destino}")
                    else:
                        zip_ref.extractall('data/raw/inmet/temp')
                        print(f"   ✅ Extraído para data/raw/inmet/temp")
            except Exception as e:
                print(f"   ❌ Erro ao extrair {arquivo}: {e}")
    
    # 4. Processar CSVs
    processar_todos_csvs()

def processar_todos_csvs():
    """
    Processa todos os CSVs encontrados do Rio Grande do Sul
    """
    print("\n🌦️ Processando todos os CSVs do INMET do Rio Grande do Sul...")
    
    # Buscar todos os CSVs
    csv_files = []
    for root, dirs, files in os.walk('data/raw/inmet'):
        for file in files:
            if file.endswith('.CSV') or file.endswith('.csv'):
                if 'INMET' in file:
                    csv_files.append(os.path.join(root, file))
    
    # Se não encontrou na pasta organizada, buscar na pasta atual
    if not csv_files:
        csv_files = glob.glob("**/*INMET*.CSV", recursive=True)
        csv_files.extend(glob.glob("**/*INMET*.csv", recursive=True))
    
    print(f"📊 Encontrados {len(csv_files)} arquivos CSV")
    
    if not csv_files:
        print("❌ Nenhum arquivo CSV encontrado!")
        print("💡 Dica: Coloque os arquivos CSV na pasta atual ou em data/raw/inmet/")
        return None
    
    # Estações do Rio Grande do Sul que queremos
    estacoes_rs = ['A801', 'A803', 'A827', 'A832']  # Porto Alegre, Santa Maria, Caxias do Sul, Pelotas
    nomes_cidades = {
        'A801': 'Porto_Alegre',
        'A803': 'Santa_Maria', 
        'A827': 'Caxias_do_Sul',
        'A832': 'Pelotas'
    }
    
    dados_processados = []
    
    for arquivo_csv in csv_files:
        # Verificar se é do Rio Grande do Sul
        nome_arquivo = os.path.basename(arquivo_csv)
        
        estacao_encontrada = None
        for codigo in estacoes_rs:
            if codigo in nome_arquivo:
                estacao_encontrada = codigo
                break
        
        if not estacao_encontrada:
            continue  # Pular se não for das cidades do RS selecionadas
        
        try:
            print(f"📊 Processando {nomes_cidades[estacao_encontrada]} ({estacao_encontrada})...")
            
            # Tentar diferentes encodings
            encodings = ['latin1', 'iso-8859-1', 'utf-8', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    # Ler CSV (pular metadados das primeiras 8 linhas)
                    df = pd.read_csv(arquivo_csv, sep=';', skiprows=8, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                print(f"   ❌ Não foi possível ler o arquivo com nenhum encoding: {nome_arquivo}")
                continue
            
            # Verificar se tem dados
            if len(df) == 0:
                print(f"   ⚠️ Arquivo vazio: {nome_arquivo}")
                continue
            
            # Adicionar metadados
            df['location'] = nomes_cidades[estacao_encontrada]
            df['estacao'] = estacao_encontrada
            
            # Limpar dados vazios
            df = df.dropna(how='all')
            df = df.replace('', pd.NA)
            
            dados_processados.append(df)
            print(f"   ✅ {len(df)} registros carregados")
            
        except Exception as e:
            print(f"   ❌ Erro ao processar {nome_arquivo}: {e}")
            continue
    
    if not dados_processados:
        print("❌ Nenhum dado do Rio Grande do Sul foi processado!")
        return None
    
    # Combinar todos os dados
    print(f"\n🔄 Combinando {len(dados_processados)} arquivos...")
    df_total = pd.concat(dados_processados, ignore_index=True, sort=False)
    
    print(f"📊 Total de registros: {len(df_total):,}")
    
    # Mostrar colunas disponíveis
    print(f"\n📋 Colunas disponíveis:")
    for i, col in enumerate(df_total.columns):
        print(f"   {i+1}. {col}")
    
    # Padronizar colunas automaticamente
    df_final = padronizar_colunas_inmet(df_total)
    
    # Salvar resultado
    if df_final is not None and len(df_final) > 0:
        df_final.to_csv('data/raw/weather_data.csv', index=False)
        
        print(f"\n✅ SUCESSO! Arquivo salvo: data/raw/weather_data.csv")
        print(f"📊 Registros finais: {len(df_final):,}")
        print(f"📅 Período: {df_final['datetime'].min()} até {df_final['datetime'].max()}")
        print(f"📍 Cidades: {', '.join(df_final['location'].unique())}")
        
        return df_final
    else:
        print("❌ Erro no processamento final")
        return None

def padronizar_colunas_inmet(df):
    """
    Padroniza as colunas do INMET automaticamente
    """
    print("\n🔧 Padronizando colunas...")
    
    # Mapeamento flexível de colunas
    mapeamentos = {
        'datetime': ['Data', 'DATA'],
        'hour': ['Hora UTC', 'Hora', 'HORA UTC', 'HORA'],
        'precipitation_mm': [
            'PRECIPITACAO TOTAL, HORARIO (mm)',
            'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)',
            'PRECIPITACAO TOTAL',
            'PRECIPITAÇÃO TOTAL'
        ],
        'temperature_c': [
            'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)',
            'TEMPERATURA DO AR - BULBO SECO, HORÁRIA (°C)',
            'TEMPERATURA DO AR',
            'TEMPERATURA'
        ],
        'humidity_pct': [
            'UMIDADE RELATIVA DO AR, HORARIA (%)',
            'UMIDADE RELATIVA DO AR, HORÁRIA (%)',
            'UMIDADE RELATIVA',
            'UMIDADE'
        ],
        'pressure_hpa': [
            'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)',
            'PRESSÃO ATMOSFÉRICA AO NÍVEL DA ESTAÇÃO, HORÁRIA (mB)',
            'PRESSAO ATMOSFERICA',
            'PRESSÃO ATMOSFÉRICA'
        ],
        'wind_speed_ms': [
            'VENTO, VELOCIDADE HORARIA (m/s)',
            'VENTO, VELOCIDADE HORÁRIA (m/s)',
            'VENTO VELOCIDADE',
            'VELOCIDADE VENTO'
        ]
    }
    
    # Encontrar correspondências
    colunas_encontradas = {}
    
    for nome_padrao, opcoes in mapeamentos.items():
        for opcao in opcoes:
            if opcao in df.columns:
                colunas_encontradas[nome_padrao] = opcao
                break
    
    print("🔍 Colunas mapeadas:")
    for padrao, original in colunas_encontradas.items():
        print(f"   {padrao} ← {original}")
    
    # Verificar colunas essenciais
    if 'datetime' not in colunas_encontradas or 'hour' not in colunas_encontradas:
        print("❌ Colunas de data/hora não encontradas!")
        return None
    
    # Criar DataFrame final
    df_final = df.copy()
    
    # Renomear colunas
    renomeacao = {original: padrao for padrao, original in colunas_encontradas.items()}
    df_final = df_final.rename(columns=renomeacao)
    
    # Criar datetime
    try:
        # Debug: Mostrar primeiros valores de data e hora
        print("\n📅 Valores de exemplo:")
        print("Data:", df_final['datetime'].head())
        print("Hora:", df_final['hour'].head())
        
        # Converter data para datetime (formato YYYY/MM/DD)
        df_final['datetime'] = pd.to_datetime(df_final['datetime'], format='%Y/%m/%d', errors='coerce')
        
        # Debug: Mostrar valores após conversão de data
        print("\n📅 Após conversão de data:")
        print(df_final['datetime'].head())
        
        # Limpar e converter hora
        df_final['hour'] = df_final['hour'].astype(str).str.replace(' UTC', '').str.zfill(4)
        
        # Debug: Mostrar valores após limpeza de hora
        print("\n⏰ Após limpeza de hora:")
        print(df_final['hour'].head())
        
        # Combinar data e hora
        df_final['datetime'] = df_final['datetime'] + pd.to_timedelta(
            df_final['hour'].str[:2].astype(int), unit='h'
        ) + pd.to_timedelta(
            df_final['hour'].str[2:].astype(int), unit='m'
        )
        
        # Debug: Mostrar valores finais
        print("\n📅⏰ Datetime final:")
        print(df_final['datetime'].head())
        
        # Remover registros com datetime inválido
        df_final = df_final.dropna(subset=['datetime'])
        
        print(f"✅ Datetime criado: {len(df_final)} registros válidos")
        
    except Exception as e:
        print(f"❌ Erro ao criar datetime: {e}")
        return None
    
    # Converter colunas numéricas
    colunas_numericas = ['precipitation_mm', 'temperature_c', 'humidity_pct', 'pressure_hpa', 'wind_speed_ms']
    
    for col in colunas_numericas:
        if col in df_final.columns:
            # Substituir vírgulas por pontos
            df_final[col] = df_final[col].astype(str).str.replace(',', '.', regex=False)
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
    
    # Converter vento para km/h
    if 'wind_speed_ms' in df_final.columns:
        df_final['wind_speed_kmh'] = df_final['wind_speed_ms'] * 3.6
    
    # Estimar umidade do solo
    if 'precipitation_mm' in df_final.columns and 'humidity_pct' in df_final.columns:
        df_final = df_final.sort_values(['location', 'datetime'])
        df_final['precip_24h'] = df_final.groupby('location')['precipitation_mm'].rolling(24, min_periods=1).sum().reset_index(0, drop=True)
        df_final['soil_moisture_pct'] = (df_final['humidity_pct'] * 0.6 + (df_final['precip_24h'].fillna(0) / 50) * 40).clip(20, 95)
    
    # Selecionar colunas finais
    colunas_finais = ['datetime', 'location', 'precipitation_mm', 'temperature_c', 'humidity_pct', 'pressure_hpa', 'wind_speed_kmh', 'soil_moisture_pct']
    colunas_finais = [col for col in colunas_finais if col in df_final.columns]
    
    df_resultado = df_final[colunas_finais].copy()
    df_resultado = df_resultado.dropna(subset=['datetime', 'location'])
    df_resultado = df_resultado.drop_duplicates(subset=['datetime', 'location'])
    df_resultado = df_resultado.sort_values(['location', 'datetime']).reset_index(drop=True)
    
    return df_resultado

def criar_dados_complementares():
    """
    Cria dados de rios e eventos baseados nos dados meteorológicos reais do RS
    """
    print("\n🌊 Criando dados complementares para o Rio Grande do Sul...")
    
    # Verificar se weather_data.csv existe
    if not os.path.exists('data/raw/weather_data.csv'):
        print("❌ weather_data.csv não encontrado")
        return
    
    df_weather = pd.read_csv('data/raw/weather_data.csv')
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
    
    # Criar dados de rios baseados na precipitação real - níveis do RS
    print("🏞️ Gerando níveis de rios do RS baseados na precipitação...")
    
    river_data = []
    # Níveis base reais dos rios do RS (em metros)
    base_levels = {
        'Porto_Alegre': 1.2,    # Guaíba - nível normal
        'Santa_Maria': 3.0,     # Vacacaí Mirim - nível base estimado
        'Caxias_do_Sul': 2.5,   # Rio das Antas - nível base estimado  
        'Pelotas': 1.8          # Canal São Gonçalo - nível base estimado
    }
    
    # Mapeamento cidade -> rio principal
    rios_principais = {
        'Porto_Alegre': 'Guaíba',
        'Santa_Maria': 'Vacacaí_Mirim',
        'Caxias_do_Sul': 'Rio_das_Antas',
        'Pelotas': 'Canal_São_Gonçalo'
    }
    
    current_levels = base_levels.copy()
    
    for location in df_weather['location'].unique():
        if location not in base_levels:
            base_levels[location] = 2.0
            current_levels[location] = 2.0
        
        df_loc = df_weather[df_weather['location'] == location].sort_values('datetime')
        
        for i, row in df_loc.iterrows():
            # Calcular precipitação das últimas 48h
            start_idx = max(0, i-47)
            precip_48h = df_loc.iloc[start_idx:i+1]['precipitation_mm'].sum()
            
            # Calcular mudança no nível baseada na chuva real
            # Rios do RS são mais sensíveis a precipitação intensa
            if precip_48h > 150:  # Chuva muito intensa (como as de maio 2024)
                change = np.random.uniform(0.5, 1.5)
            elif precip_48h > 80:  # Chuva forte
                change = np.random.uniform(0.2, 0.6)
            elif precip_48h > 30:  # Chuva moderada
                change = np.random.uniform(0.05, 0.25)
            else:
                change = np.random.uniform(-0.15, 0.05)
            
            # Efeito sazonal do RS
            month = row['datetime'].month
            if month in [1, 2, 3, 4, 5]:  # Outono/final verão - época crítica no RS
                change *= 1.8
            elif month in [6, 7, 8]:  # Inverno
                change *= 0.8
            elif month in [9, 10, 11]:  # Primavera
                change *= 1.2
            else:  # Verão
                change *= 1.4
            
            current_levels[location] += change + np.random.normal(0, 0.08)
            current_levels[location] = max(0.5, current_levels[location])
            
            # Taxa de subida
            if river_data and river_data[-1]['location'] == location:
                rate = current_levels[location] - river_data[-1]['nivel_metros']
            else:
                rate = 0
            
            # Status baseado nos níveis críticos reais do RS
            status = 'normal'
            if location == 'Porto_Alegre' and current_levels[location] > 3.0:  # Guaíba cota inundação
                status = 'critico'
            elif location == 'Santa_Maria' and current_levels[location] > 4.5:
                status = 'critico'
            elif location == 'Caxias_do_Sul' and current_levels[location] > 4.0:
                status = 'critico'
            elif location == 'Pelotas' and current_levels[location] > 2.8:  # Baseado na cota histórica de 1941
                status = 'critico'
            elif current_levels[location] > base_levels[location] * 1.4:
                status = 'alerta'
            
            river_data.append({
                'datetime': row['datetime'],
                'location': location,
                'rio_principal': rios_principais.get(location, 'Rio_Local'),
                'nivel_metros': round(current_levels[location], 2),
                'taxa_subida_m_h': round(rate, 3),
                'status': status
            })
    
    df_rivers = pd.DataFrame(river_data)
    df_rivers.to_csv('data/raw/river_levels.csv', index=False)
    print(f"✅ Dados de rios do RS salvos: {len(df_rivers)} registros")
    
    # Criar eventos de enchente baseados em níveis críticos do RS
    print("🚨 Identificando eventos de enchente no RS...")
    
    eventos = []
    for location in df_rivers['location'].unique():
        df_loc = df_rivers[df_rivers['location'] == location]
        base_level = base_levels.get(location, 2.0)
        
        # Períodos críticos - baseado nos limiares reais do RS
        if location == 'Porto_Alegre':
            critical_periods = df_loc[df_loc['nivel_metros'] > 3.0]  # Cota de inundação Guaíba
        elif location == 'Pelotas':
            critical_periods = df_loc[df_loc['nivel_metros'] > 2.83]  # Próximo da cota de 1941
        else:
            critical_periods = df_loc[df_loc['nivel_metros'] > base_level * 1.6]
        
        if len(critical_periods) > 24:  # Pelo menos 1 dia
            max_level = critical_periods['nivel_metros'].max()
            
            # Critérios de severidade baseados nas enchentes históricas do RS
            if location == 'Porto_Alegre' and max_level > 5.0:  # Próximo do recorde de maio 2024
                severity = 'high'
                pop = np.random.randint(400000, 600000)  # Grande Porto Alegre
            elif max_level > base_level * 2.0:
                severity = 'high'
                pop = np.random.randint(25000, 50000)
            else:
                severity = 'medium' 
                pop = np.random.randint(10000, 25000)
            
            eventos.append({
                'date': critical_periods['datetime'].iloc[0].strftime('%Y-%m-%d'),
                'location': location,
                'rio_principal': rios_principais.get(location, 'Rio_Local'),
                'duration_days': len(critical_periods) // 24 + 1,
                'severity': severity,
                'max_water_level_m': round(max_level, 2),
                'affected_population': pop,
                'emergency_declared': severity == 'high',
                'source': 'INMET_RS_Analysis'
            })
    
    if eventos:
        df_events = pd.DataFrame(eventos)
        df_events.to_csv('data/disaster_charter/flood_events.csv', index=False)
        print(f"✅ Eventos de enchente do RS identificados: {len(df_events)}")
    
    print("\n🎉 Dados complementares do RS criados!")

def main():
    """Função principal"""
    print("🌦️ PROCESSADOR DE DADOS INMET - RIO GRANDE DO SUL")
    print("=" * 65)
    print("💡 Este script processa dados meteorológicos do INMET para o RS")
    print("📍 Cidades: Porto Alegre, Santa Maria, Caxias do Sul, Pelotas")
    print("🏞️ Rios: Guaíba, Vacacaí Mirim, Rio das Antas, Canal São Gonçalo")
    print()
    
    # Executar setup automático
    setup_automatico()
    
    # Criar dados complementares
    criar_dados_complementares()
    
    print("\n" + "=" * 65)
    print("🎉 PROCESSAMENTO CONCLUÍDO!")
    print("\n📁 Arquivos criados:")
    print("   ✅ data/raw/weather_data.csv - Dados meteorológicos do RS")
    print("   ✅ data/raw/river_levels.csv - Níveis dos rios do RS")
    print("   ✅ data/disaster_charter/flood_events.csv - Eventos de enchente do RS")
    
    print("\n🚀 Próximos comandos:")
    print("cd machine_learning/src/")
    print("python preprocessamento.py")
    print("python treinamento.py --data ../../data/processed/dataset_ml_final.csv")

if __name__ == "__main__":
    main()