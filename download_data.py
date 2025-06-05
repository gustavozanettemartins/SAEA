# download_inmet_simples.py
"""
Script simples para baixar dados meteorológicos do INMET
Baixa os arquivos ZIP de 2023, 2024 e 2025 diretamente

URLs:
- https://portal.inmet.gov.br/uploads/dadoshistoricos/2025.zip
- https://portal.inmet.gov.br/uploads/dadoshistoricos/2024.zip  
- https://portal.inmet.gov.br/uploads/dadoshistoricos/2023.zip

Autores: [Gustavo Zanette Martins, Michelle Guedes Cavalari]
Data: 2025-06-02
"""

import requests
import os
from pathlib import Path
import time

def download_file(url, filename):
    """
    Baixa um arquivo da URL e salva localmente
    
    Args:
        url (str): URL do arquivo
        filename (str): Nome do arquivo local
    
    Returns:
        bool: True se sucesso, False se erro
    """
    try:
        print(f"📥 Baixando {filename}...")
        print(f"🔗 URL: {url}")
        
        # Headers para simular navegador
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Fazer a requisição
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()  # Lança exceção se erro HTTP
        
        # Obter tamanho do arquivo se disponível
        total_size = int(response.headers.get('content-length', 0))
        
        # Baixar o arquivo
        with open(filename, 'wb') as file:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    
                    # Mostrar progresso se soubermos o tamanho
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r   📊 Progresso: {percent:.1f}% ({downloaded:,}/{total_size:,} bytes)", end='')
        
        print(f"\n✅ {filename} baixado com sucesso!")
        print(f"📁 Tamanho: {os.path.getsize(filename):,} bytes")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Erro ao baixar {filename}: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Erro inesperado ao baixar {filename}: {e}")
        return False

def main():
    """
    Função principal - baixa todos os arquivos
    """
    print("🌦️ INMET DOWNLOADER - Dados Meteorológicos")
    print("=" * 50)
    print("📅 Baixando dados de 2023, 2024 e 2025")
    print("📂 Local: pasta atual do script")
    print()
    
    # URLs dos arquivos do INMET
    downloads = [
        {
            'url': 'https://portal.inmet.gov.br/uploads/dadoshistoricos/2025.zip',
            'filename': '2025.zip'
        },
        {
            'url': 'https://portal.inmet.gov.br/uploads/dadoshistoricos/2024.zip', 
            'filename': '2024.zip'
        },
        {
            'url': 'https://portal.inmet.gov.br/uploads/dadoshistoricos/2023.zip',
            'filename': '2023.zip'
        }
    ]
    
    # Estatísticas
    total_files = len(downloads)
    successful_downloads = 0
    failed_downloads = []
    
    # Baixar cada arquivo
    for i, download in enumerate(downloads, 1):
        print(f"\n📦 Arquivo {i}/{total_files}")
        print("-" * 30)
        
        url = download['url']
        filename = download['filename']
        
        # Verificar se arquivo já existe
        if os.path.exists(filename):
            print(f"⚠️ {filename} já existe!")
            resposta = input("   Deseja baixar novamente? (s/N): ").lower()
            if resposta != 's':
                print(f"⏭️ Pulando {filename}")
                successful_downloads += 1
                continue
        
        # Tentar baixar
        success = download_file(url, filename)
        
        if success:
            successful_downloads += 1
        else:
            failed_downloads.append(filename)
        
        # Pausa entre downloads para não sobrecarregar o servidor
        if i < total_files:
            print("⏳ Aguardando 2 segundos...")
            time.sleep(2)
    
    # Relatório final
    print("\n" + "=" * 50)
    print("📊 RELATÓRIO FINAL")
    print("=" * 50)
    print(f"✅ Downloads bem-sucedidos: {successful_downloads}/{total_files}")
    
    if failed_downloads:
        print(f"❌ Downloads falharam: {len(failed_downloads)}")
        for filename in failed_downloads:
            print(f"   • {filename}")
    else:
        print("🎉 Todos os downloads foram concluídos com sucesso!")
    
    # Listar arquivos baixados
    print("\n📁 Arquivos na pasta atual:")
    current_dir = Path('.')
    zip_files = list(current_dir.glob('*.zip'))
    
    if zip_files:
        total_size = 0
        for zip_file in sorted(zip_files):
            size = zip_file.stat().st_size
            total_size += size
            print(f"   📄 {zip_file.name} - {size:,} bytes ({size/1024/1024:.1f} MB)")
        
        print(f"\n💾 Tamanho total: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
    else:
        print("   (Nenhum arquivo ZIP encontrado)")
    
    print("\n💡 PRÓXIMOS PASSOS:")
    print("1. Execute o script processar_csv_inmet_rs.py")
    print("2. Os arquivos ZIP serão extraídos automaticamente")
    print("3. Dados meteorológicos do RS serão processados")
    
    print("\n🎯 Download concluído!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Download interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        print("💡 Tente executar o script novamente")