import os
import re

html_calendario = 'bootstrap year calendar.html' 
pasta_base = 'ql-measurements'      
cor_padrao = '#A3E4D7'              

with open(html_calendario, 'r', encoding='utf-8') as f:
    conteudo = f.read()

# Verifica o que já existe no calendário para não duplicar
urls_existentes = set(re.findall(r"url:\s*'(.*?)'", conteudo))

novas_entradas = ""
contador = 0

# Vasculha as pastas dos anos
if os.path.exists(pasta_base):
    for ano_dir in os.listdir(pasta_base):
        caminho_ano = os.path.join(pasta_base, ano_dir)
        
        if os.path.isdir(caminho_ano):
            for arquivo in os.listdir(caminho_ano):
                # Procura os HTMLs que você já tem gerados aí
                if arquivo.endswith('_Gallery.html') or arquivo.endswith('QL_SPULidarStation.html'):
                    url_relativa = f"{pasta_base}/{ano_dir}/{arquivo}"
                    
                    if url_relativa not in urls_existentes:
                        # Extrai a data do nome do arquivo (ex: 20240913 ou 2024_09_13)
                        match = re.search(r'(\d{4})_?(\d{2})_?(\d{2})', arquivo)
                        if match:
                            ano = int(match.group(1))
                            mes_js = int(match.group(2)) - 1 # JS conta o mês de 0 a 11
                            dia = int(match.group(3))
                            
                            novas_entradas += f",\n  {{\n    startDate: new Date({ano}, {mes_js}, {dia}),\n    endDate: new Date({ano}, {mes_js}, {dia}),\n    color: '{cor_padrao}',\n    url: '{url_relativa}'\n  }}"
                            contador += 1

if contador == 0:
    print("Nenhuma medida nova para adicionar.")
else:
    print(f"Adicionando {contador} novas medidas no calendário...")
    if "// MARCADOR_AUTOMATICO" in conteudo:
        novo_conteudo = conteudo.replace("// MARCADOR_AUTOMATICO", novas_entradas + "\n  // MARCADOR_AUTOMATICO")
        with open(html_calendario, 'w', encoding='utf-8') as f:
            f.write(novo_conteudo)
        print("Calendário atualizado!")
    else:
        print("ERRO: Faltou colocar o // MARCADOR_AUTOMATICO no seu HTML.")