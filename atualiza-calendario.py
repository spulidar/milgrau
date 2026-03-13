import os
import re

# Coloquei o nome exato do arquivo que vi no seu print
html_calendario = '/home/lidarnet-compaq/Documents/01-milgrau/milgrau_xD/atualiza-site/bootstrap year calendar.html' 
cor_padrao = '#A3E4D7'              

with open(html_calendario, 'r', encoding='utf-8') as f:
    conteudo = f.read()

urls_existentes = set(re.findall(r"url:\s*'(.*?)'", conteudo))

novas_entradas = ""
contador = 0

# Vasculha a pasta atual procurando por pastas de anos (4 números)
for pasta in os.listdir('.'):
    if os.path.isdir(pasta) and pasta.isdigit() and len(pasta) == 4:
        for arquivo in os.listdir(pasta):
            if arquivo.endswith('_Gallery.html') or arquivo.endswith('QL_SPULidarStation.html'):
                # Agora a URL fica limpa: "2024/nome_do_arquivo.html"
                url_relativa = f"{pasta}/{arquivo}"
                
                if url_relativa not in urls_existentes:
                    match = re.search(r'(\d{4})_?(\d{2})_?(\d{2})', arquivo)
                    if match:
                        ano = int(match.group(1))
                        mes_js = int(match.group(2)) - 1
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
        print("Calendário atualizado com sucesso!")
    else:
        print("ERRO: Faltou colocar o // MARCADOR_AUTOMATICO no seu HTML.")