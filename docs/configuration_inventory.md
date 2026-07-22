# Inventário público de configuração

Este inventário corresponde ao `config.yaml` de 2026-07-22. Uma linha com `.*` cobre todas as chaves-folha abaixo daquele mapa; por exemplo, `physics.channels.*` cobre cada canal listado no YAML e `inversion.lidar_ratios_sr.*.*` cobre todos os comprimentos de onda e meses.

Classificações:

- **consumida**: altera hoje um caminho operacional existente;
- **metadado**: descreve a instalação, sem controlar processamento;
- **dormente**: declarada e validada, mas não conectada ao pipeline;
- **experimental**: consumida por um caminho explicitamente experimental.

| Chave ou padrão | Origem/default | Consumidor atual | Validação | Classificação | Destino |
|---|---|---|---|---|---|
| `project.{name,full_name,station_name,institution,timezone}` | valores do YAML; sem default operacional | nenhum consumidor de pipeline | strings não vazias | metadado | validar e manter |
| `processing.incremental` | YAML `false`; código usa `false` | LIBIDS, Lipancora, LIRACOS e Lebear | booleano | consumida | validar e manter |
| `processing.interactive_qa` | YAML `true`; sem default consumido | nenhum | booleano | dormente | decidir em ENG-017 |
| `processing.{console_level,file_level}` | YAML `INFO`; código usa `INFO` | `io/logging_utils.py` | nome de nível reconhecido | consumida | validar e manter |
| `processing.laser_shot_tolerance_fraction` | YAML `0.002`; código usa `0.002` | `level0/libids.py`/`quality.py` | finito, não booleano, >= 0 | consumida | validar e manter |
| `processing.dark_current_max_association_hours` | YAML `12.0`; ausência não limita | `level0/inventory.py` | finito, não booleano, >= 0 | consumida | validar e manter |
| `processing.spurious_extensions` | YAML; código usa `.dat/.dpp/.zip` | descoberta somente leitura em `io/filesystem.py` | lista de strings | consumida | validar e manter |
| `processing.quarantine_dir` | YAML; código usa `<raw>/_quarantine` | exclusão da árvore de scan; APIs explícitas recebem destino próprio | string não vazia | consumida | validar e manter |
| `processing.raw_scan_ignore_dirs` | YAML; default vazio | `io/filesystem.py` | lista de strings | consumida | validar e manter |
| `processing.{max_workers_io,max_workers_cpu}` | YAML `1`; sem default consumido | nenhum executor | inteiro >= 1, não booleano | dormente | decidir em ENG-017 |
| `directories.{raw_data,processed_data,log_dir}` | obrigatórias | paths, pipelines e logging | strings não vazias | consumida | validar e manter |
| `directories.site_output` | YAML `measurements`; sem default consumido | nenhum | string não vazia | dormente | decidir em ENG-017 |
| `site.{latitude,longitude,station_altitude_m}` | YAML; fallbacks locais em módulos | Level 0/metadados atmosféricos e Level 2 | finitos, não booleanos | consumida | validar e manter |
| `site.timezone` | YAML `America/Sao_Paulo`; mesmo fallback no inventário | `level0/inventory.py` | string não vazia | consumida | validar e manter |
| `physics.vertical_resolution_m` | obrigatória; código usa `7.5` como fallback interno | escrita Nível 0 | finito e > 0 | consumida | validar e manter |
| `physics.speed_of_light_m_s` | YAML; código usa constante SI | cálculo de bin time no Lipancora | finito e > 0 | consumida | validar e manter |
| `physics.{default_surface_temp_c,default_surface_pressure_hpa}` | YAML; código usa 25 C/940 hPa | fallback meteorológico do LIBIDS | temperatura finita; pressão finita e > 0 | consumida | validar e manter |
| `physics.{background_start_m,background_stop_m}` | YAML; defaults 29.000/29.999 m | NetCDF Nível 0 e correções | finitos, > 0 e stop > start | consumida | validar e manter |
| `physics.{pbl_min_search_m,pbl_max_search_m,pbl_smooth_bins}` | YAML; defaults 500/4.000/15 | PBL no Lipancora | altitudes > 0, max > min, bins inteiro >= 1 | consumida | validar e manter |
| `physics.channels.*.{deadtime_us,bin_shift_bins,background_offset}` | mapas nomeados do YAML; fallback neutro por canal ausente | correções instrumentais do Lipancora | três campos exatos; números finitos e shift inteiro | consumida | formato legado posicional aceito com aviso somente até 0.2.0 |
| `hardware.name_to_id.*` | mapa do YAML; writer possui fallback 9999 | IDs de canal do Nível 0 | folhas inteiras positivas, não booleanas | consumida | validar e manter |
| `radiosonde.{station_id,cache_dir}` | YAML; defaults 83779/cache padrão | integração termodinâmica e cache | strings não vazias | consumida | validar e manter |
| `radiosonde.{station_name,fallback_to_standard_atmosphere}` | YAML; alias legado normalizado | nenhum comportamento atual | string/booleano | dormente | decidir em ENG-017 |
| `surface_weather.cache_dir` | YAML; cache padrão interno | Open-Meteo e exclusão do scan | string não vazia | consumida | validar e manter |
| `surface_weather.{provider,fallback_to_config_defaults}` | YAML; implementação sempre usa Open-Meteo e fallback do LIBIDS | nenhum seletor atual | string/booleano | dormente | decidir em ENG-017 |
| `visualization.{output_format,dpi,altitude_ranges_km,channels_to_plot}` | YAML; defaults webp/120/[5,15,30]/vazio conforme módulo | LIRACOS, quicklooks e estilo | string, inteiro >= 1, lista positiva finita, lista de strings | consumida | validar e manter |
| `visualization.quicklook.{max_time_gap_minutes,missing_data_color,colormap}` | YAML; defaults do módulo | quicklooks | inteiro >= 1 e strings | consumida | validar e manter |
| `visualization.quicklook.{show_pbl,show_tropopause,mean_profile_smooth_bins}` | YAML; sem branch correspondente | nenhum controle atual | booleanos/inteiro >= 1 | dormente | decidir em ENG-017 |
| `visualization.level2_qa.enabled` | YAML/default `true` | orquestração opcional em `level2/qa.py` | booleano | consumida | validar e manter |
| `visualization.level2_qa.max_altitude_km` | YAML `30`; funções têm argumento default, mas orquestração não o repassa | nenhum controle atual | finito e > 0 | dormente | decidir em ENG-017 |
| `visualization.level2_qa.smooth_bins` | YAML/default 15 | plots de QA | inteiro >= 1 | consumida | validar e manter |
| `visualization.level2_qa.generate_*` | YAML/default `true` | seleção de plots em `viz/level2_qa.py` | booleanos | consumida | validar e manter |
| `inversion.enabled` | YAML `true`; pipeline não consulta | nenhum | booleano | dormente | decidir em ENG-017 |
| `inversion.interactive_qa` | YAML `true`; sem interação no pipeline | nenhum | booleano | dormente | decidir em ENG-017 |
| `inversion.wavelengths_to_process` | YAML; default `[532]` | Lebear | lista não vazia de números positivos e finitos | consumida | validar e manter |
| `inversion.block_average_minutes` | YAML; fallback legado/default 15 | janelas do retrieval | inteiro >= 1 | consumida | validar e manter |
| `inversion.kfs_mode` | YAML/default `two_sided` | KFS/retrieval e metadados | somente `two_sided` no pipeline; ramos isolados existem na API científica | consumida | versão Fernald 2; forward validado matematicamente e ainda sensível a ruído |
| `inversion.products.*` | todos `true` no YAML | writer salva atualmente o schema completo sem consultar esses flags | booleanos | dormente | decidir em ENG-017 |
| `inversion.{monte_carlo_iterations,random_seed,beta_ref_relative_std,aerosol_ref_fraction,min_lidar_ratio_sr,allow_negative_aerosol}` | YAML; defaults no retrieval | retrieval/KFS | inteiros válidos, números finitos e booleano conforme campo | consumida | validar e manter |
| `inversion.molecular_fit.{ref_alt_min_m,ref_alt_max_m,ref_window_bins,max_relative_slope,max_relative_variance,min_valid_fraction}` | YAML; defaults em `level2/config.py` | seleção/QA da referência molecular | finitos, bins inteiro, max altitude > min | consumida | validar e manter |
| `inversion.molecular_fit.lidar_ratio_molecular_sr` | YAML; alias normalizado | nenhum cálculo atual | finito e > 0 | dormente | decidir em ENG-017 |
| `inversion.gluing.*` | YAML; defaults em `level2/config.py` | gluing e critérios de qualidade | finitos/inteiros/booleano; search max > min | consumida | validar e manter |
| `inversion.cloud_screening.*` | YAML; helpers implementados mas não chamados pelo retrieval | nenhum caminho integrado | estrutura e tipos validados | dormente | decisão científica em ENG-017 |
| `inversion.lidar_ratio_std_sr.*` | mapa por comprimento de onda; default 10 sr | retrieval | folhas finitas e > 0 | consumida | validar e manter |
| `inversion.lidar_ratios_sr.*.*` | mapa por comprimento de onda/mês; default 60 sr | retrieval | folhas finitas e > 0 | consumida | validar e manter |

## Compatibilidade, remoções e desconhecidas

- `normalize_config` ainda injeta apenas em memória os aliases legados `physics.speed_of_light`, `physics.bg_start`, `physics.bg_stop`, `physics.bg_start_m`, `physics.bg_stop_m`, `radiosonde.fallback_to_standard`, `inversion.lidar_ratios` e `inversion.molecular_fit.lidar_ratio_molecular`. Eles não duplicam chaves no YAML.
- `processing.quarantine_spurious_files` e `processing.delete_spurious_files` foram removidas do YAML: scan é sempre somente leitura desde ENG-012. Se reaparecerem em configuração carregada, a validação falha com instrução para usar ações explícitas de filesystem.
- Chaves desconhecidas em seções de estrutura fixa são rejeitadas com o caminho completo. Mapas abertos (`physics.channels`, IDs de hardware e tabelas de lidar ratio) aceitam nomes dinâmicos, mas validam suas folhas.
- Chaves dormentes são preservadas e validadas, porém não foram conectadas a nenhum pipeline nesta tarefa. As decisões de manter, ativar, depreciar ou remover pertencem a ENG-017; a migração das listas posicionais de canal pertence a ENG-018.
