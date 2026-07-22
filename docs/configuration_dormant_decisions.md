# Decisões sobre controles dormentes

Decisão de ENG-017 em 2026-07-22. Nenhuma chave desta lista passa a controlar um pipeline por efeito desta decisão. “Aprovação” indica se uma futura ativação exige revisão científica; permanecer inerte não altera produto e, portanto, não exige essa aprovação.

| Chaves | Decisão | Responsável pela próxima decisão | Justificativa | Aprovação / tarefa |
|---|---|---|---|---|
| `processing.interactive_qa` | manter inerte e marcar como reservado | manutenção da interface | não existe fluxo interativo nas CLIs atuais | sem impacto atual; tratar em ENG-064 junto da UX |
| `processing.max_workers_io`, `processing.max_workers_cpu` | manter inertes; não paralelizar automaticamente | engenharia de performance | ativação exige benchmark, limites e política determinística | sem aprovação científica enquanto resultados forem idênticos; implementação somente em ENG-034 |
| `directories.site_output` | manter como metadado reservado, sem promessa de escrita | manutenção de paths | remover agora pode quebrar configurações externas; nenhum writer usa o caminho | ENG-064 deve manter, depreciar com aviso ou conectar após caso real |
| `radiosonde.station_name` | manter como metadado | manutenção de metadados | o download usa `station_id`; o nome pode ser útil em proveniência futura | ENG-064, sem mudança científica automática |
| `radiosonde.fallback_to_standard_atmosphere` | manter inerte | responsável científica do Nível 1 | ativar mudaria termodinâmica e potencialmente diagnósticos | aprovação científica obrigatória antes de implementação em ENG-065 |
| `surface_weather.provider` | manter declarativo e limitado documentalmente a Open-Meteo | manutenção de I/O | não há abstração de provedores; o código usa Open-Meteo | ENG-064 deve validar/rotear somente quando houver segundo provedor |
| `surface_weather.fallback_to_config_defaults` | manter inerte | responsável científica do Nível 0 | o fallback atual é incondicional e afeta metadados atmosféricos | aprovação científica para mudar comportamento; ENG-065 |
| `visualization.quicklook.show_pbl`, `show_tropopause`, `mean_profile_smooth_bins` | manter inertes e marcadas | manutenção de visualização | ativação muda apenas apresentação, mas precisa contrato visual/testes | ENG-041 |
| `visualization.level2_qa.max_altitude_km` | manter inerte | manutenção de visualização | plotters não recebem hoje o valor da orquestração | ENG-041; sem impacto no NetCDF científico |
| `inversion.enabled` | manter inerte; não fazer o Lebear deixar de rodar silenciosamente | manutenção das CLIs | semântica entre “CLI chamada” e “pipeline desabilitado” precisa contrato/exit code | ENG-064 |
| `inversion.interactive_qa` | manter inerte | manutenção da interface e responsável científica | não existe gate interativo definido para aceitar/rejeitar retrieval | ENG-065 se a decisão afetar validade/publicação do produto |
| `inversion.products.*` | manter inertes | responsável científica do Nível 2 | omitir variáveis mudaria o schema congelado em ENG-061 | aprovação científica obrigatória; ENG-065 após ENG-020/021 |
| `inversion.molecular_fit.lidar_ratio_molecular_sr` | manter inerte | responsável científica do Nível 2 | conectar a constante pode mudar calibração/retrieval | aprovação científica obrigatória; ENG-065 |
| `inversion.cloud_screening.*` | manter como configuração experimental de biblioteca, não integrada | responsável científica do Nível 2 | máscaras e interrupção da KFS alterariam diretamente resultados e flags | aprovação científica obrigatória; ENG-065 após contrato tipado de ENG-020/021 |

## Política resultante

- Nenhuma chave dormente é ativada por fallback implícito.
- O YAML e o inventário usam a marca `DORMANT` para não prometer efeito inexistente.
- Não há nova depreciação nesta decisão. As duas flags destrutivas removidas em ENG-012/014 já falham com mensagem de migração verificável.
- Ativações operacionais ficam em ENG-034/041/064. Ativações capazes de mudar NetCDF, máscaras, termodinâmica ou retrieval ficam em ENG-065 e exigem aprovação científica explícita.
