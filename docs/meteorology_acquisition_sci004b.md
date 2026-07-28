# SCI-004B — aquisição meteorológica cache-first

Este documento descreve a aquisição isolada implementada pelo SCI-004B. Ela não
seleciona a melhor fonte para a região Rayleigh, não altera KFS/gluing e não
conecta os snapshots ao LEBEAR. Essa integração e a política científica temporal
continuam no SCI-004C.

## Diagnóstico do caminho anterior

| Componente | Estado anterior | Cache | Integridade | Rede | Problema |
| --- | --- | --- | --- | --- | --- |
| Siphon/Wyoming | `WyomingUpperAir.request_data`, station `83779` | CSV por 00/12 UTC | existência do CSV | automática no Nível 1 | seleção fixa 00/12 por faixa horária |
| Payload Siphon | DataFrame com P, z, T, Td, vento e metadados repetidos | CSV já deduplicado | sem hash | resposta HTTP não persistida | não reproduz a resposta completa |
| Sidecar legado | station, horário nominal, download, fonte e CSV | JSON ao lado do CSV | sem schema/hash | n/a | truncamento/corrupção não detectados |
| Nível 1 | incorpora apenas altura, P e T, além de tropopausas | embutido no NetCDF | contrato estrutural do Nível 1 | chama Siphon quando processa | umidade, vento, horário real e hash descartados |
| Falhas legadas | `tenacity` tenta três vezes e retorna `None` | cache por existência | sem causa estruturada | retry exponencial | indisponibilidade vira warning genérico |
| Diretórios | `01-data/wyoming_cache` configurável | ano/mês | nenhuma | n/a | incompatível com bruto/normalizado/manifests imutáveis |

O Siphon 0.10 instalado expõe publicamente somente o DataFrame; seu método
interno `_get_data_raw` devolve o texto integral usado pelo parser. O adaptador
novo conserva esse conteúdo como `raw_payload_kind=http_response`. Transportes
que não oferecem o texto integral devem usar uma serialização canônica de todas
as colunas, unidades, dtypes, valores e metadados, marcada como
`canonical_dataframe_snapshot`; ela não é descrita como HTML bruto.

## Pedido e planejamento

`MeteorologyRequest` é imutável e não aceita credenciais. Ele contém site,
coordenadas, altitude, timestamps de medição, provider, modo, station ID,
horários nominais de radiossonda explícitos, contrato ERA5 L137, grade, cache,
ERA5T, timeout, retries e versão.

Timestamps devem ser timezone-aware e são normalizados para UTC. Um horário
exato pede uma análise; um horário intermediário pede somente as análises
imediatamente anterior e posterior. O resultado é deduplicado, ordenado e
separado por mês. Mudanças de dia, mês e ano não produzem extrapolação.

Para o SPU (`-23.5615`, `-46.7383`) a grade de 0,25 grau é:

```text
(-23.75, -46.75)  (-23.75, -46.50)
(-23.50, -46.75)  (-23.50, -46.50)
```

O pedido MARS usa somente `levtype=ml`: T/q (`130/133`) nos níveis
`1/to/137`, e os campos 2D geopotencial/`lnsp` (`129/152`) arquivados no
nível 1. Ambos usam grade `0.25/0.25`, a área dos quatro pontos e os mesmos
horários planejados; as mensagens são concatenadas num único GRIB mensal.

MARS trata `date` e `time` como produto cartesiano. Quando dias do mesmo mês
precisam de conjuntos de horas diferentes, uma única seleção MARS pediria horas
sem uso. O planejador divide esses casos no menor número de retângulos temporais
exatos e em dois grupos de níveis/variáveis exigidos pelo arquivo MARS,
concatena as mensagens num único GRIB mensal e não baixa o produto cartesiano
excedente. Dias com o mesmo conjunto de horas permanecem agrupados.

## Cache, manifesto e retenção

Layout:

```text
meteorology_cache/
  radiosonde/wyoming/83779/YYYY/MM/{raw,normalized,manifests}/
  era5/model_levels/spu/YYYY/MM/{final,era5t_provisional}/
    {raw_grib,normalized,manifests}/
```

Identidades usam JSON canônico do pedido de dados, provider, dataset, release,
horários, área, variáveis, níveis e versão do manifesto. A identidade não muda
entre `auto`, `cache_only` e `prefetch`, pois o modo controla I/O e não o
conteúdo científico.

Cada bruto e normalizado tem manifesto `milgrau-meteorology-cache-v1` com:

- provider, dataset, release e flag provisório;
- pedido canônico, horários, área, variáveis e níveis;
- instante de aquisição, tamanho e SHA-256;
- normalizador/versão e hashes de origem;
- tipo de payload, versões de dependências e status de validação.

Arquivo sem manifesto, vazio, truncado, com tamanho/hash divergente, identidade,
release ou pedido incompatível é inválido. Artefato e manifesto são publicados
com temporário, `fsync` e `os.replace`; uma interrupção não apaga o cache válido
anterior. Não há limpeza automática e nenhuma API de limpeza foi adicionada.

## Radiossonda

A aquisição recebe somente horários 00/12 UTC explicitamente escolhidos; não
decide limiares científicos de 3 h/6 h. O station ID padrão é `83779`. A resposta
integral ou o snapshot canônico é armazenado antes da normalização. O kernel
SCI-004A converte o DataFrame completo em `AtmosphericProfile`, e o snapshot
normalizado é persistido em NetCDF.

## ERA5 e ERA5T

O bruto é GRIB. O reader opcional ecCodes valida:

- mensagens T/q em todos os 137 níveis;
- `lnsp` e geopotencial de superfície;
- quatro pontos e horários exatos;
- coeficientes híbridos `pv`;
- `expver` para distinguir `final` e `era5t_provisional`.

Depois ele chama o reconstrutor SCI-004A; a física não é duplicada. O NetCDF
normalizado conserva todos os perfis horários. Final e provisório têm
identidades/diretórios diferentes. `--refresh-provisional` é a atualização
explícita que consulta novamente sem sobrescrever o provisório; a conexão dessa
mudança ao reprocessamento L2 pertence ao SCI-004C/SCI-010.

Limitação operacional: a documentação pública do ECMWF informa que ERA5T em
model levels não está disponível pelo catálogo público
`reanalysis-era5-complete` via CDS/MARS em todos os ambientes. O contrato aceita,
detecta e preserva `expver=0005` quando o backend autorizado o fornece; caso
contrário, a falha fica explícita e a radiossonda continua utilizável.

## Modos, credenciais e falhas

- `auto`: valida normalizado, reconstrói de bruto válido ou baixa somente o
  ausente;
- `cache_only`: nunca chama transportes e falha explicitamente em qualquer
  miss/corrupção;
- `prefetch`: prepara cache e inventário sem executar LEBEAR.

`cdsapi` usa `~/.cdsapirc`; `CDSAPI_URL` junto com
`CDSAPI_KEY`/`CDSAPI_TOKEN` é apenas adaptação operacional. Tokens nunca entram
no pedido, logs, NetCDF ou manifestos. Erros passam por redação.

Providers falham independentemente. Se um funcionar, seu snapshot é preservado
com warning sobre o outro. Se ambos falharem, a API retorna USSA-1976 marcada
como `fallback_diagnostic`, fonte `standard_atmosphere` e
`quantitative_retrieval_allowed=false`; isso não é sucesso observacional.

## CLI e testes

```bash
milgrau-meteorology prefetch \
  --start 2026-07-05T12:00:00Z \
  --end 2026-07-05T14:00:00Z \
  --site spu

milgrau-meteorology prefetch \
  --start 2026-07-05T12:00:00Z \
  --end 2026-07-05T14:00:00Z \
  --site spu --dry-run

milgrau-meteorology prefetch \
  --start 2026-07-05T12:00:00Z \
  --end 2026-07-05T14:00:00Z \
  --site spu --cache-only
```

Testes padrão usam transports/decoders falsos e fixtures sintéticas pequenas;
não acessam rede. Testes reais devem ser explicitamente marcados `network` e
executados manualmente com credenciais/licenças aceitas.

O benchmark local mede planejamento, hit de cache, normalização de bruto mock,
tamanhos do GRIB mock/NetCDF e overhead de manifestos. Latência pública não é
critério reproduzível.

Medição local de 2026-07-23, cinco repetições:

| Medida | Resultado |
| --- | ---: |
| planejamento de 24 timestamps em 6 horas ERA5 | `0,000110 s` |
| hit de cache normalizado | `0,02370 s` |
| normalização do bruto mock local | `0,02316 s` |
| GRIB mock | `4.265 bytes` |
| NetCDF normalizado | `47.628 bytes` |
| dois manifests | `9.622 bytes` |
| overhead dos manifests sobre bruto + normalizado | `18,54%` |

`network_calls=0`; a medição não usa nem estima latência pública.

## Pendência SCI-004C

Continuam fora deste escopo: política científica anterior/posterior, seleção
radiossonda/híbrido/ERA5, conexão ao Rayleigh/KFS/LEBEAR, bloqueio produtivo da
atmosfera padrão e invalidação/reprocessamento de produtos L2.
