# Benchmarks reproduzíveis do Nível 2

O protocolo de ENG-030 mede o pipeline LEBEAR com dados de Nível 1 gerados de
forma determinística. Ele não usa medições privadas, não altera configuração de
produção e força a execução efetiva de gluing, Rayleigh/KFS com Monte Carlo,
montagem do dataset e escrita NetCDF.

## Cenários

| Cenário | Perfis | Bins | Comprimentos de onda | Iterações MC | Uso |
|---|---:|---:|---|---:|---|
| `ci` | 3 | 240 | 355 e 532 nm | 5 | smoke benchmark multiespectral pequeno; modos `complete` e `partial` |
| `typical` | 24 | 800 | 355 e 532 nm | 30 | medida reduzida de duas horas |
| `large` | 288 | 4.000 | 355 e 532 nm | 300 | protocolo local de um dia; não executar no CI |

Os limiares de gluing e Rayleigh usados pela fixture são exclusivos do
benchmark e garantem que os blocos sintéticos percorram a inversão. Eles não são
valores científicos propostos para dados reais.

## Execução

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  benchmarks/benchmark_level2.py \
  --scenario typical \
  --product-mode complete \
  --repetitions 3 \
  --warmup 1 \
  --threads 1 \
  --output /tmp/milgrau-level2-typical.json
```

Cada repetição usa um processo isolado e executa um pipeline completo de
warm-up antes da medição. O coordenador fixa `PYTHONHASHSEED=0` e um thread para
Numba, OpenMP, OpenBLAS e MKL. O input sintético é escrito antes de iniciar os
workers. O cenário grande usa o mesmo comando com `--scenario large`, mas deve
ser executado localmente em máquina com recursos adequados.

## Método de medição

- tempos total e por estágio: `time.perf_counter()`;
- pico de memória: `resource.getrusage(RUSAGE_SELF).ru_maxrss` dentro de cada
  worker Linux; o relatório preserva o RSS absoluto e seu incremento sobre o
  RSS imediatamente anterior ao warm-up;
- saída: tamanho do NetCDF comprimido no filesystem;
- materializações: contagem de arrays NumPy únicos observados nos limites
  explícitos dos estágios e soma de `nbytes` desses arrays.

A contagem de materializações é um indicador comparativo dos objetos que
atravessam limites, não um contador de eventos de alocação ou cópia. O pico de
RSS inclui imports e warm-up do processo, portanto comparações devem repetir o
mesmo protocolo e ambiente.

## Baseline local de 2026-07-22

Ambiente: commit base `c3efcd66dcc45f4597b4d5e6d887bb3b010df91f`
com alterações locais; Linux 7.1.4 x86_64; Python 3.14.6; Intel Core i7-7500U,
4 CPUs lógicas e 7,63 GiB de RAM. Pacotes: MILGRAU 0.1.0, NumPy 2.4.6,
Pandas 3.0.3, xarray 2026.4.0, netCDF4 1.7.4, SciPy 1.17.1 e Numba 0.65.1.
Foram usados três workers, um warm-up completo por worker e um thread por
runtime.

| Métrica | `ci` | CV | `typical` | CV |
|---|---:|---:|---:|---:|
| tempo total mediano | 0,158 s | 4,04% | 4,339 s | 0,90% |
| pico RSS mediano | 306,656 MiB | 0,06% | 316,547 MiB | 0,13% |
| incremento de pico mediano | 96,199 MiB | 0,19% | 106,059 MiB | 0,39% |
| NetCDF escrito | 0,319 MiB | 0,00% | 1,165 MiB | 0,00% |
| arrays observados | 159 | 0,00% | 231 | 0,00% |
| bytes observados nos limites | 0,181 MiB | 0,00% | 6,435 MiB | 0,00% |
| blocos válidos de retrieval | 1 | 0,00% | 16 | 0,00% |

| Estágio | `ci` mediana | `ci` CV | `typical` mediana | `typical` CV |
|---|---:|---:|---:|---:|
| abertura, load e validação | 0,00894 s | 0,07% | 0,01014 s | 3,41% |
| seleção e blocos | 0,00237 s | 0,40% | 0,00775 s | 0,59% |
| gluing | 0,06287 s | 0,92% | 3,54508 s | 0,97% |
| modelo molecular | 0,00053 s | 4,25% | 0,00119 s | 0,90% |
| Rayleigh/KFS | 0,01202 s | 0,69% | 0,64682 s | 1,53% |
| montagem do resultado | 0,00028 s | 5,83% | 0,00106 s | 1,58% |
| montagem do dataset | 0,00268 s | 1,00% | 0,00354 s | 0,87% |
| validação da saída | 0,00004 s | 1,14% | 0,00004 s | 1,91% |
| escrita NetCDF | 0,06559 s | 8,37% | 0,10046 s | 0,17% |

O cenário `large` fica deliberadamente sem baseline neste host: o protocolo
existe para execução local explícita e não deve consumir tempo ou memória do CI.

## SCI-003: produto completo e parcial em 2026-07-22

SCI-003 tornou o cenário `ci` explicitamente multiespectral. O modo
`--product-mode complete` disponibiliza e processa 355/532 nm; o modo
`--product-mode partial` solicita ambos, disponibiliza somente 532 nm e mede a
falha localizada de 355 nm mais a escrita sem slice científico NaN. Por isso o
tempo completo não deve ser comparado diretamente com o baseline histórico
`ci` de uma única wavelength. Foram usados processos isolados, um warm-up por
processo e um thread por runtime.

| Modo | Repetições | Tempos totais (s) | Mediana | CV | Pico RSS mediano | NetCDF | Dataset em memória |
|---|---:|---|---:|---:|---:|---:|---:|
| completo 355/532 | 5 | 0,2882; 0,2985; 0,2997; 0,2894; 0,3059 | 0,2985 s | 2,24% | 308,8 MiB | 432.820 bytes | 163.270 bytes |
| parcial, somente 532 | 3 | 0,2128; 0,2092; 0,2111 | 0,2111 s | 0,71% | 309,2 MiB | 400.563 bytes | 82.854 bytes |

O produto completo teve dois blocos válidos (um por wavelength) e permaneceu
sob o orçamento de alerta de mediana, memória e saída. A mediana de gluing foi
0,1410 s para dois wavelengths, coerente com aproximadamente duas vezes os
0,0701 s do checkpoint SCI-002 de uma wavelength; não há evidência de regressão
relevante no caminho glued por wavelength.

Omitir a wavelength falha reduziu o arquivo em 32.257 bytes (7,45%) e evitou
80.416 bytes de arrays científicos materializados no dataset (49,25%). Um slice
hipotético inteiramente NaN comprimiria de modo diferente, portanto seu tamanho
em disco não foi inventado sem materializar o schema rejeitado; os números
reportados comparam apenas os dois produtos reais aprovados.

## Orçamento inicial aprovado

Este orçamento foi aceito pela mantenedora em 2026-07-22 como limite de alerta
para comparações futuras, não como reserva ou gasto de memória.

| Cenário | Mediana total máxima | Pico RSS máximo | Incremento de RSS máximo | Saída máxima |
|---|---:|---:|---:|---:|
| `ci` | 0,30 s | 384 MiB | 160 MiB | 0,50 MiB |
| `typical` | 6,50 s | 400 MiB | 160 MiB | 1,50 MiB |

Uma otimização futura deve comparar ao menos três repetições no mesmo cenário,
ambiente, número de threads e política de warm-up. Além de respeitar o orçamento,
ela não deve regredir a mediana total em mais de 10% sem um trade-off aprovado
antes da implementação. O orçamento do cenário `large` só deve ser definido
depois do primeiro baseline em uma máquina local identificada.
