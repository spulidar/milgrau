# SCI-004A — contrato meteorológico e validação molecular offline

Este documento descreve somente o kernel físico offline concluído pelo
SCI-004A. Ele não altera a fonte meteorológica consumida pelo LEBEAR produtivo,
não baixa dados e não autoriza calibração quantitativa quando a única fonte é a
atmosfera padrão.

## Hierarquia científica aprovada

A hierarquia futura é: (1) radiossonda do Campo de Marte adequada; (2) perfil
híbrido radiossonda + ERA5; (3) ERA5 puro; (4) atmosfera padrão apenas como
fallback diagnóstico. SCI-004A implementa normalização, reconstrução, fusão e
validação com snapshots locais. Os critérios temporais e a seleção automática
permanecem fora deste kernel.

## Auditoria do caminho anterior

| Componente | Fonte anterior | Unidade | Interpolação | Extrapolação | Fallback | Limitação confirmada |
| --- | --- | --- | --- | --- | --- | --- |
| aquisição de radiossonda | Siphon/University of Wyoming | tabela pandas | sounding nominal 00/12 UTC | não | retorno `None` | janelas fixas, não horário efetivamente mais próximo |
| Nível 1 | colunas `height`, `temperature`, `pressure` | m, K, hPa | ordenação e remoção de altura duplicada | não | superfície apenas | umidade, horário e demais campos não entram no NetCDF |
| Nível 2 P/T | radiossonda embutida | altitude do lidar AGL somada a 760 m; P hPa; T K | linear em P e T | não | padrão em todo bin externo | sem flags de origem/operação; P não usa log-interpolação |
| atmosfera padrão | uma única lei troposférica | m, hPa, K | analítica | ilimitada | sempre | T fixa em 216,65 K acima da tropopausa, mas P mantém a lei da troposfera |
| densidade molecular | escala padrão por P/T | número implícito | n/a | n/a | conforme P/T | sem umidade ou componentes de massa/número explícitos |
| `alpha_m`/`beta_m` | Bucholtz-style | m-1, m-1 sr-1 | n/a | n/a | conforme P/T | testes anteriores verificavam apenas positividade/forma |
| calibração Rayleigh | sinal molecular simulado | fator escalar | janela de referência | n/a | padrão permitido | fallback não impedia uso quantitativo |

O `height` do payload Wyoming era rotulado como altitude MSL sem uma conversão
documentada entre altura geopotencial e altitude geométrica. O cache CSV
preservava todas as colunas devolvidas pelo Siphon depois de ordenar e eliminar
alturas duplicadas; o Nível 1 preservava somente altura, temperatura, pressão,
station ID, disponibilidade e tropopausas.

## Contrato e coordenadas

`AtmosphericProfile` é uma dataclass congelada. Arrays são copiados, marcados
read-only, unidimensionais e conformáveis. A coordenada canônica é
`geometric_altitude_m`: altitude geométrica acima do nível médio do mar/geóide
na aproximação esférica adotada. A altura acima da estação é derivada somente
quando a altitude MSL da estação é fornecida explicitamente.

- geopotencial: `Phi` em m2 s-2;
- altura geopotencial: `H = Phi/g0`, com `g0 = 9,80665 m s-2`;
- altitude geométrica: `z = Re H/(Re-H)`, com `Re = 6.356.766 m`;
- altura AGL do lidar: `z - z_station`.

O contrato inclui P, T, q, temperatura virtual, densidade de massa, densidade
numérica total, componentes seco/vapor, origem, operação, fallback, qualidade,
peso da radiossonda, horários, coordenadas, provider, estação/dataset, SHA-256
do snapshot, versões das fórmulas, cobertura e parâmetros de blend. A conversão
para xarray é explícita; o kernel físico não depende de xarray.

Origem, operação e fallback são dimensões semânticas separadas:

- `PrimarySource`: invalid, radiosonde, ERA5, blended, standard atmosphere;
- `InterpolationFlag`: invalid, direct, interpolated, extrapolated;
- `FallbackFlag`: none, standard atmosphere;
- `HumidityFlag`: measured, derived from dew point, dry assumed, missing.

Pressão finita deve ser positiva e estritamente decrescente; temperatura deve
ser positiva; `q` deve estar em `[0, 0.1]`; altitude deve crescer estritamente.
Estados contraditórios entre fonte, fallback, peso, ausência e qualidade são
rejeitados. Umidade ausente permanece `NaN`; ar seco assumido é `q=0` com flag
própria.

## Radiossonda offline

A fixture local reproduz as colunas usuais do Wyoming/Siphon para `83779`, o
identificador usado atualmente para Campo de Marte. Ela contém duplicata de
altura, nível sem pressão, gaps, dew point, umidade relativa e vento. O
normalizador:

1. converte P de hPa para Pa e T/dew point de Celsius para kelvin;
2. escolhe, em altura duplicada, a primeira linha com maior completude;
3. remove somente níveis sem altura/P/T e preserva a separação vertical restante;
4. calcula `q` do dew point pela expressão de pressão de vapor de Bolton e pela
   razão de mistura `r = epsilon e/(p-e)`, `q=r/(1+r)`;
5. não extrapola até o topo do lidar;
6. reporta resíduo hidrostático em `Delta ln p`, sem ajustar observações.

Na fixture: uma duplicata e um nível incompleto são removidos; permanecem oito
níveis entre 760 e 15.000 m, cinco gaps maiores que 1 km e gap máximo de 3 km.
Quatro camadas possuem umidade suficiente para o diagnóstico hidrostático; o
resíduo absoluto médio em `Delta ln p` é 0,02334 e o máximo 0,06137.

## ERA5 em 137 model levels

A fixture sintética redistribuível usa os 138 pares oficiais `a(n), b(n)` da
grade L137, quatro pontos ao redor do SPU e campos locais determinísticos de T,
q, lnsp e geopotencial de superfície. Não contém nem executa aquisição.

Para cada interface:

```text
sp = exp(lnsp)
p_half(n) = a(n) + b(n) sp
p_full(k) = [p_half(k) + p_half(k+1)]/2
```

O nível 1 é o topo e o 137 o mais próximo da superfície. O geopotencial é
integrado da superfície para cima. Para níveis exceto o topo:

```text
dlogp = ln(p_below/p_above)
alpha = 1 - p_above dlogp/(p_below-p_above)
Phi_full = Phi_half_below + Rd Tv alpha
Phi_half_above = Phi_half_below + Rd Tv dlogp
```

No nível 1, a convenção ECMWF usa `dlogp=ln(p_below/0,1 Pa)` e
`alpha=ln(2)`. O cálculo omite condensado, gelo, chuva e neve, portanto mantém a
mesma aproximação declarada pelo exemplo ECMWF. A validação independente usa
linhas publicadas da tabela L137 a `sp=1013,25 hPa`: 0,02000365 hPa na primeira
interface, 137,2703 hPa na interface 66, 1013,25 hPa na superfície e
1012,0493 hPa no full level 137.

A interpolação espacial é bilinear nos quatro cantos antes da reconstrução e
rejeita pontos fora do retângulo. A ordem dos quatro pontos não altera o
resultado. ecCodes pode ser usado futuramente para leitura, mas não é uma
dependência do kernel.

## Termodinâmica úmida

Para umidade específica `q`, a formulação exata da mistura ideal usada é:

```text
epsilon = Rd/Rv
Tv = T [1 + q(1/epsilon - 1)]
rho = p/(Rd Tv)
rho_v = q rho
rho_d = (1-q) rho
N_d = rho_d N_A/M_d
N_v = rho_v N_A/M_v
N = N_d + N_v
```

O coeficiente de primeira ordem é aproximadamente 0,608, compatível com
`T(1+0,61q)`. A 1013,25 hPa, 296 K e `q=0,016`, a densidade de massa diminui de
1,19252 para 1,18104 kg m-3 (-0,963%). Mantidos P e T, o número total de
moléculas não muda na mistura ideal; a composição e a massa mudam. Numa coluna
hidrostática de 3 km, a umidade eleva a pressão final de 70.888,3 para 71.006,6
Pa (+0,1669%), propagando-se então para a densidade molecular.

SCI-004A não adiciona correção avançada de composição úmida à seção de choque
Rayleigh.

## Interpolações

- vertical: P em `ln(P)`, T/q/Phi lineares; q é limitado ao domínio físico;
- espacial: bilinear em retângulo de quatro pontos;
- temporal: linear entre dois horários ERA5 em grade vertical idêntica;
- extrapolação vertical: desabilitada por padrão e marcada quando habilitada
  explicitamente;
- extrapolação espacial/temporal: rejeitada;
- gaps acima do limite fornecido: permanecem ausentes, sem ponte silenciosa.

Bins coincidentes são `direct`; novos bins internos são `interpolated`; bins
externos explicitamente permitidos são `extrapolated`; ausência é `invalid`.

## Fusão radiossonda + ERA5

As duas fontes são primeiro colocadas numa grade geométrica MSL comum. Onde a
radiossonda é válida, seu peso é 1; fora da cobertura é 0; em cada fronteira de
cobertura usa-se rampa cosseno contínua. T e q são combinadas pelo peso. A
pressão preliminar usa pesos em `ln(P)` e a pressão final é reintegrada por:

```text
d ln p/dz = -g0/(Rd Tv)
```

Isso garante pressão positiva e monotônica sem esconder as diferenças de
entrada. Na fixture (grade de 200 m; blend nominal 1.200 m), a sobreposição vai
de 800 a 6.400 m e o blend de 5.400 a 6.200 m. Diferenças na sobreposição:

| Diagnóstico | média | máximo |
| --- | ---: | ---: |
| T | 6,450 K | 16,513 K |
| Tv | 6,565 K | 16,820 K |
| pressão absoluta | 649,7 Pa | 1.603,7 Pa |
| pressão relativa | 1,020% | 2,376% |
| densidade molecular | 3,393e23 m-3 | 1,030e24 m-3 |

O maior salto de T na troca abrupta seria 16,218 K e cai para 0,295 K; pressão
relativa cai de 4,207% para 2,458% e densidade molecular relativa de 9,878%
para 2,353%. Esses números são diagnósticos, não thresholds universais.

## Coeficientes moleculares 355/532

Mantém-se a composição óptica de ar seco de Bucholtz (1995), agora aplicada à
densidade numérica explícita do contrato:

```text
alpha(lambda,z) = N(z) sigma_R(lambda)
beta(lambda,z,pi) = alpha P_R(pi,lambda)/(4 pi)
S_m(lambda) = alpha/beta
T2(z) = exp[-2 integral alpha dz]
```

O índice de refração padrão, fator de King e depolarização de Bates do módulo
anterior foram auditados. Não foi encontrado erro inequívoco nas equações
espectrais. A única diferença no ponto STP é a atualização da densidade padrão
para constantes SI exatas: -0,02016% em relação à constante histórica
`2,54743e25 m-3`; o caminho produtivo anterior não foi alterado.

| lambda | sigma_R (m2) | alpha STP (m-1) | beta STP (m-1 sr-1) | S_m (sr) |
| ---: | ---: | ---: | ---: | ---: |
| 355 nm | 2,75515005e-30 | 7,01713710e-5 | 8,25189933e-6 | 8,5036630 |
| 532 nm | 5,16482987e-31 | 1,31543904e-5 | 1,54818991e-6 | 8,4966258 |

Os testes independentes calculam `N=p/(k_B T)`, aplicam as seções de choque
tabuladas e reconstroem a fase de retroespalhamento sem chamar o adaptador
produtivo. A tolerância do teste analítico é `2e-10` relativa; a razão espectral
é testada a `2e-15` relativa.

## Atmosfera padrão e autorização

A implementação por camadas cobre `H=0–84,852 km`, com bases de pressão
recalculadas recursivamente para não introduzir saltos por arredondamento.
Temperatura e pressão são validadas nas sete bases publicadas, com tolerância
máxima de `1,2e-4` relativa para P devido às constantes/tabelas arredondadas.

Todo perfil padrão é marcado:

```text
primary_source = standard_atmosphere
profile_quality = fallback_diagnostic
quantitative_retrieval_allowed = false
```

Esse bloqueio é semântico no contrato e nos testes; conectá-lo à calibração
produtiva pertence ao SCI-004C.

## Proveniência, benchmark e fronteiras

Cada perfil registra provider, estação/dataset, horário nominal/real,
coordenadas, SHA-256 do conteúdo bruto, versão do normalizador e da
termodinâmica, cobertura e flags por bin. O híbrido acrescenta método,
parâmetros e peso por bin.

Benchmark local, 30 repetições com `tracemalloc`:

| Etapa | mediana | pico alocado mediano | arrays do perfil |
| --- | ---: | ---: | ---: |
| radiossonda, 8 níveis | 0,01388 s | 36.102 bytes | 872 bytes |
| ERA5 L137 | 0,00702 s | 46.742 bytes | 14.933 bytes |
| híbrido, 385 bins | 0,27678 s | 199.527 bytes | 41.965 bytes |

SCI-004B permitirá rede e implementará aquisição cache-first para Siphon e
ERA5. O LEBEAR poderá iniciar aquisição automática, mas deverá sempre consumir
um snapshot congelado e hasheado. Nenhuma chamada de rede pertencerá aos testes
padrão. SCI-004C implementará seleção temporal/qualitativa e integração
produtiva no LEBEAR. Nenhuma dessas duas etapas foi antecipada.

## Referências

- ECMWF/Copernicus, [ERA5: compute pressure and geopotential on model levels,
  geopotential height and geometric height](https://confluence.ecmwf.int/spaces/CKB/pages/158636068/ERA5+compute+pressure+and+geopotential+on+model+levels+geopotential+height+and+geometric+height).
- ECMWF, [L137 model level definitions](https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions).
- Bucholtz, A. (1995), [Rayleigh-scattering calculations for the terrestrial
  atmosphere](https://doi.org/10.1364/AO.34.002765), *Applied Optics* 34,
  2765–2773.
- NOAA/NASA/USAF (1976), [U.S. Standard Atmosphere,
  1976](https://www.ngdc.noaa.gov/stp/space-weather/online-publications/miscellaneous/us-standard-atmosphere-1976/us-standard-atmosphere_st76-1562_noaa.pdf).
