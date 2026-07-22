# Representação temporal aprovada para o Nível 2

Esta é a decisão aprovada em ENG-032/ADR-003. Ela ainda não altera o retrieval
nem o schema NetCDF atual: ENG-022 e ENG-035 permanecem explicitamente fora do
escopo até a publicação da branch e a validação real da CI.

## Semântica atual

O retrieval calcula um resultado independente por bloco temporal. No cenário
padrão de 15 minutos, `block_time` contém o início arredondado de cada bloco e
as variáveis com sufixo `_block` guardam os valores realmente calculados.

A coordenada `time` preserva os timestamps de aquisição do Nível 1. Quinze
variáveis de Nível 2 repetem o resultado do bloco para cada perfil de aquisição:

- quatro sinais glued e suas incertezas;
- a flag por bin da origem usada no gluing;
- dez flags e diagnósticos escalares de gluing.

Esses valores em `time` não representam retrievals independentes. Eles são
expansões determinísticas das variáveis `_block`. As variáveis `_mean`, por sua
vez, são agregações sobre blocos que passaram gluing e Rayleigh QA; não são
médias ponderadas pela quantidade de perfis repetidos em `time`.

Perfis moleculares e diagnósticos agregados por comprimento de onda não têm
dimensão temporal. Os pares `aerosol_backscatter`/`aerosol_backscatter_mean` e
equivalentes são aliases públicos com valores idênticos, mas sua eventual
remoção não faz parte desta proposta.

## Consumidores encontrados no repositório

| Consumidor | Dependência atual | Impacto de uma representação por bloco |
|---|---|---|
| `milgrau.level2.dataset` | emite `time`, `block_time` e médias simultaneamente | deve emitir schema versionado e mapa explícito `time -> block_time` |
| contrato NetCDF | exige `glued_range_corrected_signal(time, wavelength, altitude)` | deve validar separadamente schemas v1 e v2 |
| QA Matplotlib | exige sinais e diagnósticos expandidos em `time`, embora use médias/medianas | pode preferir `_mean` e diagnósticos `_block`, mantendo leitura v1 |
| Explorer | usa os mesmos nomes do QA e possui renderização genérica de `block_time` | pode aceitar v1/v2 por um adaptador comum |
| proveniência/incremental | fingerprint cobre configuração do Nível 2 | versão do schema deve entrar no contrato/fingerprint da escrita |
| testes | golden fixa nomes, dimensões, dtypes, atributos e bytes | precisa de golden por versão e teste de reconstrução exata |

Não há outro consumidor interno de Nível 2. Consumidores externos não podem ser
inventariados pelo repositório e devem ser protegidos por versionamento e leitura
retrocompatível.

Há ainda uma diferença semântica visível: QA e Explorer chamam de “blocos” o
tamanho de `gluing_success_flag`, mas hoje essa variável está expandida por
perfil. Se blocos tiverem quantidades diferentes de perfis, a taxa atual fica
ponderada por perfis. Usar a variável `_block` calcula uma taxa por bloco real.
Essa correção de interpretação requer aprovação científica.

## Estimativa com o benchmark `typical`

O dataset tem 24 perfis, oito blocos, dois comprimentos de onda e 800 bins.
Todas as 15 variáveis expandidas foram reconstruídas a partir de `_block` e de
um índice `time_block_index(time)`; valores, NaNs e flags foram exatamente
iguais.

| Medida | Schema atual | Projeção por bloco | Redução |
|---|---:|---:|---:|
| tamanho residente do dataset | 2.573.850 bytes | 1.303.578 bytes | 49,35% |
| NetCDF comprimido | 1.221.214 bytes | 855.777 bytes | 29,92% |

A projeção foi feita removendo apenas as 15 expansões e adicionando o índice
explícito. Ela não demonstra ainda redução do pico RSS do pipeline, porque a
transformação foi aplicada depois da montagem completa. Pico, tempo e tamanho
devem ser medidos novamente em ENG-035 se a opção for aprovada.

## Opções

### A — Preservar integralmente o schema v1

Mantém todos os consumidores sem adaptação e permite apenas otimizações
internas que não mudem o arquivo. É a opção de menor risco público, mas mantém a
duplicação de aproximadamente metade do dataset residente no cenário típico.

### B — Tornar `block_time` canônico em um schema v2 — aprovada

O arquivo preserva a coordenada original `time`, os produtos `_block`, as
médias e os perfis sem dimensão temporal. As 15 expansões deixam de ser
persistidas e entra `time_block_index(time)`, índice inteiro que aponta para
`block_time` e permite reconstrução exata mesmo com aquisições irregulares ou
lacunas.

Esta opção preserva informação, torna explícita a granularidade real do
retrieval e oferece a maior economia medida. Ela muda o schema público e a
semântica das estatísticas de sucesso para blocos reais.

### C — Emissão v1/v2 configurável durante transição

Permite adoção gradual, mas duplica contratos, goldens, suporte e caminhos de
configuração. Também prolonga a ambiguidade temporal. Só é indicada se houver
consumidor externo conhecido que não possa migrar junto.

A opção de remover `_block` e manter apenas as expansões em `time` foi rejeitada:
ela conserva a forma maior, mascara a granularidade real e perde diagnósticos
independentes por bloco.

## Migração proposta para a opção B

1. ENG-022 centraliza as definições dos schemas v1 e v2 e introduz um atributo
   explícito de versão.
2. Um adaptador interno oferece acesso canônico por bloco e consegue ler tanto
   arquivos antigos v1 quanto novos v2.
3. QA e Explorer passam a usar médias e diagnósticos por bloco; testes congelam
   as estatísticas aprovadas.
4. ENG-035 muda somente o writer novo para v2, mantém leitura de v1 e adiciona
   `time_block_index` com teste de reconstrução exata.
5. Arquivos v1 existentes continuam válidos e não são reescritos
   automaticamente. Uma conversão, se necessária, será uma ação explícita.
6. O benchmark `typical` compara pico RSS, tempo e NetCDF com ENG-030; o cenário
   `large` continua local.

## Condições da decisão aprovada

A opção B foi aprovada pela mantenedora em 2026-07-22 com as seguintes
condições vinculantes para ENG-022 e ENG-035:

1. `block_time` será a dimensão canônica de produtos calculados em blocos de 15
   minutos.
2. A coordenada de aquisição `time` será preservada. A variável inteira
   `time_block_index(time)` apontará para a posição de `block_time`; valores
   válidos estarão em `0..B-1` e `-1` significará perfil sem bloco associado.
   Leitores devem mascarar `-1` antes de qualquer indexação.
3. `block_time` representa o início inclusivo da janela, obtido pelo floor de
   15 minutos; o fim é exclusivo. O v2 terá
   `block_start_time(block_time)`, `block_end_time(block_time)` e
   `block_profile_count_available(block_time)`. Uma contagem
   `block_profile_count_valid` será emitida quando houver definição não
   ambígua; se a
   validade variar por variável, comprimento de onda ou altitude, a contagem
   terá essas dimensões e não será reduzida a um escalar enganoso.
4. Variáveis científicas em `time` reconstruíveis exatamente não serão
   persistidas no v2. A reconstrução será feita somente pelas variáveis
   canônicas em `block_time` e por `time_block_index`.
5. A métrica principal será
   `retrieval_block_success_fraction(wavelength)`, calculada sobre blocos reais.
   A métrica distinta
   `retrieval_profile_coverage_fraction(wavelength)` será a fração de todos os
   timestamps originais associados a blocos válidos; perfis com índice `-1`
   contam como não cobertos. Flags de bloco continuam disponíveis para auditoria.
6. Médias mantêm exatamente a ponderação científica atual: cada bloco válido
   tem o mesmo peso em `valid_block_mean`, e a combinação de erros permanece a
   de `valid_block_error`. Não haverá ponderação por número de perfis nesta
   tarefa.
7. Arquivos novos terão atributo explícito
   `MILGRAU_Level2_Schema_Version = "2.0"`. A ausência desse atributo identifica
   o legado v1. Leitores internos aceitarão v1 e v2, e nenhum arquivo antigo
   será reescrito automaticamente.
8. QA e Explorer usarão dados canônicos por bloco para estatísticas do retrieval
   e poderão relacioná-los a `time` pelo mapa explícito.
9. Goldens cobrirão reconstrução exata das antigas expansões em `time`, quando
   aplicável, e preservação de valores, NaNs, flags, unidades, atributos e
   semântica. Casos com `time_block_index = -1`, blocos incompletos e lacunas
   temporais serão obrigatórios.
10. Esta decisão deve permanecer apenas documental até a organização dos
    commits, publicação da branch e CI real verde em Python 3.12 e 3.14. Só
    então ENG-022 ou ENG-035 poderá ser iniciado.
