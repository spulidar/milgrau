# MILGRAU — SCC Refactor & Main Integration Tracker

**Branch de trabalho:** `scc-integration-refactor`  
**Branch de destino:** `main`  
**Status:** Implementação SCC funcional; simplificação arquitetural e integração com `main` pendentes.  
**Atualizado:** 20/10/2026

## Objetivo

Concluir a integração opcional do SCC ao MILGRAU, simplificar a implementação, remover ferramentas transitórias e incorporar as melhorias recentes do `main`, preservando a independência dos produtos científicos L0/L1/L2.

O resultado deverá ser uma implementação enxuta, previsível e fácil de manter, sem compatibilidade desnecessária com formatos antigos.

## Princípios arquiteturais

1. O MILGRAU define sessões científicas contínuas. O SCC possui identificadores e partições próprios.
2. L0/L1/L2 são independentes do SCC. Falhas de exportação, upload ou download não invalidam produtos científicos.
3. SCC é opcional. Quando desabilitado, o processamento científico não depende de `scc.yaml` nem dos serviços externos.
4. `station.yaml` mantém informações instrumentais, calibrações e climatologia; `scc.yaml` mantém exclusivamente configurações e identificadores SCC; `config.yaml` controla execução e opções operacionais.
5. Os inputs e resultados SCC pertencem à respectiva sessão científica.
6. Uploads e exclusões remotas são operações explícitas, nunca acionadas automaticamente pelo processamento científico.
7. `--replace-existing --confirm` autoriza substituição remota sem exigir backup prévio, mas não permite retries cegos de operações incertas.
8. O servidor SCC é a referência para o estado remoto. O journal local registra operações, hashes e situações incertas, sem ser versionado no Git.
9. Arquivos históricos, evidências de incidentes e medições remotas não devem ser apagados automaticamente durante refatorações.
10. Evitar camadas de compatibilidade e ferramentas de reparo que não atendam a um requisito atual.

---

## 0. Baseline e estado atual

### Implementação concluída

- [x] Separar a configuração SCC das configurações científicas e instrumentais.
- [x] Preservar os identificadores reais de canais, configurações e produtos SCC.
- [x] Implementar geração de pacotes SCC por regime solar e mês UTC, sem dividir a sessão científica.
- [x] Utilizar `session/scc/scc-inputs/<Measurement_ID>/` para pacotes de entrada.
- [x] Utilizar `session/scc/scc-results/<Measurement_ID>/` para resultados remotos.
- [x] Validar arquivos de entrada por manifesto e SHA-256.
- [x] Implementar cliente HTTPS com autenticação, upload, status e download.
- [x] Implementar upload individual e em lote.
- [x] Implementar substituição explícita sem backup obrigatório com `--replace-existing --confirm`.
- [x] Manter journal SQLite local em `.state/scc/`, ignorado pelo Git.
- [x] Permitir consulta remota independente da existência do cache local.
- [x] Permitir download sem journal quando a sessão puder ser identificada pelo manifesto.
- [x] Impedir que falhas opcionais de finalização SCC invalidem um L1 científico válido.
- [x] Aprovar os seis jobs de CI da implementação SCC atual.

### Referências de auditoria (antes da integração)

- `main`: `827a2e5`
- `scc-integration-refactor`: `d204a64`
- Merge base: `7d8032e`
- Divergência identificada: 18 commits exclusivos de `main` e 356 da branch SCC.

Esses identificadores registram o baseline original da auditoria.

### Estado após a Fase 1 (20/10/2026)

- `main` preservada em `827a2e5`.
- `scc-integration-refactor` integrada no merge `a6b598c` (PR #4).
- Branch de preparação: `integration/scc-main`, commit validado `89858c4`.
- Histórico de `main` incorporado sem rebase ou push forçado; commit de integração com os dois pais.
- CI do PR: 6/6 jobs aprovados — [GitHub Actions](https://github.com/spulidar/milgrau/actions/runs/39486432591).
- Nenhum upload, DELETE ou substituição remota SCC foi executado pela integração.


### Pendências de preparação

- [x] Conferir se os HEADs remotos mudaram desde o baseline.
- [ ] Inventariar pacotes SCC históricos ainda presentes no computador.
- [ ] Identificar journals com operações incertas que ainda precisam de reconciliação.
- [ ] Preservar evidências e arquivos relevantes antes de eliminar ferramentas históricas.
- [x] Manter operações remotas destrutivas desabilitadas durante os testes automatizados.

**Gate 0:** estado atual conhecido, preservado e reproduzível.

---

## 1. Merge das melhorias recentes de `main`

### 1.1. Estratégia

- [x] Atualizar e comparar os HEADs remotos por API GitHub (equivalente à verificação de `git fetch origin`).
- [x] Criar uma branch temporária `integration/scc-main` a partir de `scc-integration-refactor`.
- [x] Registrar a integração com commit Git de dois pais por API (equivalente histórico ao merge de `origin/main`, sem rebase).
- [x] Resolver conflitos preservando os novos comportamentos científicos do `main`.
- [x] Não utilizar `git push --force` nem reescrever o histórico das duas branches.
- [x] Executar os testes antes de incorporar o merge à branch SCC.
- [x] Manter `main` inalterada até a aprovação do Pull Request final.

### 1.2. Preservar as mudanças do LIBIDS

Os 18 commits exclusivos de `main` incluem alterações que deverão ser mantidas:

- [x] Preservar o inventário de sessões científicas contínuas.
- [x] Preservar a resolução leve de identidade de perfil e calibração por cabeçalho Licel.
- [x] Preservar a otimização de associação dos dark currents por busca temporal.
- [x] Preservar o tratamento de sessões sobrepostas, empates e limites temporais.
- [x] Preservar a normalização de timestamps em nanossegundos.
- [x] Preservar os diagnósticos de sessões muito curtas ou IDs inválidos.
- [x] Preservar o processamento de QA individual por sessão.
- [x] Garantir que uma sessão rejeitada pelo QA não impeça o processamento de outras sessões.
- [x] Manter mensagens resumidas de progresso no INFO.
- [x] Manter diagnósticos detalhados de Licel, QA e SCC no DEBUG.
- [x] Preservar `.dat` na lista de arquivos espúrios ignorados.

### 1.3. Arquivos que exigem revisão conjunta

| Arquivo | Resolução esperada |
|---|---|
| `config.yaml` | Combinar configuração opcional SCC com as configurações atuais do `main`, incluindo `.dat` |
| `milgrau/level0/libids.py` | Integrar QA por sessão, processamento incremental e `--scc-reexport` |
| `milgrau/level0/netcdf.py` | Preservar diagnóstico temporal SCC em DEBUG |
| `tests/test_level0_netcdf.py` | Preservar testes de ambos os fluxos |
| `tests/test_pipeline_orchestration.py` | Cobrir QA por sessão e integração SCC opcional |

Esses arquivos possuem alterações nas duas branches e devem ser revisados mesmo que o Git consiga combiná-los automaticamente.

### 1.4. Testes específicos do merge

- [ ] Verificar seleção e processamento de uma única sessão.
- [x] Verificar inventário de sessões atravessando dias.
- [x] Verificar inventário de sessões atravessando meses UTC.
- [x] Comparar os resultados da associação otimizada de dark currents com uma implementação de referência.
- [x] Verificar identificação de perfil e calibração antes e depois do merge.
- [x] Confirmar que sessões rejeitadas pelo QA não bloqueiam sessões válidas.
- [x] Confirmar que SCC desabilitado não exige `scc.yaml`.
- [x] Confirmar que SCC habilitado gera inputs para os regimes e meses elegíveis.
- [x] Confirmar que a ausência de inputs SCC não torna o L0 científico desatualizado.
- [x] Confirmar que `--scc-reexport` preserva os bytes e o hash do L0 científico existente.
- [ ] Verificar que reexportação não substitui automaticamente pacotes SCC já submetidos.

**Gate 1:** merge tecnicamente resolvido e comportamento científico das melhorias do `main` preservado.

**Situação (20/10/2026):** integração de código concluída e 6/6 jobs automatizados aprovados. Os dois itens não marcados na seção 1.4 ainda exigem verificação operacional específica: seleção explícita de uma única sessão e ausência de substituição remota já submetida durante reexportação. As validações com dados reais e os inventários locais pertencem à preparação/validação subsequentes; não foram executados via GitHub.

---

## 2. Simplificação do workflow SCC

### Progresso da Fase 2 — integrada no PR #5 (20/10/2026)

- Merge de integração em `scc-integration-refactor`: `d915f28`, via [PR #5](https://github.com/spulidar/milgrau/pull/5).
- Código da Fase 2 validado em [GitHub Actions, 6/6 jobs aprovados](https://github.com/spulidar/milgrau/actions/runs/39490065222); push de merge e tracker sujeitos às execuções posteriores de CI.
- Identidade imutável de manifesto/hashes centralizada no journal SQLite, sem migração destrutiva do schema.
- Reconciliação de DELETE incerto com arquivo verificado **ou** dispensa explícita de backup registrada na revisão corrente; SCC deve confirmar ausência antes de qualquer POST.
- Um único caminho journalizado de POST de recuperação mantém estados incertos bloqueados e nunca repete DELETE.
- `status --remote` fornece orientação somente de leitura para estados incertos, inclusive quando a API retorna 404 (ausência não autoriza retry de POST).
- Preflight da substituição sem backup ocorre antes de reiniciar a revisão do journal, preservando histórico em caso de falha.
- Os comandos específicos para incidentes antigos foram mantidos provisoriamente: sua remoção depende do inventário de arquivos e journals locais das Fases 0/3.
- Segurança verificada em testes simulados, sem POST/DELETE reais. A validação com servidor SCC e dados históricos permanece pendente.

### 2.1. `scc/workflow.py`

- [ ] Separar claramente três operações: envio novo, substituição explícita e reconciliação de estado incerto.
- [x] Unificar validações repetidas de ID, manifesto, hashes e configuração.
- [ ] Reduzir caminhos condicionais associados a incidentes históricos.
- [x] Manter preflight remoto antes de qualquer exclusão.
- [x] Manter confirmação explícita para substituições.
- [x] Tratar a substituição sem backup como fluxo operacional normal autorizado.
- [x] Implementar reconciliação de DELETE incerto mesmo quando nenhum backup foi produzido.
- [x] Padronizar reconciliação após POST incerto.
- [x] Permitir nova tentativa somente após avaliação explícita do estado remoto e dos registros disponíveis.
- [x] Manter proteção contra repetição automática de POST/DELETE após resultado desconhecido.
- [ ] Evitar manter diversos comandos de recuperação específicos para incidentes antigos.

### 2.2. `scc/backup.py`

- [x] Tornar o backup remoto estritamente opcional.
- [x] Remover dependências obrigatórias de backup do fluxo de substituição confirmada.
- [ ] Manter apenas funções que sejam necessárias para backup solicitado pelo operador ou verificação de dados.
- [ ] Avaliar centralizar verificações de identidade remota em um único módulo.
- [x] Evitar duplicação de download de arquivos já implementado em `results.py`.

### 2.3. `scc/state.py`

- [x] Manter SQLite como histórico operacional local, sem utilizá-lo como fonte do estado atual do servidor.
- [x] Preservar registro das operações potencialmente irreversíveis e seus resultados incertos.
- [ ] Revisar estados e transições redundantes.
- [x] Centralizar a reconciliação do journal.
- [x] Garantir que `status --remote` funcione sem journal local.
- [x] Garantir que download de resultados não exija journal quando a sessão for conhecida.
- [x] Preservar leitura de journals históricos necessários à resolução de operações pendentes.
- [x] Manter journal e arquivos operacionais fora do GitHub.

**Gate 2:** upload, substituição e recuperação utilizam um fluxo coerente, sem dependências artificiais de backups ou incidentes antigos.

---

## 3. Remoção do código transitório

**Fase 3 — implementação concluída no código (20/10/2026):** por confirmação do operador de que os journals estão resolvidos, retirados os módulos históricos de migração/reparo/backfill e os comandos exclusivos desses incidentes. A inspeção do computador do operador não foi realizada por esta integração; nenhuma pasta histórica, journal ou pacote local foi apagado. O código atualizado passou nos seis jobs de CI de [GitHub Actions #39494187032](https://github.com/spulidar/milgrau/actions/runs/39494187032), incluindo 708 testes no Linux/Python 3.12. Os testes operacionais reais continuam sendo gates separados.

### 3.1. Ferramentas históricas

| Módulo histórico | Resultado |
|---|---|
| `scc/migrate.py` | Removido |
| `scc/lr_repair.py` | Removido |
| `scc/sounding_repair.py` | Removido |
| `scc/sounding_backfill.py` | Removido; novos soundings continuam no fluxo automático LIPANCORA |

- [ ] Inventário local de pacotes históricos (não verificável via GitHub; operador confirmou somente journals resolvidos).
- [ ] Distinguir eventuais pacotes locais não submetidos daqueles com histórico remoto, caso ainda existam.
- [ ] Identificar eventuais pacotes incompatíveis que precisem de reexportação.
- [ ] Reexportar manualmente apenas eventuais inputs históricos não submetidos que sejam necessários, preservando evidências. Não existe migração automática.
- [x] Preservar registros de operações incertas ainda não resolvidas.
- [x] Remover os módulos históricos quando não houver dependências operacionais relevantes.
- [x] Remover os comandos CLI exclusivos das ferramentas eliminadas.
- [x] Remover testes exclusivamente associados a funcionalidades removidas.
- [x] Preservar testes de segurança e integridade que continuem relevantes ao fluxo novo.

### 3.2. Remoção do `milgrau-scc cleanup` — decisão operacional

Os inputs SCC são artefatos persistidos por sessão científica, não cache
descartável. A limpeza será responsabilidade manual do operador, fora da CLI,
**sem exclusão automática após upload ou download**.

- [x] Remover o subcomando `milgrau-scc cleanup`, seu handler e o módulo `scc/cleanup.py` quando não houver consumidores.
- [x] Remover `scc.cleanup.policy` da configuração, incluindo o legado `after_download`.
- [x] Remover `download --keep-cache` e todos os caminhos de exclusão pós-download; downloads devem sempre preservar os inputs SCC.
- [x] Revisar mensagens, testes e documentação para usar "inputs SCC por sessão" em vez de "cache descartável".
- [x] Preservar a interpretação do estado histórico `cleaned` em journals SQLite antigos, sem apagar nem migrar registros.
- [x] Não apagar automaticamente pacotes com envio pendente, POST/DELETE incerto ou evidências ainda necessárias.
- [x] Documentar a exclusão manual restrita a `<sessão>/scc/scc-inputs/<Measurement_ID>/`, somente após verificar status remoto, journal e necessidade de reenvio/auditoria; nunca apagar `scc-results`, `.state/scc`, L0/L1/L2 ou figuras.
- [x] Garantir por testes que upload, download, status e reexportação não apagam inputs.

### 3.3. `level0/scc_staging.py`

- [x] Remover lógica de descoberta e recuperação de layouts históricos depois da transição.
- [x] Manter apenas os diretórios de entrada por sessão.
- [x] Preservar criação atômica dos pacotes.
- [x] Preservar prevenção de colisões de `Measurement_ID`.
- [x] Preservar manifesto, hashes e identificação da sessão científica.
- [x] Garantir que nenhuma etapa altere L0/L1/L2 como consequência de limpeza SCC.

### 3.4. `scc/sounding_finalize.py`

- [x] Simplificar a finalização dos soundings para o layout atual por sessão.
- [x] Preservar apenas soundings observados e rastreáveis.
- [x] Preservar `missing_policy: skip` e `fail` enquanto necessários.
- [x] Não alterar pacotes com histórico de submissão.
- [x] Reduzir a dependência operacional entre LIPANCORA e preparação SCC.
- [x] Manter falhas SCC isoladas do resultado científico do L1.

**Gate 3:** o fluxo comum não depende de funções de migração ou reparos históricos.

---

## 4. CLI e configuração

**Fase 4 — implementação:** os três comandos `upload`, `status` e `download`
foram mantidos. Removidos `--continue-on-error` e os parâmetros
`scc.upload.*` (`automatic` sem efeito; `retries` ambíguo). A
configuração `scc.http.timeout_seconds`/`scc.http.get_retries` limita
explicitamente retries a GETs. Erros operacionais de batch passam a
interromper sempre a execução, sem repetir POST/DELETE. A recuperação
do DELETE incerto permanece exclusiva de `upload --id ID
--resume-after-delete --confirm`. Código, testes e documentação atualizados diretamente na branch SCC.
[GitHub Actions #39495985015](https://github.com/spulidar/milgrau/actions/runs/39495985015): **6/6 jobs aprovados** (Ruff, Web Explorer e pytest Linux/Windows Python 3.12/3.14; 722 testes aprovados no Linux/Python 3.12).

**Decisão de integração:** manter `scc.enabled: true` nesta branch
experimental. Para a futura integração na `main`, avaliar explicitamente
o default conforme a política de publicação (sem alterar `main` agora).

### 4.1. Interface operacional

Manter como interface principal:

```bash
milgrau-scc upload
milgrau-scc status
milgrau-scc download
```

- [x] Preservar upload individual e em lote.
- [x] Preservar `--dry-run` para validação local.
- [x] Preservar `--replace-existing --confirm` para substituição explícita.
- [x] Preservar consulta remota somente de leitura.
- [x] Preservar download em lote.
- [x] Consolidar a recuperação de operações incertas em uma interface clara e documentada.
- [x] Remover opções exclusivas de migrações ou reparos eliminados.
- [x] Revisar nomes de opções que ainda utilizam o termo `cache` para pacotes persistidos nas sessões.
- [x] Manter preparação de sounding apenas se ainda for uma operação manual necessária.

### 4.2. `config.yaml`

- [x] Remover `scc.upload.automatic` (e o grupo `scc.upload`) sem efeito de upload.
- [x] Substituir `scc.upload.retries` por `scc.http.get_retries`, aplicável apenas a GET; POST/DELETE nunca são repetidos automaticamente.
- [x] Remover toda a configuração `scc.cleanup` (incluindo `after_download` e `manual`); a CLI não gerencia exclusão de inputs.
- [x] Simplificar opções de limpeza e flags redundantes.
- [x] Remover referências restantes ao staging global.
- [x] Preservar `scc.enabled: true` na branch experimental.
- [ ] Decidir explicitamente o valor padrão de SCC antes do merge em `main`.

### 4.3. Retirada do cleanup da interface operacional

- [x] Remover `scc/cleanup.py` e `milgrau-scc cleanup` na Fase 3.
- [x] Não oferecer exclusão automática de inputs via download, upload ou status.
- [x] Preservar leitura de estados históricos `cleaned` e proteção contra operações incertas.
- [x] Documentar exclusão de diretórios de input pelo operador, após auditoria.

**Gate 4:** três comandos operacionais principais, configuração HTTP explícita com retry apenas de GET, batch fail-stop e nenhuma exclusão automática de inputs SCC. Configuração padrão na futura `main` fica para o Gate 7.

---

## 5. Validação e regressões

**Fase 5 — CI e cenários simulados validados em 20/10/2026:**
[GitHub Actions #39497902391](https://github.com/spulidar/milgrau/actions/runs/39497902391) aprovou **6/6 jobs** no código `a3c6c211`, incluindo **726 testes aprovados no Linux/Python 3.12**. Também foi reforçada a confirmação pós-upload: configuração SCC divergente ou ausência de `creation_date` mantém o journal `uncertain`, sem retry. A matriz e o procedimento de validação real estão em [`docs/scc_phase5_validation.md`](../docs/scc_phase5_validation.md).

**Limite do gate:** testes em CI utilizam mocks e arquivos científicos sintéticos. Nenhum pacote de dados reais do operador foi comparado antes/depois, nenhuma API SCC real foi acionada, nenhum upload/DELETE remoto foi realizado. Portanto o Gate 5 completo continua pendente, embora a parte automatizada esteja aprovada.

### 5.1. Testes automatizados

- [x] Executar Ruff.
- [x] Executar pytest no Linux com Python 3.12.
- [x] Executar pytest no Linux com Python 3.14.
- [x] Executar pytest no Windows com Python 3.12.
- [x] Executar pytest no Windows com Python 3.14.
- [x] Executar os testes do Web Explorer.
- [x] Confirmar 6/6 jobs aprovados no commit final.

### 5.2. Testes científicos

- [x] Verificar continuidade e identidade das sessões.
- [x] Verificar QA por sessão.
- [x] Verificar associação dos dark currents.
- [x] Verificar processamento incremental.
- [x] Verificar processamento sem SCC.
- [x] Verificar geração SCC sem modificar produtos científicos.
- [x] Verificar `--scc-reexport`.
- [ ] Verificar resultados consistentes em amostras representativas antes e depois do merge.

### 5.3. Testes SCC

- [x] Validar pacote novo com `--dry-run`.
- [x] Simular envio de um ID ausente.
- [x] Simular ID existente sem autorização de substituição.
- [x] Simular substituição explicitamente confirmada sem backup.
- [x] Simular DELETE incerto sem backup e posterior reconciliação.
- [x] Simular POST incerto sem retry automático.
- [x] Verificar lote contendo pacotes válidos e inválidos.
- [x] Verificar status remoto sem arquivos locais.
- [x] Verificar download sem journal local, com manifesto válido.
- [x] Verificar gerações de resultados com proveniência preservada.
- [x] Verificar que nenhum comando SCC apaga inputs de sessão automaticamente, nem produtos científicos ou resultados.

### 5.4. Validação real com SCC

- [x] Executar `milgrau-scc upload --dry-run` em dados reais (**1/45 pacotes válido** no lote; 44 pacotes existentes aguardando finalização LIPANCORA, não considerar todos aprovados).
- [x] Verificar o ID `20200107spu1130` no SCC real: ausente antes do POST, presente depois, configuração 565 e intervalo UTC confirmado.
- [x] Confirmar upload real **do ID `20200107spu1130`**, com resposta `SCC API confirmed`, journal `accepted` e status remoto `present` (testes realizados pelo operador; **não são execuções CI**).
- [ ] Confirmar aceitação de sounding observado, quando aplicável.
- [ ] Confirmar download e integridade dos resultados.
- [ ] Realizar testes de substituição destrutiva somente com autorização explícita e ID de teste adequado.

**Evidência operacional do operador (20/10/2026):** `upload --dry-run --id 20200107spu1130` OK (configuração 565, raw/LR/RS); `status --remote` inicialmente NOT FOUND; `upload --id 20200107spu1130 --confirm` confirmou criação no SCC; `status --remote` confirmou ID, configuração e journal `accepted`/`processing`. Após processamento o ELPP reportou **-127** e ELDA **0**; o download recusou legitimamente por não haver conclusão ELPP/ELDA. Não marcar download, sounding remoto processado com sucesso ou equivalência científica antes/depois como aprovados. Logs mostraram que L0/L1 preservaram hashes após tentativa de reexport que foi bloqueada por pacote já existente. O operador observou que inventário de `-i` percorreu ~246.853 arquivos antes de selecionar uma sessão. A saída local indicou branch `main`; confirmar HEAD local/branch antes de atribuir os resultados exclusivamente à branch experimental.

**Gate 5 (parcial):** testes automatizados aprovados e nenhuma regressão científica conhecida nos cenários simulados. Comparação de amostras científicas reais antes/depois e verificação real no servidor SCC permanecem obrigatórias antes de declarar validação integral.

### 5.5. Otimização de inventários e planejamento atmosférico — plano aprovado (20/10/2026)

**Decisão:** implementação de código iniciada e integrada **antes da Fase 6**, diretamente em `scc-integration-refactor`, sem PR e sem alteração de `main`. A bateria de regressão foi aprovada no código em [GitHub Actions #39529676453](https://github.com/spulidar/milgrau/actions/runs/39529676453), **6/6 jobs**, commit `77615712` (737 testes no Linux/Python 3.12). A comparação científica e benchmarks em dados reais permanecem gates independentes. O cache LIBIDS e o cache ERA5 são reconstruíveis e **não substituem** a proveniência científica L0/L1/L2 nem o journal operacional SCC. Não executar operações remotas SCC destrutivas.

**Motivação observada na máquina do operador:** `milgrau-libids -i <Session_ID> --scc-reexport` leu 246.853 candidatos Licel e levou 276,9 s para inventariar seis anos antes de selecionar uma sessão (137 medições e 22 dark currents). A execução real do SCC aceitou `20200107spu1130` (configuração 565), mas o ELPP retornou `-127`, impossibilitando download; não atribuir a causa às nuvens sem diagnóstico do SCC. Pacotes históricos `awaiting_lipancora` não mudam quando `scc_policy.sounding.enabled` é desligado depois da criação. A branch local utilizada pelo operador deve ser confirmada antes de validar conclusões de regressão.

#### 5.5.1. LIBIDS: cache persistente apenas de cabeçalhos Licel

- [x] Criar índice **reconstruível**, preferencialmente SQLite em `.cache/libids/inventory.sqlite`, sem torná-lo journal científico nem versioná-lo no Git.
- [x] Na primeira execução, examinar cabeçalhos Licel e guardar caminho, timestamps UTC, duração, disparos e frequência; classificação permanece derivada da varredura atual, não persistida como verdade.
- [ ] Em execuções seguintes, fazer descoberta leve no filesystem; reutilizar cabeçalhos se caminho, tamanho, `mtime_ns` e versão do parser/esquema forem compatíveis. Ler somente novos/alterados; remover registros de arquivos excluídos. Se o índice estiver ausente/inválido, reconstruí-lo com segurança.
- [x] Aplicar seleções `-i` por sessão/data **antes** da resolução de identidade e sessãoização, com margem temporal de dark currents; seletor explícito de caminho continua lendo o índice completo, mas QA só ocorre após seleção.
- [x] Preservar exatamente a sessãoização global, travessias de dias/meses, perfis/calibrações, empates e associação de dark currents na janela configurada. Cabeçalhos podem ser cacheados, mas **sessões, QA e decisões científicas não devem ficar congelados no SQLite**.
- [x] Sem `-i`, processar o inventário completo a partir dos cabeçalhos indexados, atualizando apenas alterações; sem falhas silenciosas quando dados ou configuração mudarem.
- [ ] Testar índice frio/quente, arquivos acrescentados/modificados/excluídos, mudança do parser, mudança de configuração científica, períodos de fronteira e equivalência contra o inventário completo.
- [ ] Medir na máquina do operador tempo de descoberta, leitura de cabeçalhos, seleção e QA. Critério principal: `-i` não exige reler ~246 mil cabeçalhos inalterados.

#### 5.5.2. SCC-only reexport: reutilizar L0 e índice

- [x] `milgrau-libids -i SESSION_ID --scc-reexport` deve localizar o L0 canônico existente e os arquivos Licel de origem via proveniência do L0 + índice de cabeçalhos, sem refazer a leitura integral do acervo.
- [x] Como o L0 atual pode registrar **nomes base** (`Source_Files`), e não caminhos absolutos, resolver nomes somente quando únicos; sinalizar ausência/ambiguidade, sem escolher um arquivo arbitrariamente.
- [ ] Preservar QA relevante, tempos/segmentos, identidade instrumental e hashes dos L0/L1/L2. Nunca reescrever ciência canônica nem sobrescrever automaticamente inputs SCC já staged/journalled/aceitos.
- [ ] Comparar reexport por índice versus inventário completo sobre sessões reais e sintéticas, incluindo dark currents e múltiplos arquivos com o mesmo basename.

#### 5.5.3. LIPANCORA: planejar ERA5 globalmente e baixar em lote no run normal

- [x] Integrar o prefetch ao `milgrau-lipancora` existente, **sem comando adicional obrigatório**: com `-i` considerar somente L0/sessões selecionados; sem `-i` descobrir todos os L0 elegíveis e planejar os horários de uma vez, antes de processar os L1.
- [x] Ler períodos UTC dos L0; calcular exatamente as horas de análise requeridas pela cadência da atmosfera, inclusive horas de enquadramento e sessões que atravessam dias; deduplicar as horas de todas as sessões.
- [x] Criar plano reconstruível em `.cache/era5/requests/<hash>.json`, com requisição, horas esperadas, campos, resultado e estado; o inventário de horas é calculado em memória dos L0 e os NetCDF horários validam a disponibilidade, sem `inventory.json` redundante.
- [x] Validar os arquivos horários existentes (integridade, configuração, coordenadas/grade, níveis e variáveis) antes de considerá-los disponíveis; baixar **somente horas ausentes** e evitar solicitar horas de L1 já atualizado, salvo necessidade explícita de recomputação científica.
- [x] Selecionar horários esparsos, agrupando apenas dias de um mesmo mês com listas idênticas de horas; contabilizar o produto cartesiano CDS e **não solicitar horas extras**. Agrupamento com overfetch intencional ficou dispensado por ora.
- [x] Usar **teto inicial de 80.000 campos por request** (referência discutida: 120.000 campos, **não garantia de aceitação/capacidade**). Configuração atual: temperatura + geopotencial × 37 níveis = **74 campos por hora**; máximo matemático de 1.081 combinações hora-localização por request sob esse teto. Dividir por período/limites do CDS, tamanho estimado e produto cartesiano; evitar request único de múltiplos anos.
- [ ] Despachar **uma solicitação ativa por vez**, com controle de saturação, esperas e retomada segura. Registrar IDs/estado de jobs quando o CDS permitir; não reabrir cegamente um pedido que continua em andamento nem criar filas massivas. Distinguir falha de capacidade, falha definitiva e resposta parcial.
- [x] Validar cada resposta NetCDF, extrair apenas horas correspondentes, gravar arquivos horários e sidecars de forma atômica com proveniência; reutilizar o layout `.cache/era5/YYYY/MM/` e preservar todos os arquivos já obtidos quando um batch posterior falhar.
- [x] Para reprocessamento histórico definitivo, definir política explícita de ausência ERA5 (preferencialmente exigir ERA5 antes de concluir um L1 definitivo). Manter USSA76 como fallback **deliberado**, com fonte registrada; não marcar como cientificamente atualizado um L1 provisório apenas porque CDS estava indisponível. Preservar resultados e física do LIPANCORA.
- [ ] Testar horários esparsos, dias com padrões iguais/diferentes, meses/anos e UTC, deduplicação, produto cartesiano, limite de 80k, cache hit/miss, dados inválidos, retomada, throttling e equivalência de temperatura/pressão dos L1 contra perfis obtidos pelo fluxo atual.
- [ ] Benchmark real: número de requests, campos solicitados úteis/extras, hits de cache, tempo até L1 e comportamento com CDS saturado. Não confundir limite de campos com garantia de slot disponível.

**Follow-up operacional em 20/10/2026 (logs `libids(6).log` / `lipancora(2).log`):**

- **LIBIDS cache real comprovado:** primeira leitura `0 reused / 246853 parsed` (~271 s de cabeçalhos) e segunda `246853 reused / 0 parsed` (~17 s), ambas identificando **575 sessões**. Benchmark real de header cache satisfeito; equivalência de variáveis científicas e custo de QA/sessão ainda não verificados integralmente.
- **Regressão real ERA5 identificada e corrigida no código:** 552 L0 falharam no preflight global com `missing or empty Level 0 time coordinate`; os L0 nativos têm dimensão `time` e `Raw_Data_Start_Time`, mas não coordenada datetime `time` até a ingestão L1. O pré-download foi corrigido para reutilizar `_decode_level0_time_axis`, inclusive na resolução temporal do perfil da estação, com teste reproduzindo NetCDF L0 nativo sem coordenada `time`. O run anterior retornou a um request isolado de 2h/148 campos; isso **não** valida batching global nem prova erro de download CDS. A correção da decodificação foi validada em [GitHub Actions #39531955552](https://github.com/spulidar/milgrau/actions/runs/39531955552), commit `c940b9ba`, **6/6 jobs, 739 testes aprovados no Linux/Python 3.12**. O teste com L0 nativo sem coordenada `time` agora está incluído. Essa evidência é offline; o CDS real e o bulk continuam pendentes.
- **Aguardando reexecução local:** confirmar `global ERA5 plan | <N> L0 files | <M> unique site-hours` no início do LIPANCORA, em vez de 552 avisos; validar número/limite dos requests enviados e comportamento do CDS. O CI testa somente mocks; não marcar o pré-download ERA5 real como aprovado antecipadamente.

**Gates de aceitação desta extensão da Fase 5:** equivalência científica e temporal comprovada, reexport não destrutivo, nenhuma dependência de estado persistente irreconstruível, seleção eficiente de sessão única, requests ERA5 apenas necessários (com overhead contabilizado), nenhum request >80k campos e CI 6/6. A validação remota SCC e a comparação L0/L1/L2 com dados reais continuam gates separados.

---

## 6. Documentação e finalização

- [ ] Atualizar README com a arquitetura SCC final.
- [ ] Atualizar `docs/scc_batch_operator_guide.md`.
- [ ] Reduzir `docs/scc_native_client.md` ao protocolo, arquitetura e operações relevantes.
- [ ] Atualizar referências aos diretórios por sessão.
- [ ] Remover instruções operacionais obsoletas.
- [ ] Revisar a instalação opcional do SCC.
- [ ] Atualizar este checklist conforme cada fase for concluída.
- [ ] Garantir que os exemplos de comandos correspondam ao código implementado.

---

## 7. Merge final em `main`

- [ ] Atualizar os refs remotos e verificar se surgiram novos commits em `main`.
- [ ] Resolver qualquer divergência adicional.
- [ ] Revisar o diff completo do Pull Request.
- [ ] Confirmar que nenhum arquivo local, journal, credencial, cache ou NetCDF foi incluído nos commits.
- [ ] Confirmar que SCC continua opcional.
- [ ] Confirmar que o processamento científico não depende de serviços SCC.
- [ ] Confirmar que as otimizações recentes do LIBIDS foram preservadas.
- [ ] Confirmar que o CI está integralmente aprovado.
- [ ] Decidir explicitamente a configuração SCC padrão em `main`.
- [ ] Fazer merge sem reescrever o histórico.
- [ ] Realizar testes rápidos de processamento L0/L1 e operações SCC somente de leitura após o merge.

**Gate final:** `main` contém o LIBIDS atualizado e a integração SCC simplificada, testada e independente dos produtos científicos.

---

## Estado geral

| Fase | Status |
|---|---|
| 0. Baseline e preparação | Parcialmente concluída |
| 1. Merge das mudanças de `main` | Integrada, CI 6/6; dois testes operacionais pendentes |
| 2. Simplificação do workflow SCC | Núcleo integrado e CI do PR aprovado; revisão de estados e comandos históricos pendente |
| 3. Remoção de código transitório | Concluída no código, CI 6/6; pacotes locais não inspecionados pela API |
| 4. CLI e configuração | Concluída no código, CI 6/6; default de `main` reservado para Gate 7 |
| 5. Validação e regressões | Otimizações LIBIDS/ERA5 implementadas e CI 6/6 (737 testes); benchmark científico real e SCC download pendentes |
| 6. Documentação | Pendente |
| 7. Merge final em `main` | Pendente |

### Próxima ação

**Extensão da Fase 5 (§5.5) implementada e validada no CI 6/6 (commit `77615712`):** cache reconstruível de cabeçalhos LIBIDS e seleção temporal; SCC-only reexport com fontes L0 unívocas; prefetch automático global ERA5 com horários esparsos, teto de 80.000 campos, requests sequenciais, cache horário verificado e política científica explícita. **Ainda faltam** benchmark no acervo de seis anos, equivalência científica L0/L1/L2 em amostras reais, download SCC de medição com ELPP/ELDA concluídos e testes operacionais restantes do Gate 1. Nenhuma operação POST/DELETE remota SCC foi executada pelo assistente. Depois verificar equivalência e desempenho em dados reais e fechar o Gate 5 com download SCC de um ID que conclua ELPP/ELDA. CI anterior 6/6 no código `a3c6c211`; continuam pendentes a comparação científica real e os testes operacionais da Fase 1. Nenhuma operação remota SCC executada nesta atualização; `main` permanece reservada ao Gate 7.

Não executar uploads, substituições ou DELETE reais sem autorização explícita, ID adequado e preflight remoto. L0/L1/L2 e arquivos históricos devem continuar preservados.
