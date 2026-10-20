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

### 3.1. Ferramentas históricas

| Módulo | Destino pretendido |
|---|---|
| `scc/migrate.py` | Remover após conferir pacotes históricos |
| `scc/lr_repair.py` | Remover após resolver os nomes LR antigos |
| `scc/sounding_repair.py` | Remover após resolver os nomes de sounding antigos |
| `scc/sounding_backfill.py` | Avaliar remoção após finalizar os pacotes históricos |

- [ ] Inventariar pacotes no layout antigo `.cache/scc/pending/`.
- [ ] Distinguir pacotes nunca submetidos de pacotes com histórico remoto.
- [ ] Identificar quais pacotes ainda exigem correções de nome ou conteúdo.
- [ ] Migrar ou regenerar apenas os pacotes elegíveis, mantendo evidências de integridade.
- [ ] Preservar registros de operações incertas ainda não resolvidas.
- [ ] Remover os módulos históricos quando não houver dependências operacionais relevantes.
- [ ] Remover os comandos CLI exclusivos das ferramentas eliminadas.
- [ ] Remover testes exclusivamente associados a funcionalidades removidas.
- [ ] Preservar testes de segurança e integridade que continuem relevantes ao fluxo novo.

### 3.2. `level0/scc_staging.py`

- [ ] Remover lógica de descoberta e recuperação de layouts históricos depois da transição.
- [ ] Manter apenas os diretórios de entrada por sessão.
- [ ] Preservar criação atômica dos pacotes.
- [ ] Preservar prevenção de colisões de `Measurement_ID`.
- [ ] Preservar manifesto, hashes e identificação da sessão científica.
- [ ] Garantir que nenhuma etapa altere L0/L1/L2 como consequência de limpeza SCC.

### 3.3. `scc/sounding_finalize.py`

- [ ] Simplificar a finalização dos soundings para o layout atual por sessão.
- [ ] Preservar apenas soundings observados e rastreáveis.
- [ ] Preservar `missing_policy: skip` e `fail` enquanto necessários.
- [ ] Não alterar pacotes com histórico de submissão.
- [ ] Reduzir a dependência operacional entre LIPANCORA e preparação SCC.
- [ ] Manter falhas SCC isoladas do resultado científico do L1.

**Gate 3:** o fluxo comum não depende de funções de migração ou reparos históricos.

---

## 4. CLI e configuração

### 4.1. Interface operacional

Manter como interface principal:

```bash
milgrau-scc upload
milgrau-scc status
milgrau-scc download
milgrau-scc cleanup
```

- [ ] Preservar upload individual e em lote.
- [ ] Preservar `--dry-run` para validação local.
- [ ] Preservar `--replace-existing --confirm` para substituição explícita.
- [ ] Preservar consulta remota somente de leitura.
- [ ] Preservar download em lote.
- [ ] Consolidar a recuperação de operações incertas em uma interface clara e documentada.
- [ ] Remover opções exclusivas de migrações ou reparos eliminados.
- [ ] Revisar nomes de opções que ainda utilizam o termo `cache` para pacotes persistidos nas sessões.
- [ ] Manter preparação de sounding apenas se ainda for uma operação manual necessária.

### 4.2. `config.yaml`

- [ ] Remover `scc.upload.automatic`, atualmente sem efeito de upload.
- [ ] Revisar `scc.upload.retries`, garantindo que nunca permita retry inseguro de POST/DELETE.
- [ ] Remover a política `after_download` se a limpeza manual for adotada como única opção.
- [ ] Simplificar opções de limpeza e flags redundantes.
- [ ] Remover referências restantes ao staging global.
- [ ] Preservar `scc.enabled: true` na branch experimental.
- [ ] Decidir explicitamente o valor padrão de SCC antes do merge em `main`.

### 4.3. `scc/cleanup.py`

- [ ] Manter exclusão manual dos inputs SCC.
- [ ] Exigir identificação inequívoca do pacote e da sessão.
- [ ] Evitar exclusão de pacotes com operações remotas pendentes ou incertas.
- [ ] Não remover resultados SCC, produtos científicos ou figuras.
- [ ] Simplificar verificações duplicadas sem perder garantias de integridade.

**Gate 4:** quatro comandos operacionais principais, configuração reduzida e limpeza previsível.

---

## 5. Validação e regressões

### 5.1. Testes automatizados

- [ ] Executar Ruff.
- [ ] Executar pytest no Linux com Python 3.12.
- [ ] Executar pytest no Linux com Python 3.14.
- [ ] Executar pytest no Windows com Python 3.12.
- [ ] Executar pytest no Windows com Python 3.14.
- [ ] Executar os testes do Web Explorer.
- [ ] Confirmar 6/6 jobs aprovados no commit final.

### 5.2. Testes científicos

- [ ] Verificar continuidade e identidade das sessões.
- [ ] Verificar QA por sessão.
- [ ] Verificar associação dos dark currents.
- [ ] Verificar processamento incremental.
- [ ] Verificar processamento sem SCC.
- [ ] Verificar geração SCC sem modificar produtos científicos.
- [ ] Verificar `--scc-reexport`.
- [ ] Verificar resultados consistentes em amostras representativas antes e depois do merge.

### 5.3. Testes SCC

- [ ] Validar pacote novo com `--dry-run`.
- [ ] Simular envio de um ID ausente.
- [ ] Simular ID existente sem autorização de substituição.
- [ ] Simular substituição explicitamente confirmada sem backup.
- [ ] Simular DELETE incerto sem backup e posterior reconciliação.
- [ ] Simular POST incerto sem retry automático.
- [ ] Verificar lote contendo pacotes válidos e inválidos.
- [ ] Verificar status remoto sem arquivos locais.
- [ ] Verificar download sem journal local, com manifesto válido.
- [ ] Verificar gerações de resultados com proveniência preservada.
- [ ] Verificar que cleanup nunca exclui arquivos científicos ou resultados.

### 5.4. Validação real com SCC

- [ ] Executar `milgrau-scc upload --dry-run` em dados reais.
- [ ] Verificar IDs e configurações existentes no servidor.
- [ ] Confirmar o comportamento de upload em um único ID autorizado.
- [ ] Confirmar aceitação de sounding observado, quando aplicável.
- [ ] Confirmar download e integridade dos resultados.
- [ ] Realizar testes de substituição destrutiva somente com autorização explícita e ID de teste adequado.

**Gate 5:** testes automatizados aprovados, ausência de regressão científica conhecida e comportamento SCC real validado nos cenários autorizados.

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
| 3. Remoção de código transitório | Pendente |
| 4. CLI e configuração | Pendente |
| 5. Validação e regressões | Pendente |
| 6. Documentação | Pendente |
| 7. Merge final em `main` | Pendente |

### Próxima ação

**O núcleo da Fase 2 foi incorporado à branch SCC pelo PR #5.** Verificar os journals de operações incertas e pacotes locais da Fase 0 antes de eliminar caminhos históricos, consolidar comandos e revisar estados redundantes. Os dois testes operacionais restantes da Fase 1 também permanecem pendentes.

Não executar uploads, substituições ou DELETE reais sem autorização explícita, ID adequado e preflight remoto. L0/L1/L2 e arquivos históricos devem continuar preservados.
