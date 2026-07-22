# Klett–Fernald–Sasano two-sided — convenções da implementação 2

## Equação e unidades

Para altitude geométrica crescente `z` em metros, o sinal elástico corrigido
por alcance `X(z) = P(z) z²` é modelado diretamente por

```text
X(z) = C [βm(z) + βa(z)]
       exp{-2 ∫[z0,z] [Sm βm(s) + Sa(s) βa(s)] ds}.
```

`βm` e `βa` estão em m⁻¹ sr⁻¹, `Sm` e `Sa` em sr, e `C > 0` é uma constante
instrumental arbitrária. O forward model dos testes integra essa extinção
total por trapézios e não reutiliza o kernel nem a álgebra inversa.

Definindo as integrais orientadas

```text
M(z) = ∫[zr,z] [Sa(s) - Sm] βm(s) ds,
Y(z) = X(z) exp[-2 M(z)],
```

a solução usada no kernel é

```text
β(z) = Y(z) /
       {X(zr)/βr - 2 ∫[zr,z] Sa(s)Y(s) ds},
βa(z) = β(z) - βm(z).
```

`βr` é a retroespalhamento total `βm(zr) + βa(zr)` em um único bin `zr`.
No forward (`z > zr`) as duas integrais orientadas são positivas. No backward
(`z < zr`) elas são negativas; se o kernel acumula a integral reversa positiva
`∫[z,zr]`, o fator molecular equivalente é necessariamente

```text
exp{+2 ∫[z,zr] [Sa(s) - Sm] βm(s) ds}.
```

A implementação 1 acumulava `∫[z,zr] βm ds > 0`, mas aplicava um sinal
negativo no exponencial. Esse é o defeito corrigido na versão 2. O denominador
backward já possuía a orientação positiva correta. O ramo forward foi derivado
e testado separadamente e não precisou de mudança de sinal.

## Referência e junção

A busca Rayleigh continua escolhendo uma janela para calibração e QA. O centro
selecionado dessa janela é o `ref_idx` da inversão; a janela não é transformada
em vários bins de contorno. A condição física escalar é

```text
βr = SRr βm(zr)
   = [1 + aerosol_ref_fraction] βm(zr),
aerosol_ref_fraction = βa(zr)/βm(zr) = SRr - 1.
```

Backward e forward recebem exatamente o mesmo `βr`. O bin `zr` é escrito uma
vez como `βa(zr) = βr - βm(zr)`, não é reintegrado e recebe a flag de ramo 2.
Bins abaixo recebem ramo 1 e bins acima, ramo 3. Cada lado mantém uma flag de
validade própria; sinal inválido ou denominador não positivo interrompe somente
o lado afetado, preserva `NaN` nos bins não resolvidos e nunca copia resultados
do outro lado.

Com `allow_negative_aerosol=True`, valores negativos são preservados. Com
`False`, `βa` é limitado a zero; se a condição na referência implicar aerossol
negativo, a condição aplicada passa coerentemente a `βr = βm(zr)`. A validação
científica principal usa aerossol estritamente positivo com as duas políticas,
portanto clipping não mascara a correção.

## Lidar ratio e Monte Carlo

A API determinística aceita `Sa` escalar ou perfil com o mesmo shape da grade.
No caso variável, tanto `M` quanto o termo de denominador integram `Sa(s)` no
respectivo bin. Esse teste verifica formulação e shapes; não valida um prior
operacional variável.

O wrapper Monte Carlo preserva as perturbações existentes: ruído do RCS,
lidar ratio escalar e condição de contorno. Todas as realizações usam o kernel
two-sided corrigido e a seed permanece determinística. Os desvios retornados
são **dispersão Monte Carlo parcial**, não incerteza total; não incluem todo o
orçamento sistemático e não são usados para validar a equação determinística.

## Caso sintético, tolerâncias e convergência

Os casos independentes usam atmosfera padrão amostrada entre 0,3 e 12 km,
perfil molecular Bucholtz calculado separadamente em 355 e 532 nm, aerossol
`2,2e-6 exp[-(z-300 m)/2100 m] m⁻¹ sr⁻¹`, `Sm = 8π/3 sr`, `Sa = 55 sr`,
referência em 7,2 km, contorno total exato e sinal sem ruído. Cinco bins são
excluídos em cada borda externa (150 m na grade principal de 30 m); o bin de
referência não é excluído. O limite relativo de 0,35% cobre o pior ramo
forward de 355 nm, onde a verdade residual é pequena, e permanece ancorado na
convergência abaixo. O limite absoluto observado também é registrado.

| wavelength | dz | região | erro L2 relativo | erro absoluto máximo (m⁻¹ sr⁻¹) |
| --- | ---: | --- | ---: | ---: |
| 355 nm | 30 m | backward | 4,874e-4 | 1,005e-9 |
| 355 nm | 30 m | forward | 2,600e-3 | 1,606e-10 |
| 532 nm | 30 m | backward | 1,865e-5 | 5,233e-11 |
| 532 nm | 30 m | forward | 7,761e-6 | 3,796e-13 |
| 355 nm | 60 m | two-sided sem bordas | 1,993e-3 | 3,717e-9 |
| 355 nm | 30 m | two-sided sem bordas | 4,980e-4 | 1,005e-9 |
| 532 nm | 60 m | two-sided sem bordas | 7,037e-5 | 1,817e-10 |
| 532 nm | 30 m | two-sided sem bordas | 1,864e-5 | 5,233e-11 |

Ao reduzir `dz` de 60 para 30 m, o erro cai para aproximadamente 25% em ambos
os comprimentos de onda, como esperado para integração trapezoidal de segunda
ordem. Sob a fórmula antiga, o erro backward equivalente era 6,969 em 355 nm e
1,091 em 532 nm.

## Versionamento e reprocessamento

Os produtos registram método `Klett-Fernald-Sasano`, modo `two_sided`, método
de incerteza `Monte Carlo`, `fernald_implementation_version = 2` e mudança
`corrected_backward_molecular_factor_sign`. A mesma identidade entra
explicitamente no payload do fingerprint de Nível 2. Produtos ópticos de
implementações Fernald anteriores à versão 2 precisam ser reprocessados;
Níveis 0 e 1 não são invalidados por esta mudança.
