# Conclusão da classificação

## Resultado geral

O projeto avaliou uma MLP própria em PyTorch para prever a variável `investmentPolicy`, com três classes: Conservadora, Moderada e Sofisticada. A métrica principal foi o F1 macro, escolhido porque a classe Sofisticada representa apenas 4,03% das observações.

No conjunto de teste, os resultados foram:

| Modelo | Acurácia | F1 macro |
|---|---:|---:|
| Classe majoritária | 0,528 | 0,230 |
| Regressão logística balanceada | 0,401 | 0,308 |
| MLP escolhida | 0,421 | 0,323 |

A MLP superou a regressão logística em F1 macro, mas a melhoria relativa foi de aproximadamente 4,7%. Como o critério definido antes do treinamento exigia uma melhoria mínima de 15%, o critério de sucesso não foi atingido. Para atingir o critério, a rede precisaria obter F1 macro de aproximadamente 0,355.

## Arquitetura e treinamento

A configuração escolhida foi a rede `16 → 128 → 64 → 3`, com 10.627 parâmetros. Foram testadas oito configurações, incluindo dropout, weight decay, diferentes larguras, profundidades, taxas de aprendizado e tamanhos de batch. O treinamento usou AdamW, pesos de classe e parada antecipada pela perda de validação. A melhor configuração foi selecionada exclusivamente pela validação.

O resultado variou entre as sementes:

| Semente | F1 macro |
|---:|---:|
| 42 | 0,323 |
| 43 | 0,292 |
| 44 | 0,332 |

A média foi aproximadamente **0,316 ± 0,021**. Essa variação é relevante porque a diferença entre a MLP e a regressão logística foi pequena. Portanto, não há evidência forte de uma vantagem estável da rede sobre o modelo linear.

## Análise dos erros

Na MLP, os resultados por classe foram:

| Classe | Precisão | Recall | F1 |
|---|---:|---:|---:|
| Conservadora | 0,498 | 0,748 | 0,598 |
| Moderada | 0,558 | 0,156 | 0,243 |
| Sofisticada | 0,076 | 0,386 | 0,127 |

O maior erro ocorreu na classe Moderada: 84,4% das observações dessa classe foram classificadas incorretamente. A classe Sofisticada também apresentou desempenho fraco, principalmente pela baixa precisão. Isso está relacionado ao forte desbalanceamento e à sobreposição entre os perfis de carteira.

Por `personType`, o erro foi de 57,0% para F e 74,6% para J. O grupo com `personType` ausente teve erro de 50%, mas continha somente quatro observações e não permite uma conclusão confiável.

## Limitações e uso responsável

O alvo representa uma política de investimento registrada e pode refletir regras de negócio que não estão totalmente presentes nas features observadas. A MLP não deve ser interpretada como uma decisão autônoma de recomendação financeira. Um erro pode direcionar um cliente para um perfil inadequado e afetar sua exposição a risco.

Outra limitação é o desbalanceamento: a classe Sofisticada possui poucos exemplos, o que dificulta estimar seu comportamento e torna o recall instável. Além disso, o split foi aleatório e não temporal; mudanças futuras nas políticas ou no comportamento dos clientes podem reduzir o desempenho.

Com mais tempo, seria importante testar validação cruzada estratificada, investigar novas variáveis não derivadas do alvo, calibrar as probabilidades, estudar limiares e avaliar modelos de custo sensível. A importância por permutação também deve ser incluída e comparada formalmente às hipóteses levantadas no EDA antes da entrega final.

## Situação da rubrica

- Etapas 0 e 1: atendidas.
- Etapa 2: atendida, com MLP própria, saída multiclasse, perda e parâmetros.
- Etapa 3: loop, AdamW e parada antecipada implementados; as curvas precisam estar apresentadas no notebook final.
- Etapa 4: oito configurações e duas técnicas de regularização comparadas.
- Etapa 5: teste, métricas e três sementes realizados.
- Etapa 6: análise por subgrupo realizada; falta incluir a importância por permutação no notebook.
- Etapa 7: conclusão preenchida neste documento, mas deve ser incorporada ao notebook/relatório final.
