# Перепроверка литературы (CrossRef, 2026-08-07)

Проверены все ключи `\cite{}` из собранной статьи против `references.bib` и CrossRef `/works/{doi}`.

## Итог по цитируемым ключам

| Ключ | DOI CrossRef | Замечание |
|------|--------------|-----------|
| Yokoi2003, Meerlo1997, Palma2000, Bergmeir2012, Rampin1991, Grant2011, Kirschvink2000, Fang2010, Saevskiy2025, Cicerone2009, Conti2021, Geller1997 | OK, title/year совпадают | — |
| Vehtari2017 | OK (CrossRef year 2016 online / journal 2017) | нормально |
| Sanford2010 | OK DOI, год **2003** | ключ вводит в заблуждение; в bib год верный |
| Vyazovskiy2005 | DOI OK | статья про **theta в бодрствовании** как sleep propensity — **слабо** для REM-скоринга; в Данных cite заменён на Fang2010+Saevskiy2025; во Intro тоже |
| Hyndman2018, Sinkovec2022, GOST2017, HoffmanGelman2014 | нет DOI в bib | книга / дисс. / ГОСТ / JMLR URL — приемлемо |
| Rodkin2011 | в bib без doi; англ. версия `10.1134/S1069351311100107` проверена | добавить doi в bib |

## Что сделано при сборке

1. Cite в Данных (REM-тета): `Fang2010, Saevskiy2025` вместо `Vyazovskiy2005`.
2. Intro: спектральные признаки REM → `Fang2010, Saevskiy2025`.
3. Запись верификации: этот файл.

## Не трогали намеренно

- Неопубликованная ВКР / сетка 896 — не в статье.
- Ключ `Sanford2010` оставлен (совпадает с bib); год в списке литературы 2003.
