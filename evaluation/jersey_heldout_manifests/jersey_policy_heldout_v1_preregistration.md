# Pre-registrazione - policy jersey su held-out v1

Scritta il 2026-07-27, **prima** di eseguire l'adattatore e prima di osservare
qualunque risultato congiunto fra le due sorgenti.

## Blocco

```text
manifest: evaluation/jersey_heldout_manifests/jersey_policy_heldout_v1.json
manifest_sha256: b8e18d31c6f4f18cf11bf2e3e178ac9ba2314175732e3f01a08b3846def95e1c
sequenze: SNGS-089 091 092 093 094 095 096
tracklet con numero GT: 116
mai usate per training, checkpoint selection o tuning di soglie
```

Nessun altro blocco GSR pulito esiste: train e' esaurito, valid non ha altre
sequenze intatte, test e' congelato. Una domanda successiva non avra' un
held-out indipendente.

## Cosa e' gia' noto, e non viene da questo blocco

I marginali delle due sorgenti sono stati osservati prima di questa
dichiarazione, ma non aggiungono informazione: la superiorita' del CTC sul SAR
pretrained era gia' documentata sul GSR test (accuracy assigned 0.7306 contro
circa 0.24 del SAR sul frozen). Cio' che **non** e' stato osservato, e che
questo esperimento misura, e' il comportamento congiunto per tracklet: quali
astensioni il CTC copre, e con quale esito.

```text
                assigned   correct   accuracy all
SAR primario      78         18         0.155
CTC regione       87         71         0.612
```

## Bracci dichiarati

Due, entrambi decisi ora. La molteplicita' e' 2 e va riportata.

```text
A  fallback     sources=[jersey_ocr_primary, jersey_region_ctc]  on_abstain=fallback
B  ctc_primary  sources=[jersey_region_ctc, jersey_ocr_primary]  on_abstain=fallback
```

Il braccio A non puo' produrre `correct -> wrong`: e' impossibile per
costruzione, non per soglia. Se ne comparisse anche uno solo, e' un bug
dell'adattatore o delle superfici, non un risultato, e va indagato prima di
riportare qualunque numero.

Il braccio B puo' regredire tracklet gia' corretti. E' il motivo per cui esiste
il braccio A.

## Metriche riportate per entrambi

```text
coverage, accuracy assigned, accuracy all-track
transizioni paired complete: abstain->correct, abstain->wrong,
  correct->correct, correct->wrong, wrong->correct, wrong->wrong
bootstrap per sequenza, 10000 repliche, seed 20260726
```

Il bootstrap campiona sequenze intere, non tracklet. Con 7 sequenze gli
intervalli saranno larghi: un guadagno piccolo dara' "non concludente", non
"nessun effetto". Questo limite e' dichiarato in anticipo.

## Cosa questo esperimento non decide

La regola di promozione ad `apply` resta quella del progetto: zero
`correct -> wrong` e zero nuove emissioni errate. Il braccio A puo'
soddisfarla; il braccio B per sua natura no. Se B risultasse nettamente
migliore di A, la decisione se rilassare quel gate e' del progetto e non
discende da questi numeri: va presa esplicitamente e motivata.

Nessun tuning di soglie, ordinamenti o checkpoint su questo blocco, ne' prima
ne' dopo aver visto i risultati.

## Condizioni di esecuzione

```text
device: CPU
motivo: driver NVIDIA disallineato con il modulo kernel su `lie`
regola di decisione: ft.decision.jersey_policy.resolve_jersey_assignments
                     la stessa funzione che gira in pipeline, non una copia
```
