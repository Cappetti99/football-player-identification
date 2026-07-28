# Handoff FT - aggiornato 2026-07-21

## Briefing iniziale

Prima di usare la cronologia di questo handoff, leggere:

```text
docs/project_brief_for_llm.md
```

Il briefing spiega finalita', domanda scientifica, pipeline, protocollo
anti-leakage, risultati, esperimenti respinti, stato validato e prossime
priorita'. Questo file conserva invece il dettaglio cronologico e operativo.

## Setup operativo per l'LLM

Quando leggi questo handoff, lavora come un esperto senior di:

- computer vision per video sportivi broadcast;
- multi-object tracking, tracklet linking e scene-cut handling;
- player ReID, metric learning e calibrazione similarity/margin;
- OCR di numeri di maglia, riconoscimento parziale e temporal aggregation;
- data analysis, ablation study, error analysis e validazione statistica;
- progettazione di pipeline conservative con evidenze auditabili.

Comportamento richiesto:

- partire dagli artifact e dal codice reale, non da ipotesi generiche;
- distinguere sempre detection, tracking, team, jersey evidence, identity
  inference e constraint;
- misurare il collo di bottiglia prima di modificare soglie o architettura;
- cambiare una sola variabile per ablation quando possibile;
- usare `unknown` invece di forzare un'identita' debole;
- richiedere zero falsi positivi per promuovere nuovi link identitari;
- separare modalita' `audit/propose` da modalita' `apply`;
- non usare assigned rows come unica metrica: controllare precisione manuale,
  duplicati, scene cut, provenance e corrispondenza fra delta e righe applicate;
- segnalare domain gap, leakage, pseudo-label circolari e assenza di ground
  truth;
- preferire modifiche FT-native, riproducibili e configurabili;
- preservare le modifiche locali esistenti e rispettare le regole operative
  sotto riportate.

Formato delle risposte:

- stile tecnico, asciutto e decisionale;
- dichiarare risultato, evidenza, rischio e prossimo test;
- fornire comandi remoti completi quando serve una run;
- non proporre una run pesante se un audit offline puo' prima invalidare
  l'ipotesi.

## Regole operative

- L'utente lavora sulla remota `/home/cappetti/FT`, che non ha git operativo.
- Il worktree locale di riferimento e': `/Users/lorenzocappetti/Downloads/Football-Tracking-main/FT-identity-evidence-v1`.
- Branch di pubblicazione: `main`.
- Non modificare questo `handoff.md` automaticamente: aggiornarlo solo quando l'utente lo chiede.
- Le run pesanti vanno lanciate sulla remota. I dati/video/modelli non sono disponibili localmente.

## Vincoli di efficienza e stile (Token Saving)

- Adottare uno stile telegrafico, asciutto e denso di informazioni.
- Evitare introduzioni, conclusioni, convenevoli (`Certamente!`, `Ecco la
  soluzione...`) e transizioni prolisse.
- Andare dritti al punto: usare elenchi puntati per i concetti e blocchi di
  codice commentati solo quando necessario.
- Per il codice, mostrare soltanto righe modificate o nuovi metodi, non file
  completi, salvo richiesta esplicita o necessita' di contesto.
- Ridurre aggettivi e avverbi non necessari.
- Preferire sintassi minimale: `Usa ByteTrack invece di DeepSORT per X motivi`
  invece di `Ti suggerisco di considerare l'implementazione di ByteTrack...`.

## Baseline e stato progetto

## Aggiornamento 2026-07-27 (2) - il vantaggio dell'ordinamento sopravvive al tracking

Seguito diretto della sezione successiva, che misurava la policy sulla
superficie offline. Qui la stessa domanda sulla superficie **pipeline**:
detection e tracking reali, `display_track_id` invece di `gt_track_id`. Il
default operativo resta invariato.

### Prerequisiti risolti

```text
GPU        driver NVIDIA 580.173.02 riallineato dopo reboot; torch.cuda True
video      7 clip generate da img1 con scripts/make_soccernet_videos.py
           (portato in FT dal repo legacy, sha256
            12f00b6c8960dfc141e9338d560060b3aa60eedb18dd07626d450ff97bb5ae28)
           750 frame, 25 fps, 1920x1080 ciascuna
frame      max_frames: 300 da configs/default.yaml: la pipeline elabora i
           primi 300 frame, non i 750. Applicato a entrambi i bracci;
           l'evaluator riceve --max-frames 300 per allineare la GT.
```

### Le prime due run erano cieche: mode propose

Le run iniziali con i profili congelati hanno prodotto `emitted_on_visible: 0`
su entrambi i bracci e su tutte le sequenze, con metriche **identiche fino
all'ultima cifra**. Causa: il profilo congelato tiene
`jersey_frame_selection.mode: propose`, che attiva `observational_propose` in
`ft/pipeline.py` e salta `_apply_jersey`. Il numero viene calcolato, la policy
lo sceglie, ma non raggiunge mai le righe: `tracklets.csv` non contiene jersey.

Il segnale che ha smascherato il problema e' l'identita' perfetta fra i due
bracci: con policy diverse e' impossibile, a meno che la policy non arrivi da
nessuna parte.

Correzione: `mode: audit`. Verificato in `ft/features/jersey_frame_selector.py`
che `self.mode` compare solo nella validazione e in una guardia per i modelli
YOLO, quindi **non** cambia quali frame vengono selezionati: la selezione resta
quella del profilo congelato.

```text
configs/gsr_jersey_pipeline_baseline_apply_v1.yaml
  sha256 dbae8964a96867c4ef93f695873ca597e1f196cc134421f1fd05d82314ccc33d
configs/gsr_jersey_pipeline_ctc_primary_apply_v1.yaml
  sha256 76c823fd07bde7e0be27bf760e98ea1ca9d118de4a1fadb7b46563d424f9690e
```

Le run cieche restano utili come prova che il disegno e' appaiato: `tracklet
rows` identiche fra i bracci su tutte e 7 le sequenze.

### Risultati, 7 sequenze, 300 frame, 114 tracklet GT visibili

Valutazione con `scripts/evaluate_ft_gsr.py`, matching per IoU 0.50, livello
tracklet. I 114 tracklet GT sono gli stessi nei due bracci.

```text
                gt visibili  emessi  corretti  coverage  acc.emessi  acc.all
baseline            114        63       19      0.5526     0.3016     0.1667
CTC primario        114        94       53      0.8246     0.5638     0.4649

emissioni su GT senza maglia visibile:  11 -> 14
```

Transizioni:

```text
abstain -> correct    20
wrong   -> correct    17
wrong   -> abstain     2
correct -> correct    16
correct -> wrong       3
abstain -> abstain    18
abstain -> wrong      13
wrong   -> wrong      25

corretti 19 -> 53, guadagno netto +34
40 discordanti, 37 a favore del CTC
McNemar / binomiale esatto: p = 1.947e-08
```

Per sequenza, corretti baseline -> CTC:

```text
SNGS-089   6 -> 11      SNGS-094   4 -> 9
SNGS-091   0 ->  1      SNGS-095   2 -> 9
SNGS-092   2 ->  9      SNGS-096   1 -> 5
SNGS-093   4 ->  9
```

Il CTC vince in 7 sequenze su 7. Bootstrap per sequenza, 10000 repliche,
seed 20260726:

```text
delta accuracy all   +0.2978   IC95 [+0.2082, +0.3740]   P(delta>0) = 1.0
```

SNGS-091 resta degenere: 0 corretti su 15 per il baseline, 1 per il CTC. In
quella sequenza l'audit CTC trova 9 regioni su 33 crop; il numero non si vede.

### Confronto fra le due superfici

```text
                        offline (GT)    pipeline (tracker)
corretti baseline          18/116           19/114
corretti CTC               72/116           53/114
delta accuracy all         +0.466           +0.298
correct -> wrong             2                3
wrong -> correct            46               17
```

Circa due terzi del guadagno sopravvive. Il recupero di errori cala molto
(46 -> 17) mentre le regressioni restano stabili (2 -> 3): il rumore del
tracking non rende il CTC piu' sbagliato, lo rende piu' silenzioso — crop
peggiori, meno regioni rilevate, meno occasioni di correggere. Degradazione
benigna.

Le due superfici non sono confrontabili numero a numero: unita' diverse
(`gt_track_id` contro `display_track_id`), finestra diversa (sequenze intere
contro 300 frame) e denominatore diverso. Vanno riportate separatamente.

Un dettaglio che conferma il meccanismo: le emissioni crescono di 31
(63 -> 94) ma i corretti di 34. Se il CTC coprisse soltanto dove il SAR tace i
due incrementi coinciderebbero; che i corretti crescano di piu' significa che
sta anche correggendo emissioni sbagliate del SAR.

Nota: il baseline sembra piu' accurato in pipeline che offline sulle emissioni
(0.302 contro 0.231). Non e' un miglioramento: emette su molte meno tracklet,
quindi parla solo dove e' facile.

### Criterio di promozione: invariato e ancora aperto

```text
correct -> wrong           3      richiesto 0
nuove emissioni errate    16      (13 su visibili + 3 su non visibili)
                                  richiesto 0
```

Il braccio B fallisce il gate su entrambe le superfici, come dichiarato in
anticipo nella pre-registrazione. Nessuna decisione presa: il default resta
`sources: [jersey_ocr_primary]`.

Il quadro completo per decidere: la regola cosi' com'e' scritta boccia un
cambiamento che triplica i tracklet corretti su due superfici indipendenti,
con p dell'ordine di 1e-08, al prezzo di tre tracklet peggiorati. Qualunque
riformulazione va dichiarata come avvenuta **dopo** aver visto questi numeri.

### Limiti

- 300 frame su 750 per sequenza.
- 7 sequenze, una sola esecuzione, nessun blocco pulito residuo in GSR.
- `roster_path: null`: filtro roster e identita' reale non girano.
- `_apply_jersey` con `mode: audit` riceve `allowed_roles=None`, quindi scrive
  su tutti i gruppi; offline i ruoli erano player e goalkeeper.

### Prossimo passo

Ispezionare le 3 regressioni `correct -> wrong` una per una. Sulla superficie
offline le 2 regressioni erano lo stesso errore (`27` letto `21`, il `7`
scambiato per `1`): se anche qui emerge un pattern di forma, diventa un
bersaglio per una calibrazione o un'astensione mirata invece che un costo
irriducibile.

## Aggiornamento 2026-07-27 - held-out policy jersey: l'ordine delle sorgenti vale +44 tracklet

Questa sezione riporta il primo esperimento sulla decision policy. Non modifica
il default operativo, che resta `sources: [jersey_ocr_primary]`. Il criterio di
promozione **non** e' stato cambiato: resta aperto e discusso in fondo.

### Held-out nuovo, dichiarato prima

GSR train e' esaurito: 57 sequenze, tutte consumate (37 OCR train, 9
development, 11 frozen). Il test e' congelato. Nello split valid restavano 7
sequenze che nessun esperimento aveva toccato.

```text
manifest: evaluation/jersey_heldout_manifests/jersey_policy_heldout_v1.json
manifest_sha256: b8e18d31c6f4f18cf11bf2e3e178ac9ba2314175732e3f01a08b3846def95e1c
sequenze: SNGS-089 091 092 093 094 095 096
tracklet con numero GT: 116
pre-registrazione: evaluation/jersey_heldout_manifests/jersey_policy_heldout_v1_preregistration.md
```

Le esclusioni sono **derivate**, non elencate a memoria: 9900 file scansionati,
ogni sequenza scartata registrata con il file che la squalifica. Dopo questo
blocco GSR non ha piu' superficie pulita: una domanda successiva richiedera' il
test congelato o un'altra fonte.

### Superficie e strumenti

Superficie offline, la stessa di frozen e test: box e track_id dalla ground
truth, chiave `(sequence, gt_track_id)`. Il tracking non entra.

```text
1. evaluation/gsr_jersey_ocr/run_eval.py            -> shared_surface (crop + SAR)
2. scripts/evaluate_jersey_number_region_ctc_ocr_run.py -> region_ctc
3. scripts/apply_jersey_policy_to_predictions.py    -> bracci A e B
4. scripts/aggregate_jersey_thesis_benchmark.py     -> transizioni e bootstrap
```

Il passo 3 non reimplementa la regola: costruisce evidenze dai due
`predictions.csv` e chiama `ft.decision.jersey_policy.resolve_jersey_assignments`,
la stessa funzione che gira in pipeline. Se la regola cambia, cambia in un solo
posto.

Divergenza deliberata dalla `shared_surface` del GSR test, che usava il SAR
**fine-tuned**: qui il baseline e' il SAR **pretrained**, cioe' il recognizer
operativo dichiarato. Misurare il fallback contro un baseline non spedito non
avrebbe risposto alla domanda.

Device CPU: driver NVIDIA disallineato con il modulo kernel su `lie`
(NVML "Driver/library version mismatch", serve reboot). Baseline e candidati
girano sullo stesso device e i numeri sono nuovi, non confrontati con i valori
GPU gia' pubblicati.

### Il baseline e' coerente con il record

```text
                        coverage   accuracy assigned
development (9 seq)      0.7548         0.1709
frozen (11 seq)          0.7168         0.2419
held-out (7 seq)         0.6724         0.2308
```

Il SAR pretrained si comporta come documentato. La superficie e' comparabile.
Anche il CTC lo e': region detection coverage 0.6075 contro 0.6104 sul test.

### Bracci dichiarati e risultati

```text
A  fallback     sources=[jersey_ocr_primary, jersey_region_ctc]
B  ctc_primary  sources=[jersey_region_ctc, jersey_ocr_primary]
```

```text
                   assigned  correct  coverage  acc.assigned  acc.all
baseline SAR          78       18      0.6724     0.2308      0.1552
A  fallback           95       28      0.8190     0.2947      0.2414
B  CTC primario       95       72      0.8190     0.7579      0.6207
```

Transizioni contro la baseline:

```text
                    A        B
abstain->correct    10       10
abstain->wrong       7        7
correct->correct    18       16
correct->wrong       0        2
wrong->correct       0       46
wrong->wrong        60       14
```

Il braccio A ha **zero** `correct->wrong`, come garantito per costruzione
quando la baseline e' anche la prima sorgente: nessuna sorgente successiva puo'
scavalcare cio' che la prima ha deciso.

Bootstrap per sequenza, 10000 repliche, seed 20260726:

```text
                    delta accuracy all     IC95                P(delta>0)
A  fallback              +0.0874      [0.0323, 0.1574]          0.9997
B  CTC primario          +0.4657      [0.4016, 0.5478]          1.0000
```

### Confronto diretto A contro B

E' la misura piu' pulita: stessa superficie, stessi crop, **stessa coverage**
(delta esattamente 0.0, entrambi assegnano gli stessi 95 tracklet). L'unica
variabile e' l'ordine di priorita'.

```text
delta accuracy all   +0.3784   IC95 [0.2750, 0.4959]   P(delta>0) = 1.0
transizioni: recovered_correct 46, correct_to_wrong 2,
             unchanged_correct 26, unchanged_wrong 15,
             wrong_to_wrong 6, both_abstain 21
48 discordanti, 46 a favore di B
test binomiale paired esatto: p = 8.363e-12
guadagno netto: +44 tracklet
```

Interpretazione: l'ordinamento conservativo cattura meno di un quinto del
guadagno disponibile. Il motivo e' che protegge una sorgente che, quando
decide, sbaglia il 77% delle volte: il braccio A lascia intatti 60 tracklet
errati del SAR, dei quali B ne corregge 46.

### Le due regressioni sono lo stesso errore

```text
SNGS-089 track 10: GT 27, SAR 27, CTC 21
SNGS-092 track  4: GT 27, SAR 27, CTC 21
```

Entrambe `27 -> 21`: una singola confusione sistematica di forma, il `7` letto
come `1`, della stessa famiglia del `9/5` gia' documentato. Bersaglio concreto
per una calibrazione o un'astensione mirata, non rumore irriducibile.

### Metrica da non riportare

`gt_top5_rate` vale 0 per entrambi i bracci policy: l'aggregatore la calcola da
`ocr_diagnostics.json`, che le directory fuse non hanno, e su sorgenti miste
sarebbe comunque mal definita perche' i top-5 del CTC vivono in un altro
artifact. Il `-0.59` contro la baseline **non e' una regressione**. Conferma:
nel confronto A contro B, dove entrambi valgono zero, il delta e' esattamente
0.0. Dichiararla non disponibile finche' l'adattatore non propaga i candidati
della sorgente vincente.

### Tensione aperta nel criterio di promozione

La regola dichiarata chiede zero `correct->wrong` **e** zero nuove emissioni
errate. Applicata a questi risultati:

```text
A  zero regressioni, ma 7 nuove emissioni errate   -> bocciato
B  44 tracklet netti in piu', 2 regressioni        -> bocciato
```

Il criterio boccia entrambi, incluso quello che quadruplica i tracklet
corretti. Un vincolo di zero nuove emissioni errate rende impossibile
qualunque aumento di coverage, salvo sorgenti perfette.

Non e' stata presa alcuna decisione: il default resta invariato. Una eventuale
riformulazione — per esempio bilancio netto con intervallo di confidenza invece
di conteggi assoluti — e' una scelta metodologica, e va dichiarata come
avvenuta **dopo** aver visto questi numeri.

### Cosa questi numeri non dimostrano

- Non dicono che il CTC sia buono in assoluto: il confronto e' contro un
  incumbent debole.
- La superficie e' offline, con box e track dalla ground truth: il rumore del
  tracking non c'e'.
- 7 sequenze, device CPU, una sola esecuzione.
- Nessuna run di pipeline con il profilo B e' mai stata eseguita.

### Profili

```text
configs/gsr_jersey_ctc_fallback_coverage_v1.yaml   braccio A
configs/gsr_jersey_ctc_primary_v1.yaml             braccio B
```

Entrambi non validati per `apply`. Nessuno dei due va eseguito sul GSR test.

### Prossimo passo previsto

Eseguire la pipeline reale con il profilo B sulle stesse 7 sequenze e
verificare se il guadagno sopravvive al rumore di detection e tracking.

La superficie e' diversa — `display_track_id` invece di `gt_track_id` — ma
l'evaluator esiste gia': `scripts/evaluate_ft_gsr.py` fa matching per IoU fra
`tracklets.csv` e le label GSR e produce coverage, accuratezza sulle emissioni
visibili, falsi su GT non visibile e calibrazione della confidenza. Le
definizioni non coincidono con quelle della superficie offline, quindi i due
esperimenti vanno riportati separatamente e non confrontati numero a numero.

Due run necessarie, identiche salvo la riga `sources`: il tracking e'
deterministico a parita' di input, quindi il confronto risulta appaiato sulle
stesse track.

## Aggiornamento 2026-07-26 - decision policy jersey, passo 2 equivalente

Questa sezione prevale sulla precedente per il percorso della decisione jersey.
Non modifica detection, tracking, OCR, soglie, checkpoint ne' i risultati
congelati. Il comportamento operativo e' identico a prima.

### Perche' il piano iniziale e' stato cambiato

Il passo 2 previsto era "portare `summarize` a leggere dalle evidenze". Il
rilievo del contratto reale di `HungarianPlayerIdentifier.summarize` lo ha
invalidato: legge sette famiglie di campi dalle righe, e le evidenze ne
coprono due e mezzo.

```text
jersey            coperta
team              coperta
ruolo             solo arbitro; portiere e PRTReID no
portiere          non coperta
gruppi semantici  non coperta
geometria/visuale non coperta (position_pitch, visual_embedding)
crop              non coperta (crop_quality, crop_path)
```

Le tre ultime sono per-frame: emetterle trasformerebbe l'artifact da tabella a
livello tracklet a tabella a livello frame, su 4901 righe. E non serve
all'obiettivo, che e' promuovere un recognizer con una riga di config.

Il passo 2 e' stato quindi ristretto al **solo canale jersey**, intercettando
nel punto in cui si sceglie quale numero diventa un assignment. Il resto della
pipeline continua a leggere dalle righe.

### Stratificazione

```text
produttori emettono claim grezzo
  -> decision policy sceglie fra le sorgenti
  -> filtro roster e filtro portiere vincolano il vincitore
  -> _apply_jersey
```

Il filtro e' un vincolo sulla decisione, non parte del produttore: un
recognizer promosso attraversa esattamente gli stessi gate del primario.

### Codice

```text
ft/decision/jersey_policy.py    resolve_jersey_assignments(evidence, policy)
ft/decision/__init__.py
ft/pipeline.py                  innesto policy + backfill estratto
ft/config.py                    blocco decision_policy nei DEFAULTS
ft/core/emitters.py             payload jersey lossless
tests/test_jersey_decision_policy.py     17 test
configs/gsr_jersey_ctc_fallback_coverage_v1.yaml
```

```text
ft/pipeline.py sha256
  cd8aaeeee6c15a6c2e37cc7c5b107daba7b8e9cfd15e7eda52d594563ed7f3f3
scripts/verify_evidence_additive.py sha256
  e323f966a686a5714d28d6d7c446f3035a49b4d3fd8efbef358a8d321ae2a22e
```

### Policy

```yaml
decision_policy:
  jersey_number:
    sources: [jersey_ocr_primary]
    on_abstain: fallback
```

`sources` e' ordinata: vince la prima sorgente che ha **deciso**. Con
`on_abstain: fallback` una sorgente successiva parla solo dove la precedente si
e' astenuta, quindi l'unica transizione possibile rispetto alla baseline e'
`unknown -> numero`; `correct -> wrong` e' impossibile per costruzione, non per
soglia. Il default a sorgente singola riproduce il comportamento storico.

Il payload CTC e' mappato esplicitamente (`recognized_frames -> votes`,
`top5 -> candidates`) e i campi senza equivalente a monte restano assenti
invece di essere inventati. Le evidenze a livello crop non sono mai promosse.

Limite dichiarato: il primario emette su `players:<id>` oppure
`players:<id>:<segmento>` quando `jersey_ocr.segment_frames > 0`; il CTC emette
sempre su `players:<id>`. Con i segmenti attivi i due soggetti non si
incontrano e una sorgente CTC non puo' vincere su un tracklet segmentato. E'
asserito da un test invece di essere risolto con una regola inventata.

### Bug trovato dal gate: aliasing diagnostics/assignments

`JerseyOCR` mette lo **stesso oggetto** dict in `assignments` e nei propri
diagnostics (`"voted": voted`). Il backfill in place di
`raw_jersey_distribution`, che apriva `normalize_and_filter_jersey_assignments`,
mutava quindi anche l'artifact esportato.

La policy costruisce dict nuovi: il backfill finiva su quelli e
`jersey_ocr.json` perdeva il campo su tutti e 13 i tracklet. Correzione: il
backfill e' stato estratto in `backfill_raw_jersey_distribution` ed eseguito
**prima** dell'emissione dell'evidenza, cosi' muta l'oggetto originale come
prima ed entra nel payload.

Verificato per lettura che `RosterAwareOCRFilter.apply` (`dict(assignment)`) e
`filter_goalkeeper_jersey_assignments` (ricostruisce `kept`) **non** mutano in
place: il backfill era l'unica mutazione della catena, quindi l'equivalenza non
dipende dal fatto che il filtro roster fosse disattivo nel profilo di gate.

### Gate esteso

Il confronto per SHA-256 puro non e' un criterio utilizzabile: la config
risolta e il suo hash cambiano per costruzione a ogni chiave aggiunta, e una
chiave nuova nei diagnostics e' additiva quanto un file nuovo. Tre bucket,
tutti riportati, nessuno silenzioso:

```text
changed_volatile_fields_only   timestamp, durate, artifacts_bytes, cache OCR
changed_config_surface_only    config e config_sha256    --allow-config-change
changed_added_keys_only        sole chiavi inserite      --allow-added-keys
changed                        tutto il resto: fallimento
```

Il criterio delle chiavi aggiunte e' esatto, non una whitelist di nomi: ogni
chiave del controllo deve essere presente nel candidato con lo stesso valore,
liste comprese; una chiave persa, un valore diverso o una lista di lunghezza
diversa restano `changed`. Entrambi i flag sono espliciti a riga di comando: il
pass dichiara cosa e' stato accettato.

### Risultato equivalenza, SNGS-025 250 frame, CPU

```text
FT jersey decision policy: sources=['jersey_ocr_primary'] subjects=13
                           assigned=13 per_source={'jersey_ocr_primary': 13}

changed                        []
changed_volatile_fields_only   run_diagnostics
changed_config_surface_only    run_manifest, source_provenance
changed_added_keys_only        jersey_ocr:/decision_policy
                               identity_assignments:/jersey_ocr/decision_policy
removed                        []
unexpected_new_files           []
passed                         true
```

Suite locale: 405 passed, 3 skipped, 1 failed (`jersey_multiframe_digits`,
richiede torch, by-design).

### Profilo di ablation, non validato

```text
configs/gsr_jersey_ctc_fallback_coverage_v1.yaml
sources: [jersey_ocr_primary, jersey_region_ctc]
on_abstain: fallback
```

Il primario tiene ogni tracklet che ha deciso; il CTC parla solo dove il
primario si e' astenuto. Non eseguito. La variante aggressiva (CTC che scavalca
il primario) e' una riga in `sources` e **non** e' spedita come profilo: puo'
regredire tracklet gia' decisi e richiede un benchmark paired suo.

### Vincoli

- Non usare il profilo di fallback sul GSR test.
- Il blocco di valutazione va scelto e dichiarato **prima** di guardare i
  numeri; development a 9 sequenze e frozen a 11 sono gia' stati aperti.
- Per l'ablation il gate additivita' non e' il criterio: serve confronto paired
  sui tracklet con la ground truth, con transizioni `correct->wrong` e
  `wrong->correct` riportate.
- Le altre famiglie di campi restano sulle righe; migrarle non e' richiesto da
  nessuna decisione in sospeso.

## Aggiornamento 2026-07-25 - layer evidenze, passo 1 additivo validato

Questa sezione prevale sulle note storiche per la struttura interna della
pipeline. Non modifica detection, tracking, OCR, soglie, checkpoint, config
congelate ne' il percorso operativo/apply.

### Motivazione

Oggi ogni stage muta strutture condivise (`tracks`, `rows`), e il decision
layer ri-aggrega dalle righe cio' che il modulo a monte sapeva gia':
`HungarianPlayerIdentifier.summarize(rows)` e `build_tracklet_evidence(rows)`
ricostruiscono entrambi il jersey che `_apply_jersey` aveva appena scritto. Da
quel giro derivano tre limiti: un solo recognizer per volta puo' scrivere,
confrontarne due richiede due run intere, e la modalita' `audit` va imposta a
mano da moduli che si trattengono dallo scrivere.

Il passo 1 introduce una tabella di evidenze immutabili come output
**aggiuntivo**. Nessuno la legge ancora.

### Codice aggiunto

```text
ft/core/evidence.py                     record Evidence, SubjectType, EvidenceKind
ft/core/evidence_store.py               store append-only, query, manifest, Parquet/JSONL
ft/core/emitters.py                     6 adattatori puri + ricostruzione inversa
scripts/verify_evidence_additive.py     gate control vs candidate
tests/test_evidence_core.py             schema, store, artifact
tests/test_evidence_emitters.py         adattatori, gate di sufficienza
tests/test_verify_evidence_additive.py  normalizzazione volatili
configs/gsr_jersey_region_ctc_sjn_to_gsr_audit_v1_cpu_gate.yaml
```

`ft/pipeline.py`: +68 righe, una sola riga rimossa (import sostituito). Nessuno
stage e' stato modificato.

### Semantica del record

```text
subject_id      sempre stringa, namespaced dal track group: players:12,
                players:12:0 (segmento jersey), referees:13
value=None      il produttore ha girato e si e' astenuto
riga assente    il produttore non ha girato
payload         chiavi legacy copiate verbatim, nessuna rinomina
produced_by     identificatore stabile: la policy del passo 2 seleziona da qui
```

Produttori: `jersey_ocr_primary`, `jersey_ocr_secondary_audit`,
`jersey_region_ctc` (livello track e livello crop), `team_color`,
`referee_palette`, `tracklet_linker`.

### Artifact runtime

```text
metadata/<video>_evidence.parquet   (fallback .jsonl se manca pyarrow)
metadata/<video>_evidence_manifest.json
```

### Protocollo del gate

Controllo e candidata devono usare la **stessa** artifacts dir, rinominata dopo
ogni run: dir diverse cambiano `crop_path` nei CSV e falsano il confronto. Il
controllo va eseguito con `ft/pipeline.py` pre-evidence
(sha256 `5b6dc14867c18034aaf81f0ced8c1bf0f5b46a56a88aead9dc7b899f85e7825f`); la
candidata con la versione evidence
(sha256 `9b17a2dc36c84ecad8b35270cd4541242588f9a7ca5c9cc5acc4a9e393827870`).
Verifica che nel log del controllo **non** compaia la riga `FT evidence:`.

Una run A/A con codice identico ha mostrato che quattro artifact differiscono
per conto proprio: timestamp, durate di stage, `artifacts_bytes` e contatori
della cache OCR. Il gate normalizza quei campi nei JSON prima di hashare, li
riporta in `changed_volatile_fields_only` e confronta byte per byte tutto il
resto.

### Risultato su SNGS-025, 250 frame

CUDA era indisponibile su `lie` (errore 804, forward compatibility su GeForce);
entrambe le run sono state eseguite su CPU con
`CUDA_VISIBLE_DEVICES=""` e il profilo `*_cpu_gate.yaml`. Il gate verifica
additivita', non qualita': il device non riapre la configurazione congelata.

```text
passed                          true
changed                         []
removed                         []
unexpected_new_files            []
changed_volatile_fields_only    run_manifest, run_diagnostics
added                           evidence.jsonl, evidence_manifest.json
control_files / candidate_files 52 / 54
tracklet rows                   4901 in entrambe
identity_mutations              0
config_hash                     a4f442d5ad61af87a98c7af52d6c7edcef1ea78093a81ed3e0fe0233ee6558ac
```

```text
evidenze: 140 righe, 4 kind, 5 produttori, 21 astensioni

jersey_number/jersey_ocr_primary   13
jersey_number/jersey_region_ctc    84   (18 track + 66 crop)
role/referee_palette               22
team/team_color                    20
link/tracklet_linker                1
```

`jersey_ocr_secondary_audit` non compare perche' disabilitato nel profilo: 5
produttori e' il numero corretto, non uno mancante.

### Due difetti corretti, entrambi trovati dai dati reali

1. `display_track_id` non e' unico fra track group: SNGS-025 ha un arbitro 13 e
   un giocatore 13. I subject id sono ora namespaced dal gruppo.
2. L'emettitore referee derivava `value="referee"` da `is_referee_palette`, che
   e' un ingrediente di `_is_player_referee_candidate`, non il suo risultato: su
   SNGS-025 etichettava 13 giocatori su 20 come arbitri. Ora il gruppo
   `players` si astiene sempre e il segnale grezzo resta nel payload. Regola
   generale: un adattatore riporta cio' che lo stage ha deciso, mai una
   decisione che lo stage non ha preso.

### Test di sufficienza

`test_evidence_reconstructs_jersey_rows_exactly` ricostruisce gli assignment
dalle evidenze, li passa a `_apply_jersey` e verifica righe e track identiche
alla via legacy. Passa: nessuna informazione jersey vive soltanto nelle righe
mutate. E' la precondizione del passo 2.

Suite locale: 379 passed, 3 skipped, 1 failed (`test_jersey_multiframe_digits`,
richiede torch, gia' dichiarato by-design in `requirements-local-test.txt`).

### Vincoli congelati

- Nessun componente legge le evidenze: il decision layer resta invariato.
- Gli stage continuano a mutare `tracks` e `rows` esattamente come prima.
- `*_cpu_gate.yaml` e' solo plumbing: non usarlo per produrre numeri di
  valutazione.
- Non aggiungere altri file agli artifact senza aggiornare `allowed_new` nel
  gate.

### Passo 2 ammesso

Portare `summarize` e `build_tracklet_evidence` a leggere dalle evidenze invece
che dalle righe, con test di equivalenza contro le versioni attuali e switch
`decision.input_source: rows | evidence` con default `rows`. Promozione solo
con output operativo invariato sullo stesso protocollo di gate.

### Nota operativa

Il disallineamento fra worktree locale e `/home/cappetti/FT` e' costato tre
cicli e ha invalidato una run per un backup mancante. Prima del passo 2 va
deciso se le modifiche nate sulla remota rientrano in locale o se la remota
riceve git.

## Aggiornamento 2026-07-20 - CTC SJN-210k nuovo candidato migliore

Questa sezione prevale sulle note storiche per detector della regione e
recognizer numerico. Non modifica ancora il percorso operativo/apply.

### Configurazione congelata

```text
frame selection: percorso frozen invariato
detector: runs/jersey_number_region/yolo26s_smoke_v1/weights/best.pt
detector SHA-256: ec3c4d06df75aff92f49779827e6cff96d99b84da496ced8a76ecfbb12c02ae4
detector confidence: 0.25
number-region padding: 0.25
recognizer: models/jersey_number_ctc_sjn_to_gsr_v1.pth
recognizer SHA-256: 11c6356895c94e734b795d9ca2403580187d1d726c5aada51273738042c7c1c5
aggregazione multi-frame: invariata
modalita': audit fino al GSR test ufficiale
```

### Dati e training

SJN-210k e' salvato sotto `/media/data-lie/cappetti/SJN-210k`. La derivazione
non duplica le immagini: i crop CTC vengono applicati in memoria usando box
normalizzati e padding 0.25.

```text
SJN sorgente:             14956 train, 6454 official test
regioni CTC localizzate:   8938 train, 3789 validation
CTC SJN best epoch:          16
CTC SJN validation acc.:  0.921087
CTC SJN SHA-256: 31a0b5590c107198159c3658cb11977de611c7386df19dfbd8c5595ed45147f4
fine-tuning GSR: 337 crop train, 92 crop/46 track validation
fine-tuning best epoch: 8
```

Lo split SJN chiamato ufficialmente `test` e' stato usato per validation e
checkpoint selection. Non presentarlo come test finale indipendente.

### Risultati development con detector automatico congelato

```text
                         baseline CTC   SJN->GSR
detection coverage       0.9783         0.9783
crop correct             64/92          77/92
track correct            34/46          41/46
track accuracy all       0.7391         0.8913
GT in top-5              42/46          46/46
```

### Risultati frozen: 11 sequenze, 173 track

```text
                         baseline CTC   SJN->GSR
region coverage          0.8162         0.8162
track coverage           0.9191         0.9191
correct                   84             124
wrong                     75              35
accuracy all             0.4855         0.7168
accuracy assigned        0.5283         0.7799
GT in top-5              124/173        146/173
```

Transizioni paired:

```text
correct->correct 79
correct->wrong    5
wrong->correct   45
wrong->wrong     44
net correct     +40
```

Sulle 50 track discordanti, il test binomiale esatto paired a due code produce
`p = 4.209852e-09`. Non modificare soglie o checkpoint dopo il frozen. Produrre
gli artifact tesi con `scripts/aggregate_jersey_thesis_benchmark.py`, quindi
eseguire una sola valutazione sul GSR test con configurazione congelata.

### Test ufficiale GSR completato 2026-07-21

La singola valutazione locked e' stata eseguita su 49 sequenze, 774 tracklet e
2954 crop selezionati. Baseline e candidato condividono gli stessi 1803 crop
con regione rilevata.

```text
                               baseline CTC   SJN->GSR
region detection coverage      0.6104         0.6104
track coverage                 0.7817         0.7817
correct                         249            442
wrong                           356            163
accuracy assigned              0.4116         0.7306
accuracy all                   0.3217         0.5711
GT in top-5                     386/774        521/774
```

Transizioni paired test:

```text
unchanged correct 236
recovered correct 206
correct->wrong     13
unchanged wrong    25
wrong->wrong      125
both abstain      169
net correct gain +193
```

Statistiche test:

```text
accuracy all delta bootstrap mean  +0.24924
accuracy all CI 95%                 [0.22222, 0.27764]
GT top-5 delta bootstrap mean       +0.17438
GT top-5 CI 95%                     [0.14800, 0.20217]
paired exact binomial p             7.537998e-46
bootstrap unit                      sequence
bootstrap samples                   10000
bootstrap seed                      20260720
```

Decisione: `models/jersey_number_ctc_sjn_to_gsr_v1.pth` e' il recognizer
numerico migliore e supera la baseline su development, frozen e GSR test. Non
eseguire altri tuning o nuove valutazioni sul test. La prossima modifica puo'
essere soltanto l'integrazione runtime della configurazione gia' congelata,
prima in audit con provenance completa.

### Integrazione runtime audit

Config congelata:

```text
configs/gsr_jersey_region_ctc_sjn_to_gsr_audit_v1.yaml
```

Il modulo `JerseyRegionCTCAuditor` continua a essere audit-only e non modifica
`jersey_assignments`, identity evidence o righe finali. La config disabilita
anche la preview di fusione (`fusion_preview_enabled: false`). Per ogni crop
esporta:

- path, SHA-256 e dimensione del crop sorgente;
- frame, crop quality, selection score/reason/rank;
- SHA-256 detector e recognizer;
- detector confidence e box regione normalizzato;
- dimensione della regione e padding;
- CTC top-1, log-probability e top-5.

Artifact runtime:

```text
metadata/<video>_jersey_region_ctc_audit.json
metadata/<video>_jersey_region_ctc_crops.csv
```

Requisiti di promozione tecnica: `identity_mutations=0`, output operativo
bit-identico rispetto al controllo senza audit, hash checkpoint verificati e
provenance presente per ogni crop rilevato.

## Regola documentazione tesi - aggiornata 2026-07-19

Il file LaTeX quantitativo di riferimento e':

```text
docs/thesis/jersey_experiments_results.tex
```

Ogni nuova run jersey validata deve aggiornare progressivamente questo file e
gli artifact sorgente, senza copiare numeri manualmente quando possono essere
rigenerati. Output target:

```text
evaluation_outputs/jersey_thesis_benchmark_v1/
  experiment_manifest.json
  per_track.csv
  per_sequence.csv
  aggregate.csv
  transitions.csv
  bootstrap.json
  tables/*.tex
  figures/*.png
```

Generatori implementati:

```text
scripts/aggregate_jersey_thesis_benchmark.py
scripts/plot_jersey_thesis_results.py
```

L'aggregatore deve ricevere soltanto run con la stessa superficie
`sequence + gt_track_id`; interrompe l'esecuzione se tracklet o ground truth
non coincidono. Registra SHA-256 degli input, produce transizioni paired e usa
bootstrap deterministico per sequenza. Il generatore grafico legge unicamente
i CSV aggregati: non deve ricalcolare metriche con definizioni differenti.

Tabelle obbligatorie:

- dataset/split, tracklet, frame/crop, jersey distribution ed esclusioni;
- classificatori frame: backbone, dati, accuracy, precision, recall, AP,
  top-k hit e runtime;
- OCR track-level: assigned, coverage, correct, wrong, accuracy assigned/all,
  GT nei top-5, correct-to-wrong e nuove emissioni errate;
- risultati per sequenza;
- runtime, memoria GPU e dimensione checkpoint;
- statistica sequence-level con media, deviazione, mediana, IQR e bootstrap
  confidence interval 95%.

Figure obbligatorie:

- accuracy/coverage e scatter coverage-accuracy;
- boxplot per sequenza;
- distribuzione score readable/unreadable e precision-recall;
- top-k curve;
- transition matrix;
- confusion matrix jersey e bias dei numeri predetti;
- diagramma pipeline ed esempi qualitativi.

Il bootstrap deve campionare sequenze intere, non crop o frame. Development,
frozen e test devono restare separati e dichiarati in ogni tabella. Celle senza
artifact verificato devono restare `DA COMPLETARE`: non inventare o inferire
valori. Dopo ogni benchmark aggiornare prima CSV/JSON, poi tabelle/figure e
infine il LaTeX.

## Aggiornamento 2026-07-18 - clean-back, OCR fine-tuned e audit secondario

Questa sezione prevale sulle note storiche successive per tutto cio' che
riguarda selezione frame jersey, OCR fine-tuned e PRTReID tracking.

### Decisione operativa corrente

```text
tracking operativo       ByteTrack
frame selector operativo ResNet34 legibility legacy
OCR operativo            SAR pretrained
YOLO clean-back          audit/propose soltanto
SAR fine-tuned            recognizer secondario audit/propose soltanto
MIL/PU utility            non usare nel runtime
PRTReID shadow tracker    respinto e rimosso dal percorso candidato
```

Regole congelate:

- nessuna proposta del recognizer secondario puo' modificare jersey, righe
  finali, identity evidence o assegnazioni;
- non ritoccare le soglie usando le 11 sequenze frozen gia' aperte;
- mantenere pretrained SAR come recognizer decisionale;
- mantenere YOLO clean-back e SAR fine-tuned disabilitati di default;
- non riattivare MIL, PU o appearance association senza una nuova ipotesi e un
  nuovo benchmark indipendente;
- GSR `test` resta congelato: non e' stato usato per questi esperimenti.

### Selettore YOLO clean-back

Target manuale: classificare il crop giocatore come `clean_back` oppure
`not_clean`. Non rileva direttamente il numero e non e' un detector bbox.

Dataset Ultralytics classification sequence-disjoint:

```text
path remoto:
  evaluation_outputs/jersey_back_yolo/datasets/gsr_train_manual_v1
train:      484 immagini, 37 sequenze, 210 clean_back, 274 not_clean
validation: 114 immagini,  9 sequenze,  52 clean_back,  62 not_clean
frozen validation osservata durante training: nessuna
review SHA-256:
  5f06b90f8149bfa24af9b8eb1af2517fda9494b33efe93616b55b44e4bb1f906
```

Training YOLO11s classification:

```text
base: yolo11s-cls.pt
best epoch: 2
validation top1: 0.89474
validation ranking AP: 0.9320608132
top-5 clean-back hit: 38/38 = 1.0
best checkpoint SHA-256:
  844803a2f06358065bddb2e05b66e2ea316f4632562143988a20b95f1f4ebbf8
checkpoint FT:
  models/jersey_back_yolo11s_cls_v1.pt
```

Conclusione end-to-end: il ranking `clean_back` e' apprendibile, ma nelle
ablation OCR non ha superato il ResNet34 legacy. Il crop torso fisso ha
aumentato le letture errate. Decisione:

- tenere `jersey_back_yolo11s_cls` solo in `audit/propose`;
- non usare il crop torso come input operativo OCR;
- non sostituire il ResNet34 legacy;
- metrica primaria futura: a parita' di top-k e OCR, presenza del numero GT
  corretto nei top-5 senza nuove decisioni errate.

Config sperimentali:

```text
configs/gsr_jersey_back_yolo11s_propose_full_crop_v1.yaml
configs/gsr_jersey_back_yolo11s_propose_v1.yaml
```

### Esperimenti MIL e PU: chiusi

MIL weakly supervised:

```text
manifest GSR train: 760 train tracklet, 173 internal validation
checkpoint: models/jersey_frame_utility_mil_resnet18_v1.pth
```

PU crop utility:

```text
pseudo-label full train: 15200 crop
positive: 729
hard negative: 3208
ignore: 11263
checkpoint: models/jersey_crop_utility_pu_resnet18_frozen_v1.pth
SHA-256: 4a75df91b57fe2a57cb0015ea23e965801de5965b79b339f8b9d26f73aaf2247
```

Il PU migliorava il ranking delle pseudo-positive rispetto a crop quality:

```text
utility AP: 0.5096 vs crop-quality AP: 0.1496
top5 hit:   0.8873 vs 0.5915
```

Questi target erano pero' indiretti/circolari rispetto all'OCR e meno
interpretabili delle annotazioni manuali clean-back. I percorsi runtime MIL/PU
sono stati rimossi. Conservare i risultati solo come evidenza storica; non
riaggiungerli al selettore.

### PRTReID detection audit e shadow association: respinti

Il refactoring paper-inspired ha separato detection, enrichment e tracking.
PRTReID detection-level e' stato verificato in ambiente dedicato:

```text
env: ft-prtreid-audit
torch: 1.13.1+cu117
torchvision: 0.14.1+cu117
numpy: 1.26.4
embedding coverage: 1.0
embedding dimension: 256
embedding norm medio: circa 1.0
```

La comparazione same-environment ha confermato che l'audit enrichment non
modifica gli output ByteTrack:

```text
detections.csv SHA-256 identico
tracklets.csv  SHA-256 identico
```

Lo shadow tracker appearance-based ha invece fallito sul controlled benchmark
SNGS-025, 250 frame:

```text
ByteTrack id switches: 45
shadow id switches: 309
ByteTrack fragmented GT rate: 0.4286
shadow fragmented GT rate: 0.9524
ByteTrack weighted purity: 0.9312
shadow weighted purity: 0.8740
```

Decisione: ByteTrack resta il tracker; non chiamare il prototipo StrongSORT e
non promuovere appearance association. PRTReID resta soltanto contesto
storico/audit opt-in.

### Dataset OCR manuale e fine-tuning SAR

Audit manuale costruito esclusivamente dalle 46 sequenze GSR train:

```text
review: 800 crop
readable: 583
unreadable: 195
uncertain: 22
```

Per i crop marcati leggibili, la trascrizione deriva dal jersey GT della track;
provenance obbligatoria: `manual_readability_plus_track_gt`. Split realmente
sequence-disjoint:

```text
train:      453 immagini, 37 sequenze
validation: 130 immagini,  9 sequenze
frozen:      11 sequenze non usate da training/checkpoint selection
```

Trainer e dataset:

```text
scripts/build_jersey_ocr_finetune_audit.py
scripts/build_mmocr_jersey_recognition_dataset.py
scripts/train_mmocr_jersey_recognizer.py
```

Training MMOCR 1.0.1 SAR:

```text
best epoch: 6
internal validation word accuracy: 0.8846
early stopping: epoch 11
checkpoint:
  models/jersey_ocr_sar_gsr_readable_v1.pth
config risolta:
  models/jersey_ocr_sar_gsr_readable_v1.py
checkpoint SHA-256:
  e93894fa213be4044bb27617452de28f09925a898726c8fcba8220c27d6d5843
size: circa 220 MB
```

L'ambiente operativo OCR e' `jersey-yolo-ocr`. Non installare pytest in questo
ambiente solo per i test: usare `jersey-legibility` per pytest e
`jersey-yolo-ocr` per MMOCR/CUDA e smoke reali.

### Ablation recognizer interna

Stesse 9 sequenze usate per checkpoint selection; risultato diagnostico, non
unbiased:

```text
                       pretrained    fine-tuned
tracklet                    155          155
assegnate                   117          136
corrette                     20           66
errate                       97           70
coverage                  0.7548       0.8774
accuracy assigned         0.1709       0.4853
correct candidate rate    0.5290       0.9355
```

Il fine-tuned diretto introduceva 2 `correct -> wrong` e 13 nuove emissioni
errate. Una fusione offline e' stata definita usando soltanto questa validation:

```text
usa fine-tuned se votes >= 4 e winner_margin >= 0.08
altrimenti fallback al pretrained
```

Sulla validation interna la regola produceva:

```text
correct: 20 -> 42
wrong:   97 -> 76
assigned: 117 -> 118
correct -> wrong: 0
nuove emissioni errate: 0
```

Le soglie `4/0.08` sono congelate e non possono essere modificate usando il
blocco frozen successivo.

### Validation frozen recognizer

Sequenze mai usate per training o scelta checkpoint:

```text
SNGS-067 SNGS-068 SNGS-071 SNGS-077
SNGS-097 SNGS-098 SNGS-099 SNGS-105
SNGS-154 SNGS-165 SNGS-170
```

Confronto pretrained vs fusione congelata:

```text
                       pretrained    fusion
tracklet                    173        173
assegnate                   124        128
corrette                     30         52
errate                       94         76
coverage                  0.7168     0.7399
accuracy assigned         0.2419     0.4063
correct candidate rate    0.5607     0.6474

recovered correct: 24
correct -> wrong: 2
new wrong emission: 2
```

Regressioni frozen:

```text
SNGS-097 track 15: GT 16, pretrained 16, fusion 10
SNGS-097 track 17: GT 20, pretrained 20, fusion 24
SNGS-097 track 4:  GT 6, pretrained unknown, fusion 10
SNGS-154 track 15: GT 6, pretrained unknown, fusion 10
```

Verdetto: miglioramento forte ma gate zero-regression fallito. Non promuovere
la fusione o il fine-tuned in `apply`. Il bias residuo piu' evidente e' verso
`10`.

### Recognizer secondario audit-only integrato

Nuovo blocco configurazione:

```yaml
jersey_ocr_secondary_audit:
  enabled: false
  mode: audit
  backend: mmocr_rec
  mmocr_rec: SAR
  mmocr_rec_weights: null
  mmocr_rec_weights_sha256: null
  min_votes: 4
  min_winner_margin: 0.08
```

Profilo congelato:

```text
configs/gsr_jersey_ocr_finetuned_secondary_audit_v1.yaml
```

Proprieta' implementate:

- supporta esclusivamente `audit/propose`; `apply` e' rifiutato;
- verifica SHA-256 del checkpoint;
- salva proposte, voti, margine, candidati e confronto col recognizer primario;
- ogni proposta contiene `applied: false`;
- dichiara e verifica `identity_mutations: 0`;
- non passa assegnazioni secondarie a `_apply_jersey`, identity evidence,
  Hungarian o propagation;
- feature disabilitata: nessun nuovo file/campo negli artifact legacy.

Artifact:

```text
{video_id}_jersey_ocr_secondary_audit.json
```

Smoke end-to-end SNGS-025, 250 frame:

```text
pipeline status: ok
tracklet rows: 4901
secondary proposals: 6
agreement: 4
disagreement: 1 (track 20: primary 2, secondary 5)
primary only: 8
secondary only: 1 (track 9: secondary 10)
identity mutations: 0
accidental final mutations: 0
```

File principali modificati/aggiunti:

```text
ft/features/jersey_ocr.py
ft/pipeline.py
ft/config.py
ft/validation.py
configs/default.yaml
configs/gsr_jersey_ocr_finetuned_secondary_audit_v1.yaml
evaluation/gsr_jersey_ocr/run_eval.py
tests/test_jersey_ocr_secondary_audit.py
```

### Prossimo esperimento ammesso

Obiettivo: verificare se `YOLO clean-back -> top-5 -> SAR fine-tuned audit`
aumenta la presenza del numero GT corretto rispetto a
`ResNet34 legacy -> top-5 -> SAR pretrained`, mantenendo invariati tracking,
numero crop e aggregazione.

Vincoli:

- prima solo offline/audit sulle sequenze gia' consumate;
- nessuna modifica a soglie sulla validation frozen gia' aperta;
- per una nuova promozione serve un nuovo held-out indipendente;
- nessun numero fine-tuned applicato alle righe;
- esportare provenance completa del frame/crop e sorgente recognizer;
- non usare GSR `test` finche' non esiste una candidata congelata che supera
  zero `correct -> wrong` e zero nuove emissioni errate.

## Aggiornamento operativo 2026-07-09

Stato corrente da usare come riferimento prima di proporre nuove run.

## Aggiornamento 2026-07-16 - paper jersey: subject filter respinto, prefix conservativo validato

### Obiettivo e implementazione

Sono stati trasferiti in FT due contributi di
`A General Framework for Jersey Number Recognition in Sports Video`:

1. filtro del soggetto principale basato su embedding PRTReID;
2. consolidamento conservativo dei conflitti una-cifra/due-cifre.

Entrambi sono disabilitati di default e supportano solo `audit/propose`.
`propose` non muta righe, track o identity. Tracking, detector, ResNet34 di
leggibilita', backend OCR, roster e profilo best-known Int-Ata non sono stati
modificati.

File principali:

```text
ft/features/jersey_subject_filter.py
ft/features/jersey_prefix_consolidation.py
tests/test_jersey_subject_filter.py
tests/test_jersey_prefix_consolidation.py
scripts/compare_jersey_decision_runs.py
configs/gsr_prefix_consolidation_votes3_propose_v1.yaml
```

Artifact aggiunti:

```text
*_jersey_subject_filter_scores.csv/json
*_jersey_subject_filter_tracks.csv/json
*_jersey_prefix_counterfactuals.csv/json
```

Metodo e pesi upstream sono CC BY-NC 3.0, ricerca/non commerciale. I pesi
restano esterni alla repository.

Checkpoint PRTReID remoto verificato:

```text
path: /home/cappetti/FT/models/reid/prtreid-soccernet-baseline.pth.tar
sha256: 1304562c4c930a4a54bbf2d44e221eb911fe31408f6875c5058b2d6ed621cf3b
size: 378 MB
embedding: globl, 256 dimensioni
```

### Filtro main-subject PRTReID

Algoritmo implementato per ogni `display_track_id`:

```text
player crop con quality >= 0.08
centroide embedding
keep se distance <= mean + 3.5 * std
3 round
min_samples=3
fallback se restano meno di 2 crop
```

La formula corregge il confronto errato del codice upstream. Il filtro non
abilita PRTReID linking, team clustering o identity bridge.

Audit completo SNGS-023, 750 frame:

```text
scored crops: 14398
embedding calcolati: 14237
failed: 0
outlier: 125 = 0.88%
display track interessate: 17 / 35
fallback: 0
round esclusione: 73 round1, 39 round2, 13 round3
```

Ablation SNGS-023:

```text
baseline:     15 emesse, 7 corrette, 8 errate
subject only: 15 emesse, 7 corrette, 8 errate
```

Held-out iniziale SNGS-039/054/078/085:

```text
baseline:     56 emesse, 23 corrette, 33 errate
subject only: 55 emesse, 23 corrette, 32 errate
accuracy all-track invariata: 0.184
frame accuracy all-visible invariata: 0.22964
```

Interpretazione: gli outlier PRTReID quasi mai coincidono con i top-5 crop gia'
scelti dal ResNet34. Il filtro rimuove una emissione errata, ma non aumenta
correttezza track/frame e aggiunge costo GPU rilevante. Decisione congelata:

- escludere PRTReID subject filtering dalla candidata principale;
- lasciare `jersey_subject_filter.enabled: false`;
- conservare codice e artifact per audit/riproducibilita';
- non eseguire altre ablation PRTReID subject senza una nuova ipotesi che
  giustifichi il costo.

### Prefix-aware: sviluppo e correzioni

Policy:

- candidato corto di una cifra;
- candidato lungo di due cifre contenente il corto in prima o seconda
  posizione;
- supporto da crop indipendenti;
- nessun numero inventato e nessun trasferimento di voti;
- si riduce solo il peso del corto;
- winner estraneo al conflitto produce `unknown`;
- parita' fra lunghi compatibili o margin insufficiente produce `unknown`.

La griglia audit iniziale era `[1.0, 0.75, 0.50, 0.25]`. Solo `0.25` cambiava
decisioni su SNGS-023:

```text
display 18: 2 -> 23, GT 23, correzione
display 12: 1 -> candidato estraneo 5, poi bloccato dal nuovo compatibility gate
```

Il primo held-out SNGS-039/054/078/085 con `min_long_votes=2` migliorava le
metriche aggregate ma introduceva due regressioni reali:

```text
4 -> 14 con GT 4
8 -> 18 con GT 8
```

L'error analysis mostrava che tutte le regressioni avevano solo 2 voti per il
lungo. I recuperi piu' affidabili `1 -> 17` e `1 -> 15` ne avevano 3. Da qui e'
stata progettata la candidata finale:

```yaml
selected_short_vote_weight: 0.25
min_long_votes: 3
mode: propose
```

Il primo held-out e' quindi design data per `min_long_votes=3` e non va contato
come validazione finale della candidata.

### Validazione finale prefix, parametri congelati

Profilo riproducibile:

```text
configs/gsr_prefix_consolidation_votes3_propose_v1.yaml
subject filter: disabled
prefix mode: propose
short weight: 0.25
min long votes: 3 crop indipendenti
```

Blocco validation 1, mai usato per tuning OCR della policy finale:

```text
SNGS-025, SNGS-036, SNGS-038, SNGS-041
96 display track valutabili

baseline: 54 emesse, 33 corrette, 21 errate
candidate: 54 emesse, 35 corrette, 19 errate
delta: 5 -> 15 corretto; 3 -> 34 corretto
regressioni corretto->errato: 0
```

Blocco validation 2 espanso, selezione deterministica congelata:

```text
SNGS-021, SNGS-022, SNGS-024, SNGS-026,
SNGS-027, SNGS-028, SNGS-029, SNGS-030
193 display track valutabili

baseline: 100 emesse, 60 corrette, 40 errate
candidate: 100 emesse, 64 corrette, 36 errate
regressioni corretto->errato: 0
nuove emissioni: 0
astensioni: 0
4 correzioni; 2 cambi errato->errato
```

Risultato pooled sui 12 video di validazione finale:

```text
display track valutabili: 289
emissioni: 154 -> 154
coverage: 53.29% -> 53.29%
corrette: 93 -> 99
errate: 61 -> 55
accuracy emitted: 60.39% -> 64.29%
accuracy all-track: 32.18% -> 34.26%

visible frame: 86845
frame corretti: 39340 -> 40820, delta +1480
frame accuracy all-visible: 45.30% -> 47.00%

regressioni corretto->errato: 0
nuove emissioni: 0
role contamination: 0
false no-jersey GT: 11 track / 1041 frame, invariati dalla baseline
```

La candidata supera la validation: migliora accuracy track/frame senza ridurre
coverage e senza regressioni osservate. L'effetto e' conservativo e raro, come
desiderato.

### Decisione congelata jersey paper

- Mantenere `jersey_prefix_consolidation` disabilitato nel default generale.
- Profilo sperimentale consigliato solo in `propose`:
  `configs/gsr_prefix_consolidation_votes3_propose_v1.yaml`.
- Non riaprire sweep su peso corto o `min_long_votes` usando le 12 sequenze di
  validazione finale.
- Non promuovere ancora ad `apply`: restano emissioni baseline su identita'
  senza jersey GT e `apply` muterebbe righe/tracce.
- Prossimo e unico benchmark per questa policy: split GSR `test` congelato,
  nessun tuning successivo.
- Nel test confrontare baseline e prefix-only; non includere PRTReID subject.
- Criteri: zero regressioni corretto->errato, zero nuove emissioni errate, zero
  nuovi falsi no-jersey, coverage non inferiore, frame accuracy all-visible non
  inferiore, provenance completa.
- Qualunque failure sul test lascia la feature `propose`-only; non correggere
  soglie sul test.

## Aggiornamento 2026-07-14 - TVCalib valutata e scartata dal percorso operativo

### Obiettivo

Valutare se coordinate metriche sul campo potessero migliorare il linker
`raw_track_id -> display_track_id` della baseline GSR `confidence=0.12`.
TVCalib e' stata eseguita in ambiente separato `tvcalib-ft`, senza modificare
l'ambiente `mmocr`, su tutte le 12 sequenze del pilot validation congelato.

Runner e strumenti riproducibili aggiunti:

```text
scripts/run_tvcalib_gsr.py
scripts/reproject_tracklets_tvcalib.py
scripts/audit_pitch_linking.py
scripts/run_gsr_pitch_link_audit.py
scripts/aggregate_gsr_pitch_link_audit.py
scripts/audit_gsr_link_candidate_ranking.py
tests/test_tvcalib_reprojection.py
tests/test_pitch_linking_audit.py
tests/test_link_candidate_ranking_audit.py
```

Checkpoint esterno, non versionato:

```text
/home/cappetti/tvcalib/data/segment_localization/train_59.pt
30 frame campionati per sequenza: 1, 26, ..., 726
seed: 10
optim_steps: 1000
```

### Qualita' geometrica

TVCalib funziona tecnicamente come calibrazione:

```text
12 sequenze, 9000 frame
errore pitch mediano medio per sequenza: 1.1018 m
errore pitch P90 medio per sequenza:    2.7091 m
coppie linker con pitch utilizzabile:   1162 / 1230 = 94.5%
```

Confronto con la calibrazione automatica legacy `auto:field_quad`:

```text
auto field quad:
  mediana media: 35.78 m
  P90 medio:     58.28 m

TVCalib:
  mediana media: 1.10 m
  P90 medio:     2.71 m
```

Conclusione geometrica: TVCalib e' accurata; il fallimento seguente riguarda
il suo valore operativo nel linker corrente, non la stima dell'omografia.

### Esperimento 1 - gate rigido velocita'

Audit iniziale a `12 m/s`:

```text
link correnti bloccati: 5
wrong offline bloccati: 4
correct offline bloccati: 1
```

Lo sweep offline individuava `14 m/s` come soglia apparentemente perfetta sul
pilot: 4 link wrong bloccati e 0 correct. L'ablation end-to-end ha invece
peggiorato il tracking:

```text
config: configs/gsr_detection_tracking_conf012_tvcalib_pitch_apply_v1.yaml
run: gsr_valid_pilot12_conf012_tvcalib_pitch14_apply_v1

delta paired medio vs confidence=0.12:
  HOTA:  -0.000763
  AssA:  -0.001203
  IDF1:  -0.001749
  switch improvement: -0.0336 / 1000 GT, quindi peggioramento
  frammentazioni: 0
sequenze: 1 migliorata, 8 pari, 3 peggiorate su HOTA/AssA
```

Il gate rigido e' respinto. Una coppia raw con GT maggioritari differenti non
garantisce che separarla migliori l'allineamento globale della traiettoria.

### Esperimento 2 - candidate ranking e costo pitch morbido

E' stato esportato il dataset dei candidati reali del linker, con GT aggiunta
solo offline:

```text
candidate rows: 78
target valutabili: 50
candidato corretto disponibile: 28 / 50 = 56%
baseline Recall@1: 25 / 50 = 50%
baseline Recall@3: 54%
baseline Recall@5 / oracle: 56%
MRR baseline: 0.524
```

Il collo di bottiglia principale e' candidate generation: in 22 target su 50
il candidato corretto non esiste, quindi nessun reranker puo' recuperarli.

La penalita' sulla velocita' non cambiava alcun ranking. Un costo morbido sulla
distanza pitch con peso minimo efficace `0.10` correggeva offline un caso
(`SNGS-038`, target raw 18) senza errori etichettati:

```text
soft_score = baseline_score / 160 + 0.10 * pitch_distance_m / 10
Recall@1 offline: 0.50 -> 0.52
MRR offline:       0.524 -> 0.535
```

L'ablation end-to-end ha cambiato due decisioni, non una: il secondo gruppo
(`SNGS-079`) non era valutabile nell'audit per GT raw insufficiente.

```text
config: configs/gsr_detection_tracking_conf012_tvcalib_pitch_rerank_w010_v1.yaml
run: gsr_valid_pilot12_conf012_tvcalib_pitch_rerank_w010_v1

SNGS-038:
  raw 16 -> raw 17 per target 18
  purity +0.001859
  HOTA -0.000008, AssA -0.000015, IDF1 invariato

SNGS-079:
  raw 7 -> raw 10 per target 38
  HOTA -0.002230
  AssA -0.002608
  MOTA -0.000630
  switch +0.630 / 1000 GT
  purity -0.003331

delta paired medio complessivo:
  HOTA: -0.000187
  AssA: -0.000219
  IDF1: 0
  0 sequenze migliorate, 10 pari, 2 peggiorate
```

Il reranker morbido e' respinto. Non fare altri sweep di soglia/peso sullo
stesso pilot: sarebbe tuning post-hoc con beneficio osservato nullo.

### Decisione congelata TVCalib

- Scartare TVCalib dal percorso operativo e dalla roadmap immediata di
  detection/tracking.
- Baseline GSR confermata: `configs/gsr_detection_tracking_conf012_v1.yaml`.
- Mantenere `calibration.tvcalib.enabled: false`.
- Mantenere `linking.pitch_gate.enabled: false`.
- Mantenere `linking.pitch_reranking.enabled: false`.
- Non usare le config `*_tvcalib_*apply*` o `*rerank*` come profili consigliati.
- Conservare codice, config e artifact solo per riproducibilita' dell'esito
  negativo; non cancellarli senza una pulizia esplicitamente richiesta.
- TVCalib non modifica la detection: precision, recall, F1 e AP sono rimasti
  identici in tutte le ablation.
- Prossimo audit tracking: attribuire i 22 candidati corretti mancanti a
  `gap`, distanza pixel, team gate, appearance gate, overlap o tracklet corto.

Interpretazione finale: TVCalib stima bene la geometria, ma non produce valore
operativo misurabile nel linker FT corrente e introduce ambiente separato,
dipendenze, inferenza e complessita'. Per questo progetto viene considerata
esperimento concluso e non promosso.

## Aggiornamento 2026-07-14 - benchmark GSR detection/tracking e confidence YOLO

### Obiettivo e perimetro

E' stato costruito un benchmark riproducibile per separare e misurare:

```text
raw YOLO detection -> raw ByteTrack identity -> FT display-track linking
```

Il benchmark usa soltanto sequenze annotate SoccerNet-GSR. OCR, roster,
identity evidence, propagation e PRTReID sono disabilitati, quindi non possono
alterare detection/tracking. La ground truth entra solo nell'evaluator offline,
mai nella pipeline operativa.

File principali aggiunti o modificati:

```text
ft/evaluation/gsr_detection_tracking.py
scripts/evaluate_ft_gsr.py
scripts/build_gsr_detection_tracking_manifest.py
scripts/run_gsr_detection_tracking_benchmark.py
scripts/aggregate_gsr_detection_tracking_benchmark.py
scripts/compare_gsr_detection_tracking_benchmarks.py
scripts/sweep_gsr_detection_confidence.py
configs/gsr_detection_tracking_benchmark_v1.yaml
configs/gsr_detection_tracking_conf010_v1.yaml
configs/gsr_detection_tracking_conf012_v1.yaml
docs/gsr_detection_tracking_benchmark.md
tests/test_gsr_detection_tracking_metrics.py
```

La pipeline esporta ora anche le detection YOLO pre-tracker con confidence.
L'evaluator implementa:

- matching Hungarian frame-level;
- precision, recall, F1 e mean IoU a IoU 0.50/0.75;
- AP50/AP75 da ranking confidence;
- breakdown per ruolo e dimensione bbox;
- HOTA TrackEval-compatible su IoU 0.05:0.95;
- DetA, AssA, MOTA, MOTP, IDF1, switch, frammentazioni e purity;
- metriche separate per `raw tracker ID` e `display_track_id` FT;
- aggregazione macro, micro e bootstrap CI appaiato per sequenza;
- CSV/JSON e grafici PNG/SVG.

### Pilot validation congelato

Manifest:

```text
evaluation/detection_tracking_manifests/valid_pilot12_v1.json
split: valid
seed: 20260713
12 sequenze x 750 frame = 9000 frame
SNGS-025, 036, 038, 039, 041, 050, 051, 055, 059, 079, 082, 088
model: /home/cappetti/FT/best_yolo26x_gsr_light.pt
```

Baseline iniziale, confidence `0.05`:

```text
detection micro:
  precision@0.50: 0.77968
  recall@0.50:    0.95805
  F1@0.50:        0.85971
  false positives: 42080
  false negatives:  6521

detection macro:
  AP50: 0.9508
  AP75: 0.6185
  FP/frame: 4.6756
  FN/frame: 0.7246

display tracking macro:
  HOTA: 0.60183
  DetA: 0.71007
  AssA: 0.51668
  IDF1: 0.70743

display tracking micro:
  IDF1: 0.70457
  ID switches: 1298
  fragmentations: 1941
```

Interpretazione baseline:

- il detector ha recall alta ma produce troppi falsi positivi;
- il forte calo AP50 -> AP75 indica un limite di localizzazione bbox;
- il collo di bottiglia tracking e' l'associazione (`AssA circa 0.517`), non
  soltanto la detection;
- MOTA alta non deve nascondere IDF1/AssA modesti.

Artifact baseline:

```text
evaluation_outputs/detection_tracking/gsr_valid_pilot12_baseline_v1
evaluation_outputs/detection_tracking/gsr_valid_pilot12_baseline_v1_aggregate
```

### Effetto linker raw -> display

Il linking FT ha effetto eterogeneo e non va disabilitato globalmente:

```text
IDF1 display - raw:
SNGS-050  -0.061
SNGS-039  -0.031
SNGS-051  -0.019
SNGS-079  +0.008
SNGS-055  +0.016
SNGS-082  +0.022
SNGS-088  +0.088
```

Le frammentazioni raw/display sono quasi identiche, mentre il linker aumenta
gli ID switch in 11 sequenze su 12. In alcuni video assegna correttamente una
stessa identita' globale a segmenti separati; in altri crea merge errati. Una
purity alta non e' un gate sufficiente: SNGS-051 ha purity `0.998` ma IDF1
display peggiore del raw.

Riferimenti per il prossimo audit linker:

```text
fallimenti: SNGS-050, SNGS-039, SNGS-051
successo forte: SNGS-088
tracking complessivamente debole: SNGS-079, SNGS-050, SNGS-039, SNGS-055
controlli positivi: SNGS-036, SNGS-088
```

### Sweep confidence e ablation pesanti

Lo sweep offline sulle curve PR esistenti ha indicato:

```text
confidence 0.12, vincolo recall >=0.95:
  precision=0.8541 recall=0.9503 F1=0.8996

confidence 0.38, best detection F1:
  precision=0.9537 recall=0.9187 F1=0.9359
```

`0.38` e' stata scartata senza run tracking: perde troppa recall e rischia di
spezzare le tracce. Sono state eseguite due ablation complete, una sola
variabile, a confidence `0.10` e `0.12`.

Risultati assoluti:

```text
conf   precision recall   F1      FP     FN    HOTA    AssA   IDF1micro switches fragments
0.05   .77968   .95805  .85971  42080   6521  .60183  .51668  .70457    1298     1941
0.10   .83985   .95256  .89266  28234   7374  .60206  .51689  .70489    1276     1918
0.12   .85451   .95078  .90008  25163   7650  .60243  .51703  .70546    1270     1949
```

`0.12` vs `0.05`:

- falsi positivi `42080 -> 25163`: `-16917`, circa `-40.2%`;
- falsi negativi `6521 -> 7650`: `+1129`;
- detection F1 `0.8597 -> 0.9001`;
- precision e F1 migliorano in tutte le 12 sequenze;
- recall cala di circa 0.7 punti in tutte le sequenze;
- DetA e MOTA migliorano leggermente con CI appaiato positivo;
- HOTA, AssA e IDF1 restano statisticamente sostanzialmente neutrali;
- ID switch totali `1298 -> 1270`;
- frammentazioni sostanzialmente stabili `1941 -> 1949`.

Regressioni associative localizzate a `0.12` vs `0.05`:

```text
SNGS-041: HOTA -0.01121, AssA -0.02151, IDF1 -0.01753
SNGS-079: HOTA -0.00779, AssA -0.01270, IDF1 -0.01425
SNGS-059: variazioni trascurabili
```

L'ablation `0.10` non ha recuperato un vantaggio tracking complessivo rispetto
a `0.12`: ha F1 detection inferiore in tutte le sequenze, 3071 FP in piu',
DetA inferiore e IDF1 display inferiore in 9/12 sequenze. Il solo vantaggio
concreto e' `31` frammentazioni in meno.

### Decisione congelata detection/tracking

Usare come nuova baseline **sperimentale** per le prossime ablation GSR:

```text
configs/gsr_detection_tracking_conf012_v1.yaml
detection.confidence: 0.12
model: best_yolo26x_gsr_light.pt
```

Non modificare ancora il default generale del progetto e non dichiarare
`0.12` produzione globale: la decisione e' validata sul pilot GSR validation
di 12 sequenze. Non provare altre confidence sullo stesso pilot, per evitare
overfitting al validation set.

Artifact principali:

```text
evaluation_outputs/detection_tracking/gsr_valid_pilot12_conf010_v1
evaluation_outputs/detection_tracking/gsr_valid_pilot12_conf010_v1_aggregate
evaluation_outputs/detection_tracking/gsr_valid_pilot12_conf012_v1
evaluation_outputs/detection_tracking/gsr_valid_pilot12_conf012_v1_aggregate
evaluation_outputs/detection_tracking/baseline_vs_conf010_v1
evaluation_outputs/detection_tracking/baseline_vs_conf012_v1
evaluation_outputs/detection_tracking/conf012_vs_conf010_v1
```

Prossima milestone:

- congelare detector/model/confidence;
- analizzare causalmente merge raw -> display nei fallimenti 050/039/051 e
  nel successo 088;
- progettare gate selettivi del linker, non disabilitazione globale;
- cambiare una sola variabile e confrontare HOTA/AssA/IDF1/switch/fragments;
- affrontare la localizzazione bbox/AP75 solo dopo la fase association.

## Aggiornamento 2026-07-13 - GSR ground truth e selettore automatico di leggibilita'

### Obiettivo e stato

SoccerNet-GSR viene ora usato come ground truth per detection, tracking, ruolo,
team e jersey OCR. Il selettore di frame leggibili e' stato integrato in
`JerseyOCR` con modalita' `audit/propose/apply`; resta disabilitato di default e
non e' promosso nel profilo operativo. Non modificare il profilo best-known
Int-Ata finche' la policy non viene congelata e verificata sullo split GSR
`test`.

Dataset remoto:

```text
/media/data-lie/cappetti/dataset/SoccerNet-GSR
Labels-GameState.json v1.3
750 frame per sequenza, 25 FPS, 1920x1080
GT: bbox_image, bbox_pitch, track_id, role, team, jersey
```

### Numeric-only OCR

Implementata modalita' OCR numerica conservativa:

- `ft/features/jersey_ocr.py`: `strict_numeric_only` e parsing `\d{1,2}`;
- EasyOCR/Tesseract usano allowlist/whitelist numerica;
- MMOCR/SAR mantiene il vocabolario pretrained ma scarta output misti;
- `ft/config.py`, `configs/default.yaml`: default `false`;
- config: `configs/best_known_int_ata_numeric_only_ocr.yaml`;
- config GSR: `configs/gsr_evaluation_control.yaml` e
  `configs/gsr_evaluation_numeric_only_ocr.yaml`.

Su `SNGS-034`, numeric-only non ha cambiato numeri assegnati o accuracy; ha
solo migliorato leggermente la calibrazione confidence:

```text
control:      Brier 0.27050, ECE 0.25174
numeric-only: Brier 0.26691, ECE 0.24728
```

Decisione: non promuovere numeric-only come miglioramento OCR autonomo.

### Evaluator e audit aggiunti

File principali:

```text
scripts/evaluate_ft_gsr.py
scripts/audit_gsr_ocr_oracle.py
scripts/build_gsr_legibility_audit.py
scripts/score_jersey_legibility.py
scripts/evaluate_legibility_weighted_ocr.py
scripts/evaluate_display_track_legibility_ocr.py
```

`evaluate_ft_gsr.py` usa IoU + Hungarian frame-level e riporta detection,
tracking proxy, team mapping, ruolo, jersey frame/tracklet, Brier/ECE e pitch
preliminare. La metrica pitch osservata non e' ancora valida: i sistemi di
coordinate non sono allineati.

Baseline `SNGS-034`:

```text
detection: P=0.9513 R=0.9426 F1=0.9469 meanIoU=0.838
tracking: 21 GT, 71 pred, switches_proxy=411, fragments_proxy=157
team accuracy=0.9197
role accuracy=0.9985
jersey frame visible coverage=0.4815
jersey frame accuracy emitted=0.6004
jersey frame accuracy all visible=0.2891
jersey tracklet=9/19=0.4737
```

### Oracle OCR e ipotesi frame leggibile

Su `SNGS-034`:

```text
current correct:                 9/19  = 47.4%
correct raw candidate exists:   18/19  = 94.7%
correct crop winner exists:     16/19  = 84.2%
recoverable current errors:      7/10
```

Conclusione: il collo di bottiglia principale non e' leggere ogni frame, ma
riconoscere il raro frame pulito e aggregarlo correttamente sul track.

Audit manuale iniziale `labels.csv`, 58 crop da una sola sequenza:

```text
readable=19, partial=15, not_readable=24
OCR correct: readable=7, not_readable=1
OCR wrong: readable=12, partial=15, not_readable=23
```

La leggibilita' e' necessaria ma non sufficiente: un crop leggibile puo'
comunque produrre OCR errato.

### Classificatore pretrained di leggibilita'

Repository remoto separato:

```text
/home/cappetti/tools/jersey-number-pipeline
commit: 007d54e
license upstream: CC BY-NC 3.0
```

Environment isolato, non modificare `mmocr`:

```text
conda env: jersey-legibility
Python 3.10.20
torch 2.2.2+cu121
torchvision 0.17.2+cu121
numpy 1.26.4
GPU RTX 3060 12 GB
```

Checkpoint:

```text
/home/cappetti/tools/jersey-number-pipeline/models/legibility_resnet34_soccer_20240215.pth
size: 85289629 bytes
sha256: b9c61dabaea4a6ec99528c5ae394f5875aecb8207de38484eccb0f977a373e41
architecture: ResNet34 -> 1 sigmoid, input 224x224 ImageNet normalization
```

Caricamento strict state dict e inference CUDA verificati.

Valutazione sui 43 crop manuali binari:

```text
precision=0.7917
recall readable=1.0
specificity=0.7917
F1/accuracy=0.8837
ROC-AUC=1.0
```

Il modello satura spesso vicino a 1 sui crop readable/partial: usarlo come
gate/ranking, non come confidence calibrata fine.

### Correzione leakage e policy display-track

Il primo esperimento raggruppava per `gt_track_id`: risultato utile come oracle
ma non deployable. Correzioni:

- `audit_gsr_ocr_oracle.py` conserva `display_track_id`;
- conserva anche crop senza match GT;
- selezione e voto sono raggruppati solo per `display_track_id`;
- GT entra soltanto dopo la decisione, per le metriche;
- gate ruolo track-level e frame-level usano `pred_role`, non GT.

Policy studiata:

```text
crop OCR -> P(readable) -> top-k temporalmente diversi
         -> voto confidence OCR -> decisione display track
         -> propagazione sul track con gate ruolo
```

Su `SNGS-034`, policy iniziale top5/gap5/player-only/frame-role gate:

```text
tracklet diagnostic: 15/19 = 78.9% vs 9/19 baseline
frame visible coverage: 88.52% vs 48.15%
frame accuracy emitted: 60.79% vs 60.04%
frame accuracy all visible: 53.82% vs 28.91%
false emission no-jersey GT: 5/1356 = 0.37% vs 1.18% baseline
```

Il frame-role gate ha rimosso contaminazioni ID-switch verso arbitri senza
perdere frame visibili utili su questa sequenza.

### Generalizzazione GSR validation

Policy valutata senza cambiare soglie su cinque nuove sequenze congelate:

```text
SNGS-023 Offside
SNGS-039 Shots off target
SNGS-054 Goal
SNGS-078 Clearance
SNGS-085 Kick-off
```

Baseline tracklet totale:

```text
17/99 = 17.2%
```

Policy aggressiva `min_winner_votes=1`:

```text
24/99 = 24.2%
problema: un singolo OCR errato puo' propagarsi per centinaia di frame
false rate: SNGS-023=40.8%, SNGS-054=17.0%
```

Cause osservate: giocatore senza jersey GT, portieri classificati player e
arbitri contaminati; quasi tutti i falsi derivavano da un solo voto OCR.

Policy conservativa candidata:

```yaml
top_k: 5
min_legibility_score: 0.50
min_frame_gap: 5
min_winner_votes: 2
min_margin: 0.0
allowed_roles: [player]
gate_each_frame_role: true
```

Risultati validation:

```text
sequence   baseline track  candidate track  frame_all base->candidate  false candidate
SNGS-023       5/19             5/19             .212 -> .239              .002
SNGS-039       1/20             2/20             .001 -> .112              .000
SNGS-054       3/20             3/20             .071 -> .114              .000
SNGS-078       4/20             5/20             .094 -> .233              .000
SNGS-085       4/20             7/20             .144 -> .303              .000
totale        17/99            22/99
```

La variante `min_margin=0.10` azzera gli ultimi falsi ma scende a `20/99` e
riduce frame accuracy/coverage in 4 sequenze su 5. Tenerla come profilo strict,
non come candidata principale.

Decisione attuale:

- candidata bilanciata: `min_votes=2`, `margin=0.0`;
- strict zero-emission: `min_votes=2`, `margin=0.10`;
- il selettore e' integrato con `audit/propose/apply`, ma nessuna policy e'
  ancora promossa nel profilo operativo;
- prossimo test scientifico: congelare entrambe e misurare sullo split GSR
  `test`, senza altra regolazione;
- `SNGS-039` e `SNGS-054` hanno oracle OCR basso: il selettore non puo'
  recuperare numeri corretti che nessun backend OCR produce.

### Refactoring leggibilita' pipeline

Aggiunti commenti/docstring ai file centrali:

```text
ft/cli.py
ft/pipeline.py
ft/features/jersey_ocr.py
scripts/evaluate_display_track_legibility_ocr.py
```

Prima tranche di refactoring incrementale in `ft/pipeline.py`:

```text
RunSetup (dataclass immutabile)
TemporalAnalysis (dataclass immutabile)
initialize_run()
read_video_phase()
analyze_temporal_structure()
tracking_phase()
calibration_phase()
render_output_phase()
```

La logica e l'ordine degli stage diagnostici restano invariati. Le mutazioni di
`tracks` restano esplicite; non e' stato introdotto un enorme context globale.

Test aggiunto:

```text
tests/test_pipeline_phases.py
```

Verifiche locali:

```text
pipeline phase contract tests: passed
tests/test_validation.py: passed
python compileall ft/scripts/tests: passed
git diff --check: passed
```

Limite locale: OpenCV e pytest non sono installati nell'ambiente desktop. E'
stato predisposto uno smoke remoto `SNGS-023_refactor_smoke_20f`; l'esistenza
dell'artifact e' stata osservata, ma l'output finale non e' stato riportato in
chat. Non considerare ancora confermata equivalenza end-to-end bit-for-bit.

### Stato storage remoto

Lo spazio `artifacts` era circa 46 GB ed e' sceso a circa 23 GB dopo pulizia.
Molti crop GSR/numeric-only possono non essere piu' presenti. I metadata e gli
output di valutazione restano in `evaluation_outputs` (circa 1.8 GB). Prima di
proporre nuovi audit visuali verificare l'esistenza reale dei crop; se assenti,
rigenerare la run o usare gli score CSV gia' calcolati.

### Config/modello operativo da preservare

Miglior profilo osservato finora su `Int-Ata`:

```text
Int-Ata_number_region_scene_reset_control_1200f
```

Metriche:

```text
player_rows: 15119
assigned_rows: 8832
unknown_rows: 6287
jersey_rows: 8899
assigned_rate: 0.5842
identity_status: assigned=8832, unknown=5465, invalidated=822
duplicates team/jersey: 0
duplicates player: 0
display_tracks: 85
short tracks <=10 rows: 8
median track rows: 125
```

Modello operativo consigliato:

```text
best_yolo26x_gsr_light.pt
```

Config locale utile per riprodurre il profilo migliore:

```text
configs/best_known_int_ata_number_region_scene_reset_control.yaml
```

Questa config eredita il profilo number-region/scene-reset di controllo e
disabilita le componenti non promosse: activity segmentation, propagation,
PRTReID linking/bridge, jersey identity linking, scene-segmented Hungarian e
number-region cost aggressivo.

### Benchmark Identity Evaluation V1

Implementato un benchmark riproducibile per valutazione identity:

- spec: `evaluation/identity_benchmark_v1.yaml`;
- builder: `scripts/build_identity_benchmark.py`;
- evaluator: `scripts/evaluate_identity_benchmark.py`;
- output: `evaluation_outputs/identity_benchmark_v1_smoke`;
- unita': identity tracklet/scene segment, non righe frame-level;
- include direct, unknown, propagated, constraint-invalidated, accepted pair e
  rejected near-miss;
- annotazione cieca HTML reviewer A/B;
- 20% seconda revisione deterministica;
- preconstraint artifact disponibile per misurare false invalidation.

Smoke builder confermata:

```text
identity_units: 159
pairs: 43
second_review: 41 / 202
splits: development=108, test=33, external=18
artifact_hashes: 14
leaked fields HTML: []
```

Regola di promozione identity: zero falsi positivi sulle nuove identita'
determinabili, zero link errati, nessun aumento duplicati/violazioni, coverage
non inferiore alla baseline. I casi `not_determinable` rendono il delta
inconclusivo, non positivo.

### Preconstraint/export identity status

Fix applicata localmente e sincronizzata sulla remota:

- `ft/export/artifacts.py`;
- `ft/pipeline.py`.

Comportamento nuovo:

```python
track.get("identity_status") or "unknown"
```

Motivo: alcune righe `player_id=unknown`, `jersey_number=''` uscivano con
`identity_status=''`. Semanticamente sono unknown, non una quarta classe.

Smoke di verifica:

```text
run: Int-Ata_status_export_fix_smoke_80f
rows: 1084
identity_status: assigned=762, unknown=232, invalidated=90
empty identity_status: 0
```

La fix non cambia la logica di assegnazione, solo gli artifact esportati.

### Activity segmentation

Implementata e testata segmentazione per micro-spezzoni basata su numero di
elementi rilevati/scena:

- artifact: `{video_id}_activity_segments.json`;
- segmenti hard da scene cut;
- segmenti soft da persistent count change;
- utile come diagnostica, non promossa per reset/tracking.

Risultati:

```text
Inter-Juve_activity_segments_1200f:
  scene cuts: 0
  segments: 3
  hard: 0
  soft: 2
  interpretazione: video senza veri tagli, segmentazione poco informativa

Int-Ata_activity_segments_1200f:
  scene cuts: 3
  tracker_reset_frames: 1 hard_only=True
  segments: 13
  hard: 1
  soft: 11
```

Ispezione qualitativa: i microclip prodotti per count-change sono pessimi come
azioni; il numero di detection non e' un buon boundary semantico. Usare il
conteggio solo come feature di qualita'/diagnostica, non come reset ByteTrack.

### YOLO GSR person+ball

Dataset GSR person+ball v2 preparato con split corretto:

```text
/media/data-lie/cappetti/dataset/soccernet_gsr_yolo_person_ball_v2/data.yaml
classes: person, ball
sequences: 164
frames: train=42750, val=43500, test=36750
labels: person=1927096, ball=115960
```

Fix script training:

- `scripts/train_yolo_gsr_full.py`;
- `/test/` ora va in split `test`, non `val`;
- aggiunti `--skip-prepare`, `--seed`, `--non-deterministic`,
  `--close-mosaic`, `--fraction`, `--no-val`;
- `project` risolto assoluto per evitare output annidati in
  `runs/detect/runs/...`.

Training migliore standalone:

```text
run: yolo26x_person_ball_full_10e_s0
nota: nome fuorviante, lanciata a 20 epoche full dataset
model: runs/ft_yolo_gsr_v2/yolo26x_person_ball_full_10e_s0/weights/best.pt

val GSR:
all    P=0.788 R=0.650 mAP50=0.660 mAP50-95=0.360
person P=0.920 R=0.950 mAP50=0.963 mAP50-95=0.582
ball   P=0.656 R=0.351 mAP50=0.357 mAP50-95=0.138
```

Confronto:

```text
smoke 3e 5%:     ball mAP50 circa 0.278
30e 10%:         ball mAP50 circa 0.221
20e 100%:        ball mAP50 0.357
```

Conclusione detection: il modello person+ball full20e e' il migliore su
validation GSR.

Conclusione end-to-end FT: non promosso.

Run pipeline:

```text
Int-Ata_bestknown_yolo_personball_full20e_1200f_v2
```

Confronto con baseline operativa:

```text
baseline Int-Ata_number_region_scene_reset_control_1200f:
  player_rows: 15119
  assigned_rows: 8832
  unknown_rows: 6287
  jersey_empty: 6220
  invalidated: 822
  display_tracks: 85
  duplicates: 0

new_yolo person+ball:
  player_rows: 16429
  assigned_rows: 7284
  unknown_rows: 9145
  jersey_empty: 9125
  invalidated: 1688
  display_tracks: 103
  duplicates: 0
```

Interpretazione: mAP detection migliore non implica pipeline migliore. Il nuovo
detector genera piu' tracklet, piu' frammentazione e meno OCR utile; la coverage
identity peggiora e gli invalidated raddoppiano. Non usarlo come modello
operativo identity. Puo' restare come esperimento detection standalone.

### Warning noti da ignorare se la run prosegue

```text
RequestsDependencyWarning urllib3/chardet
FutureWarning ByteTrack deprecated
MMOCR unexpected key data_preprocessor.mean/std
MMEngine registry scope "mmocr"
LocalVisBackend save_dir
FT identity warning: roster visual profiles=0; strong_team_visual_trajectory cannot pass
```

Significato ultimo warning: il roster non contiene profili visuali reference,
quindi il gate `strong_team_visual_trajectory` non puo' passare. Non e' un
errore; l'identita' resta basata su OCR numero, team/colore, roster, tracking e
constraint.

Baseline storica stabile BallKalman:

```text
Inter-Atalanta_scenecuts_loose_noreset_gkfallback_spreadgate_propagation_ballkalman_gate_full
```

Checkpoint GitHub storico:

```text
commit 9d564a4 Add BallKalman baseline for broadcast ball tracking
```

Baseline V1 corrente per confronto su `Int-Ata`:

```text
Int-Ata_identity_evidence_v1_resetbytetrack_cutsensitive_1200f
```

Risultato baseline V1 cutsensitive:

```text
player_rows: 15119
assigned_rows: 5986
unknown_rows: 9133
jersey_rows: 6108
propagated_rows: 560 reali (il vecchio audit riportava erroneamente 0)
unique_display_ids: 85
duplicates: 0
scene cuts: [294, 517, 715, 925, 1052]
```

Correzione importante: le 560 righe aggiuntive della baseline storica sono
propagate (`5 -> 38`, 511 righe, e `64 -> 76`, 49 righe). Il vecchio audit
guardava soltanto `identity_status == propagated`, ma queste righe avevano
`identity_status: unknown` e `identity_evidence.status: propagated`.
`scripts/audit_identity_evidence.py` ora controlla entrambe le sorgenti.

## Identity Evidence V1

Obiettivo V1: separare evidenze e inferenza, mantenendo la pipeline FT esistente.

File principali:

- `configs/default_realvideo_identity_evidence_v1.yaml`
- `ft/identity/evidence.py`
- `scripts/audit_identity_evidence.py`
- modifiche in:
  - `ft/pipeline.py`
  - `ft/identity/hungarian.py`
  - `ft/identity/propagation.py`
  - `ft/identity/constraints.py`
  - `ft/export/artifacts.py`

Artifact V1 attesi:

```text
{video_id}_identity_evidence.csv/json
{video_id}_identity_decisions.csv/json
{video_id}_identity_constraint_actions.csv
```

Fix V1 gia' applicati sulla remota:

- `ft/identity/evidence.py`: helper `is_empty()` per evitare `TypeError: unhashable type: 'list'`.
- `ft/export/artifacts.py`: `write_csv` estende i fieldnames con tutte le chiavi presenti nelle righe.
- `scripts/audit_identity_evidence.py`: artifact presence basata su `path.exists()`.
- `ft/pipeline.py`: export evidence/decisions/constraint actions.
- `ft/pipeline.py`: `team_config()` filtra `assignment_mode`.
- `ft/pipeline.py`: `linking_cfg` filtra `embedding_mode`.

## Scene Cut e Reset ByteTrack

Su `Inter-Atalanta` full non vengono rilevati cut:

```text
Inter-Atalanta_identity_evidence_v1_resetbytetrack_full
cuts=0
reset_tracking=True
assigned_rows=2019
```

Quindi il reset non misura nulla su quel video.

Su `Int-Ata` bisogna usare config cutsensitive per trovare i 5 tagli attesi:

```yaml
scene_cuts:
  enabled: true
  tracking_reset_enabled: true
  threshold: 0.08
  hard_cut_threshold: 0.12
  min_gap: 40
```

Run confermata:

```text
Int-Ata_identity_evidence_v1_resetbytetrack_cutsensitive_1200f
cuts=5
cut_frames=[294, 517, 715, 925, 1052]
assigned_rows=5986
duplicates=0
```

## PRTReID - Stato Integrazione storica come backend visuale

Questa sezione descrive la prima integrazione ed e' conservata per contesto.
Lo stato corrente e' nella sezione `Aggiornamento 2026-06-23 - PRTReID linker
additivo`.

Obiettivo: PRTReID FT-native, senza usare TrackLab come runner.

PRTReID e' stato integrato come backend visuale:

- `visual.embedding_mode: prtreid`
- nuovo adapter: `ft/features/prtreid.py`
- config: `configs/default_realvideo_identity_evidence_v1_prtreid.yaml`
- check env: `scripts/check_prtreid_env.py`
- export dataset: `scripts/export_prtreid_dataset.py`
- train wrapper: `scripts/train_prtreid.py`

Environment remoto sistemato:

```text
torch 2.0.0+cu117
torchvision 0.15.1+cu117
numpy 1.26.4
albumentations 1.3.1
prtreid OK
torchreid OK
omegaconf OK
yacs OK
```

Checkpoint presenti:

```text
models/reid/prtreid-soccernet-baseline.pth.tar
models/reid/hrnetv2_w32_imagenet_pretrained.pth
```

Check passato:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/cappetti/FT \
python3 scripts/check_prtreid_env.py --skip-md5
```

## Hotfix PRTReID applicati sulla remota

Questi fix sono necessari per far girare PRTReID nella pipeline FT:

1. `ft/features/visual.py`

`VisualFeatureExtractor` filtra la config PRTReID prima di istanziare `PRTReIDFeatureExtractor`, per non passare chiavi runtime come:

```text
role_min_confidence
role_protect_existing
```

2. `ft/features/prtreid.py`

Registrazione dataset FT-native minimale:

```python
class FTNativeReIDDataset:
    dataset_dir = "FTNative"

    @staticmethod
    def get_masks_config(_masks_dir):
        return None
```

La config interna usa:

```text
data.sources = ["FTNative"]
data.targets = ["FTNative"]
```

3. `ft/features/prtreid.py`

`project.job_id` deve restare `int`, non stringa:

```python
"job_id": int(os.getpid())
```

4. `ft/features/prtreid.py`

`data.save_dir` usa una path temporanea unica per evitare:

```text
FileExistsError: [Errno 17] File exists: '0'
```

Valore usato:

```python
"save_dir": str(Path(tempfile.gettempdir()) / f"ft_prtreid_{os.getpid()}_{id(self)}")
```

## Smoke PRTReID riuscite

Run 50f su `Inter-Atalanta`:

```text
RUN=Inter-Atalanta_identity_evidence_v1_prtreid_50f
visual_features: ok
backend: prtreid
device: cuda
computed: 660
failed: 0
embedding_dim: 256
models: {'prtreid'}
reid_roles: player=612, goalkeeper=48
assigned_rows=324 / 756
```

Run 20f su `Int-Ata` cutsensitive reset:

```text
RUN=Int-Ata_identity_evidence_v1_prtreid_resetbytetrack_cutsensitive_20f_scenetest
scene cuts: cuts=0 reset_tracking=True
visual_features: ok
assigned_rows=114 / 274
```

## PRTReID 1200f - Risultati

### 1. PRTReID + team PRTReID-KMeans

Run:

```text
Int-Ata_identity_evidence_v1_prtreid_resetbytetrack_cutsensitive_1200f
```

Risultato:

```text
player_rows: 15119
assigned_rows: 5469
unknown_rows: 9650
jersey_rows: 5476
propagated_rows: 0
unique_display_ids: 85
duplicates: 0
```

Delta vs baseline V1 cutsensitive:

```text
assigned_rows: -517
unknown_rows: +517
jersey_rows: -632
```

Conclusione: `team.assignment_mode: prtreid_kmeans` peggiora molto. Non promuoverlo.

### 2. PRTReID + team color legacy + propagation

Run:

```text
Int-Ata_identity_evidence_v1_prtreid_colorteam_resetbytetrack_cutsensitive_1200f
```

Risultato:

```text
player_rows: 15119
assigned_rows: 5908
unknown_rows: 9211
jersey_rows: 6037
propagated_rows: 559
unique_display_ids: 85
duplicates: 0
```

Delta vs baseline V1 cutsensitive:

```text
assigned_rows: -78
unknown_rows: +78
jersey_rows: -71
propagated_rows: +559
```

Propagation accettate:

```text
5 -> 38, player_id=team1_05, Benjamin Pavard
visual_similarity=0.9608
assignment_confidence=0.554
target_frames=515

64 -> 76, player_id=team2_sub_02, Rafael Toloi
visual_similarity=0.9575
assignment_confidence=0.532
target_frames=49
temporal_gap=148
```

Conclusione: run quasi pari alla baseline, ma recupera tramite propagation debole. Da ispezionare visivamente prima di fidarsi.

### 3. PRTReID + team color legacy + no propagation

Run:

```text
Int-Ata_identity_evidence_v1_prtreid_colorteam_noprop_resetbytetrack_cutsensitive_1200f
```

Risultato:

```text
assigned_rows: 5349
tracklet_rows: 16338
```

Conclusione: senza propagation peggiora molto. Quindi PRTReID non migliora ancora l'assegnazione diretta; la run colorteam "buona" e' sostenuta dalla propagation.

## Interpretazione tecnica PRTReID storica

Questa interpretazione e' superata dal secondo linker additivo documentato
nell'aggiornamento del 2026-06-23.

PRTReID e' integrato e produce embedding validi:

```text
embedding_mode: prtreid
backend: prtreid
device: cuda
embedding_dim: 256
failed: 0
```

Pero' oggi PRTReID viene calcolato dopo:

```text
tracking -> team_first_pass -> referee_first_pass -> linking -> team_second_pass -> goalkeeper -> export_pre_identity -> visual_features
```

Quindi PRTReID non aiuta ancora il linking, perche' il linking avviene prima del calcolo degli embedding deep.

Effetto attuale:

- `prtreid_kmeans` team assignment peggiora.
- Con team color legacy e role classifier attivo, PRTReID peggiora leggermente la direct assignment.
- Disabilitando il role classifier, PRTReID torna esattamente alla baseline ma non la migliora.
- Gli embedding distinguono bene soprattutto il team, non l'identita' all'interno dello stesso team.
- La propagation PRTReID a similarity circa `0.96` non e' affidabile: molti impostori same-team superano quella soglia.

## Ablation PRTReID role classifier - completata

Run completata:

```text
Int-Ata_identity_evidence_v1_prtreid_colorteam_norole_resetbytetrack_cutsensitive_1200f
```

Configurazione rilevante:

```yaml
visual:
  embedding_mode: prtreid

team:
  assignment_mode: color

prtreid:
  role_enabled: false
```

Risultato vs baseline V1 cutsensitive:

```text
player_rows: 15119 vs 15119
assigned_rows: 5986 vs 5986
unknown_rows: 9133 vs 9133
jersey_rows: 6108 vs 6108
duplicates: 0 vs 0
propagated_rows: 560 vs 0
```

Conclusione: il role classifier PRTReID causava il calo della run precedente. Con `role_enabled: false` la direct assignment torna alla baseline, ma PRTReID resta neutro e non aggiunge assegnazioni.

## Audit discriminativo embedding PRTReID

Nuovo script:

```text
scripts/audit_prtreid_embeddings.py
```

Lo script aggrega gli embedding per display tracklet e misura:

- similarita' same-player, quando esistono identita' affidabili ripetute;
- impostori different-player same-team;
- coppie different-team;
- nearest-neighbor accuracy per player/team;
- rischio impostori alle soglie `0.90`, `0.95`, `0.97`, `0.98`, `0.99`;
- similarita' delle coppie accettate dalla propagation.

Risultato sulla run `norole`:

```text
embedded_rows: 15119
embedded_tracklets: 80
identity_labeled_tracklets: 23
players_with_multiple_tracklets: 0
embedding_dim: 256

different_player_same_team:
  mean: 0.9343
  p90: 0.9768
  max: 0.9897

different_team:
  mean: 0.7211
  p90: 0.8566
  max: 0.9714

team nearest-neighbor top1: 0.9375
```

Rischio impostori same-team:

```text
threshold 0.90: 83.46%
threshold 0.95: 40.60%
threshold 0.97: 20.30%
threshold 0.98:  7.52%
threshold 0.99:  0.00% nel campione corrente
```

Non ci sono player affidabili ripetuti tra tracklet, quindi non e' ancora possibile misurare direttamente la distribuzione same-player.

Propagation accettate nella run:

```text
64 -> 76, team2_sub_02, similarity=0.9575, confidence=0.5321
5 -> 38, team1_05, similarity=0.9608, confidence=0.5540
```

Entrambe cadono in una zona con alto rischio di impostori same-team. Non considerarle affidabili senza verifica visiva o altra evidenza forte.

## Config PRTReID colorteam storica

La config:

```text
configs/default_realvideo_identity_evidence_v1_prtreid_cutsensitive_resetbytetrack_colorteam.yaml
```

ora disabilita esplicitamente la propagation:

```yaml
identity_propagation:
  enabled: false
```

Anche la config `norole` eredita questo comportamento. Prossima eventuale run di controllo:

```text
Int-Ata_identity_evidence_v1_prtreid_colorteam_norole_noprop_resetbytetrack_cutsensitive_1200f
```

Risultato atteso: metriche identiche alla baseline con `propagated_rows: 0`.

## Aggiornamento 2026-06-23 - PRTReID linker additivo

L'architettura e' stata modificata. PRTReID non e' piu' soltanto un backend
visuale calcolato troppo tardi: ora costruisce prototipi di tracklet prima di
un secondo linker conservativo.

Flusso corrente:

```text
tracking -> team/referee -> linker legacy -> prototipi PRTReID
         -> linker PRTReID -> feature/identity pipeline legacy
```

Il descriptor deep usato dal secondo linker e' isolato dall'evidenza visuale
downstream. La pipeline identitaria continua a usare `hsv_lab_gradient`.
Disabilitando `prtreid_linking` non viene eseguita inferenza PRTReID.

File principali aggiunti o modificati:

- `ft/features/prtreid.py`: estrazione da crop in memoria e temp dir unica.
- `ft/features/prtreid_tracklets.py`: sampling temporale/qualita', massimo 8
  crop, prototipo medio normalizzato e diagnostica.
- `ft/linking/prtreid_tracklet_linker.py`: team gate, overlap gate, distanza,
  gap, similarity, margin, mutual-nearest, same/cross-scene e union sicura.
- `ft/pipeline.py`: integrazione dopo il linker legacy e prima dell'identita'.
- `ft/config.py`, `ft/validation.py`: schema e validazione `prtreid_linking`.
- `scripts/build_prtreid_pair_audit.py`: audit visuale; HTML interattivo con
  tasti `1=same`, `2=different`, `3=uncertain` ed export `labels.csv`.
- `scripts/calibrate_prtreid_linker.py`: calibrazione zero-FP; le soglie sono
  serializzate con precisione float completa per non perdere coppie al bordo.
- `scripts/export_prtreid_dataset.py`: export SoccerNet-GSR sequence-disjoint,
  PID/team/camera densi per split e sampling prima della scrittura dei crop.
- `ft/features/prtreid_dataset.py`: dataset registrato e sampler same-team.
- `scripts/train_prtreid.py`: identity CE + triplet, role/team loss escluse,
  timer PRTReID compatibile con PyTorch, early stopping dopo minimo 10 epoche.
- `scripts/evaluate_prtreid_checkpoint.py`: evaluation streaming dai path.
- `scripts/audit_identity_evidence.py`: riconoscimento corretto delle righe
  propagate e floor/ceiling configurabili.

Artifact nuovi:

```text
{video_id}_prtreid_tracklet_features.json
{video_id}_prtreid_linking.json
{video_id}_prtreid_linking.csv
```

Config importante: la prima config candidate ereditava dal vecchio backend
PRTReID `linking.appearance_min_similarity: 0.55`, alterando anche il linker
legacy. La config realmente additiva parte direttamente dalla baseline e
mantiene il valore legacy `0.72`:

```text
configs/default_realvideo_identity_evidence_v1_prtreid_linking_additive_cutsensitive_resetbytetrack.yaml
```

Profilo operativo validato, limitato a Int-Ata:

```text
configs/prtreid_linking_conservative_int_ata_v1.yaml
```

Impostazioni rilevanti:

```yaml
visual:
  embedding_mode: hsv_lab_gradient
team:
  assignment_mode: color
prtreid:
  role_enabled: false
identity_propagation:
  enabled: false
prtreid_linking:
  same_scene:
    enabled: false
  cross_scene:
    enabled: true
    max_segment_gap: 1
    max_gap: 60
    min_similarity: 0.92511475086212158
    min_margin: 0.042288899421691895
    mutual_nearest: true
```

## Audit manuale e calibrazione

Audit Int-Ata completato su 40 coppie:

```text
different: 35
same: 4
uncertain: 1
```

I quattro same erano tutti cross-scene. L'unico candidato same-scene era
errato, quindi la policy same-scene resta disabilitata.

Calibrazione mutual-nearest a zero falsi positivi:

```text
accepted: 2
true positives: 2
false positives: 0
precision: 100%
recall sulle positive mutual-nearest etichettate: 50%
```

Bridge approvati:

```text
69 -> 94,  sim=0.961268, margin=0.198345
100 -> 111, sim=0.925115, margin=0.042289
```

## Run finale PRTReID conservativa

Run da mantenere:

```text
Int-Ata_prtreid_conservative_final_1200f
```

Controllo matched, PRTReID disabilitato e propagation disabilitata:

```text
Int-Ata_prtreid_additive_control_1200f
```

Il controllo e il relativo video sono stati eliminati dopo aver salvato il
report; il report resta in:

```text
evaluation_outputs/Int-Ata_prtreid_conservative_final_audit.json
```

Risultato finale contro il controllo matched:

```text
player_rows:       15119 -> 15119
assigned_rows:      5426 -> 5426
unknown_rows:       9693 -> 9693
jersey_rows:        5548 -> 5548
propagated_rows:       0 -> 0
unique_display_ids:   85 -> 83
duplicates:             0
scene cuts: [294, 517, 715, 925, 1052]
metadata_gate_pass: true
```

Comando audit coerente con propagation disabilitata:

```bash
python3 scripts/audit_identity_evidence.py \
  --video-id Int-Ata \
  --baseline-run Int-Ata_prtreid_additive_control_1200f \
  --candidate-run Int-Ata_prtreid_conservative_final_1200f \
  --assigned-floor 5426 \
  --unknown-ceiling 9693
```

Il controllo non e' piu' presente su disco, quindi questo comando e' storico;
usare il report JSON salvato per consultare il confronto.

## SoccerNet-GSR e fine-tuning PRTReID

Dataset disponibile sulla remota:

```text
/media/data-lie/cappetti/dataset/SoccerNet-GSR
37 GB, 164 Labels-GameState.json, 150000 immagini
```

Dataset esportato:

```text
datasets/prtreid_ft_v1_full
34317 crop, 185 MB
17151 train, 3433 query, 13733 gallery
2290 identita', 115 sequenze
```

Baseline pretrained sulla validation completa:

```text
rank1: 0.466647
mAP: 0.290903
same-team zero-FP threshold: 0.9983523
zero-FP recall: 0.002428% (circa 1 coppia su 41189)
```

Il primo training si fermava durante il warm-up; e' stato aggiunto
`--min-epochs 10`. Il training corretto e' arrivato a epoca 13, poi early stop.
Nessun checkpoint e' promuovibile:

```text
miglior zero-FP recall: 0.012139% (epoca 1)
rank1 epoca 1: 0.4317
mAP epoca 1: 0.2638
miglior rank1: 0.4570 (epoca 7), zero-FP recall 0%
```

Conclusione training: il fine-tuning peggiora rank1/mAP e resta molto lontano
dal 20% zero-FP richiesto. Non usare i checkpoint fine-tuned; il profilo finale
usa ancora `models/reid/prtreid-soccernet-baseline.pth.tar`.

## Limite di generalizzazione

Run esterna:

```text
Inter-Juve_prtreid_external_validation_1200f
scene cuts: 0
PRT features: 38
accepted links: 0
```

Anche tutti i run Inter-Atalanta disponibili hanno zero scene cut. Questi video
non possono validare la policy cross-scene. Il profilo conservativo e' quindi
approvato solo come opt-in specifico per Int-Ata; non impostarlo come default
generale finche' non viene etichettato un secondo video con cut reali.

## Test eseguiti

- compile Python dei nuovi script/moduli.
- `tests/test_identity.py` completo nell'ambiente locale `tesi`: passato.
- test propagation: passati.
- test config/validation: passati, salvo file modello assenti in locale.
- test sintetici audit/calibrazione/exporter: passati.
- smoke remota 200 frame PRT attivo vs controllo disabilitato: metriche
  identiche, confermata additivita'.
- run Int-Ata 1180 frame finale: due merge, zero errori/duplicati/regressioni.

Nota ambiente locale: non contiene l'intero stack PRTReID e non puo' fare
inference reale; le run CUDA restano sulla remota.

## Comandi audit utili

Audit comparativo:

```bash
cd /home/cappetti/FT

python3 scripts/audit_identity_evidence.py \
  --video-id Int-Ata \
  --baseline-run Int-Ata_identity_evidence_v1_resetbytetrack_cutsensitive_1200f \
  --candidate-run RUN_NAME
```

Verifica PRTReID artifacts:

```bash
cd /home/cappetti/FT

python3 - <<'PY'
import json, csv
from pathlib import Path

RUN="RUN_NAME"
VIDEO_ID="Int-Ata"
root=Path("artifacts/costume-video")/RUN/"metadata"

print(json.dumps(json.loads((root/f"{VIDEO_ID}_visual_features.json").read_text()), indent=2)[:3000])

rows=list(csv.DictReader((root/f"{VIDEO_ID}_tracklets.csv").open()))
dims=[]
models=set()
roles={}
for r in rows:
    if r.get("reid_model"):
        models.add(r["reid_model"])
    if r.get("visual_embedding"):
        dims.append(len(json.loads(r["visual_embedding"])))
    if r.get("reid_role_detection"):
        roles[r["reid_role_detection"]] = roles.get(r["reid_role_detection"], 0) + 1

print("models:", models)
print("embedding_dims:", sorted(set(dims))[:10], "count", len(dims))
print("reid_roles:", roles)
PY
```

Verifica scene cuts:

```bash
cd /home/cappetti/FT

python3 - <<'PY'
import json
from pathlib import Path

RUN="RUN_NAME"
VIDEO_ID="Int-Ata"
p=Path("artifacts/costume-video")/RUN/"metadata"/f"{VIDEO_ID}_scene_cuts.json"
d=json.loads(p.read_text())
print("cuts:", len(d.get("cut_frames", [])))
print("cut_frames:", d.get("cut_frames", []))
for c in d.get("cuts", [])[:20]:
    print(c)
PY
```

## Warning noti

Warning innocuo ma da pulire:

```text
unknown config key: linking.embedding_mode
```

Motivo: alcune config V1 ereditano `linking.embedding_mode`, ma `TrackletLinker` non lo usa. Il codice remoto filtra gia' questa chiave prima del costruttore, quindi non blocca. Pulizia futura: rimuovere `linking.embedding_mode` dalle config o aggiungerlo allo schema validato.

Warning MMOCR e requests:

```text
RequestsDependencyWarning
unexpected key in source state_dict: data_preprocessor.mean/std
```

Gia' presenti prima, non bloccano le run.

## Aggiornamento 2026-06-23 - Audit causale identity constraints

E' stato aggiunto un audit specifico per distinguere cancellazioni di identita'
da pulizia di jersey OCR:

```text
scripts/audit_identity_constraints.py
```

L'audit:

- aggrega per `(action_type, team, jersey, cleared_player_id, kept_player_id)`;
- deduplica per frame, raw track ID e tipo di azione;
- separa righe identity e jersey cancellate;
- riporta overlap, display coinvolti, confidence, votes e margin OCR;
- classifica same-player bug, competing owner, unknown-vs-known e conflitti
  same-frame;
- valida l'unicita' `(team_id, jersey_number)` del roster;
- esporta JSON e CSV nella directory metadata della run.

Le azioni prodotte da `ft/identity/constraints.py` ora includono row keys
esatte e metriche pre-cancellazione. `ft/utils/run_diagnostics.py` esporta
inoltre:

```text
{video_id}_source_provenance.json
```

con SHA-256 della config risolta e di:

```text
ft/identity/constraints.py
ft/identity/hungarian.py
ft/linking/jersey_identity_linker.py
```

### A1 - controllo matched

Run:

```text
Int-Ata_identity_constraints_a1_control_1200f
```

Configurazione: propagation, PRTReID linking e jersey identity linking
disabilitati.

Risultato:

```text
video frames: 1180
scene cuts: [294, 517, 715, 925, 1052]
player_rows: 15119
assigned_rows: 5426
unknown_rows: 9693
jersey_rows: 5548
propagated_rows: 0
unique_display_ids: 85
duplicates: 0
```

Audit constraint:

```text
input_actions: 3977
normalized_actions: 2277
unique_cleared_rows: 5540
identity_rows_cleared: 0
jersey_rows_cleared: 5540
same_frame_team_jersey_conflict: 2256
unknown_owner_vs_known_owner: 21
roster_valid: true
```

Conclusione: i constraint non cancellano righe con identita' assegnata. Le
azioni osservate rimuovono jersey conflittuali da owner unknown. La diagnosi
secondo cui `global_team_jersey_owner` causava la perdita di migliaia di righe
assigned e' smentita.

### A2 - jersey identity linker conservativo

Config opt-in sperimentale:

```text
configs/identity_constraints_a2_jersey_linker_int_ata.yaml
```

Run:

```text
Int-Ata_identity_constraints_a2_jersey_linker_1200f
```

Gate testati:

```text
max_gap: 90
max_distance: 140
min_votes: 5
min_confidence: 0.20
min_head_confidence: 0.55
min_winner_margin: 0.10
```

Risultato:

```text
accepted jersey links: 0
changed_rows: 0
metriche identiche ad A1
metadata_gate_pass: true
```

Rejection summary:

```text
gap: 340
team_or_jersey: 34
distance: 4
overlap: 0
```

Gli unici gap appena oltre soglia erano:

```text
10 -> 79: gap=95, distance=225.55, team/jersey 2/#17 -> 1/#1
52 -> 79: gap=95, distance=159.82, team/jersey 2/#3  -> 1/#1
```

Entrambi sono mismatch di team e jersey; aumentare `max_gap` non produrrebbe
link. Le quattro coppie fermate dalla distanza erano a 382-1059 pixel. Non
allargare `max_distance`, non abbassare `min_votes` e non indebolire i gate
team/jersey.

Decisione identity constraints/linker:

- mantenere `jersey_identity_linking.enabled: false` nei profili promossi;
- non promuovere la config A2: e' un no-op senza beneficio;
- non modificare `global_team_jersey_owner` o il constraint same-frame;
- non modificare `frame_team_split_max_gap` per questo problema;
- baseline matched per futuri test: A1 con `assigned_rows=5426`;
- il collo di bottiglia e' upstream: `insufficient_assignment_evidence` e
  `missing_visual_similarity`, non la cancellazione dei constraint.

## Aggiornamento 2026-06-24 - Gate audit e PRTReID identity bridge

Audit per tracklet sulla A1:

```text
tracklets: 78
assigned_tracklets: 26
unknown_tracklets: 52
near_miss_tracklets: 8
tracklet visual available: 78/78
roster visual available: 0/44
roster position prior available: 22/44
```

Il gate `strong_team_visual_trajectory` e' strutturalmente morto: tutti i
tracklet hanno embedding, ma nessun giocatore roster ha un profilo visuale.
Non abbassare frame, similarity o position threshold: senza il lato roster non
cambia nessuna decisione.

I 28 tracklet unknown con best candidate locale `reliable_jersey` collidono
con un owner gia' scelto dalla Hungarian one-to-one. Non assegnarli direttamente:
alcuni gruppi, soprattutto `team1_sub_02`, contengono molti frammenti e overlap
che non possono appartenere tutti allo stesso giocatore.

Audit manuale degli 8 near-miss OCR:

```text
threshold: 0.20
accepted: 0
true positives: 0
false positives: 0
recall: 0
promotable: false
```

Decisione: non creare una A3 OCR e mantenere
`identity.reliable_jersey_min_candidate_score: 0.20`.

PRTReID identity bridge audit-only:

```text
run: Int-Ata_prtreid_identity_bridge_audit_1200f
anchors reliable_jersey: 14
candidates: 24
assigned_rows: 5426, identico alla A1
applied_rows: 0
duplicates: 0
```

Revisione manuale completa dei 24 candidati; 6 erano mutual-nearest. Soglie
zero-FP calibrate:

```text
min_similarity: 0.9794023036956787
min_margin: 0.007486522197723389
accepted: 1
true positives: 1
false positives: 0
promotable su Int-Ata: true
```

La run applicativa calibrata e' il prossimo controllo ancora da confermare:

```text
Int-Ata_prtreid_identity_bridge_calibrated_1200f
```

Promuovere solo se il delta assigned/unknown coincide esattamente con
`prtreid_identity_bridge.applied_rows`, i duplicati restano zero e il link
applicato corrisponde alla coppia manualmente approvata.

## Number Region Detector e costo Hungarian - aggiornamento 2026-06-25

Stato: implementato localmente e provato su remota `Int-Ata` 1200 frame.

File principali:

- `ft/features/number_region.py`
- `ft/identity/hungarian.py`
- `ft/pipeline.py`
- `ft/config.py`
- `ft/validation.py`
- `tests/test_number_region.py`
- `tests/test_identity.py`

Config introdotte:

```text
configs/number_region_audit_int_ata.yaml
configs/number_region_cost_int_ata.yaml
configs/number_region_cost_scene_reset_int_ata.yaml
```

### Implementazione

Pipeline number-region:

```text
player bbox -> torso_heuristic number-region crop -> MMOCR+EasyOCR
            -> tracklet evidence separata -> costo Hungarian opzionale
```

La prima versione usa `backend: torso_heuristic`; non usa VLM2, SAM2 o modelli
generativi. OCR effettivo: `mmocr+easyocr`; `pytesseract` non disponibile sulla
remota.

Artifact esportati:

```text
{video_id}_number_region_detections.csv/json
{video_id}_number_region_tracklet_evidence.csv/json
```

La sorgente e' configurabile:

```yaml
number_region:
  enabled: true
  audit_only: true
  backend: torso_heuristic
  frame_interval: 5
```

### Run audit-only

Run:

```text
Int-Ata_number_region_audit_1200f
```

Output:

```text
player_rows: 15119
assigned_rows: 5426
unknown_rows: 9693
unique_display_ids: 85
scene_cuts: [294, 517, 715, 925, 1052]
sampled_player_rows: 2998
localized_regions: 2998
usable_region_crops: 2995
recognized_regions: 2050
tracklet_evidence_rows: 77
identity_mutations: 0
legacy_jersey_mutations: 0
```

Distribuzione rilevante:

```text
recognized_region_values top:
#2: 325
#1: 229
#3: 210
#0: 191
#7: 171

tracklet_winners top:
#2: 16
#3: 12
#1: 12
#0: 9
#8: 5
partial_regions: 0
```

Interpretazione: il backend produce molta evidenza e preserva la baseline, ma
la dominanza di `#2` resta un rischio. L'audit-only non promuove identita'.

### Integrazione costo Hungarian

`HungarianPlayerIdentifier` ora supporta:

```yaml
identity:
  number_region_cost_enabled: false
  number_region_bonus_weight: 0.20
  number_region_mismatch_penalty: 0.20
  number_region_min_votes: 2
  number_region_min_mean_confidence: 0.20
  number_region_min_consecutive_support: 1
```

Formula concettuale:

```text
costo(tracklet, player) =
  base + team + jersey + number_region + position_prior + visual + ...
```

Importante: `number_region` entra nel costo, non nel gate autoritativo.
Il gate `assignment_gate` non assegna da solo un player solo per numero-region.

Test locale passato:

```text
test_number_region_cost_enters_hungarian_matrix_without_gate
test_summarize_attaches_number_region_evidence_by_display_id
```

### Run costo globale - scartata

Run:

```text
Int-Ata_number_region_cost_1200f
```

Risultati:

```text
assigned_rows: 4677
unknown_rows: 10442
assigned_display_ids: 20
nr_best_component_count: 58
changed_rows vs audit: 2696
changed_display_ids vs audit: 14
```

Problema:

```text
number_region_component spesso circa -0.20
candidate cost spesso 0.0
```

Decisione: non promuovere il costo globale. L'Hungarian globale fa competere
tutte le azioni insieme e un buon segnale nella prima parte puo' rubare o
bloccare identita' in altre azioni.

### Reset assegnamento per scene cut - incoraggiante

Implementato:

```yaml
identity:
  assignment_scope: scene_segment
```

Effetto: Hungarian viene risolto separatamente per ogni `scene_segment_id`.
Il vincolo one-to-one del roster vale dentro la singola azione/scena, non fra
azioni separate da cut.

ID interno usato per lo scope scena:

```text
scene_identity_tracklet_id = display_track_id * 100000 + scene_segment_id
```

Test locale passato:

```text
test_scene_segment_assignment_scope_resets_roster_competition
```

Run:

```text
Int-Ata_number_region_cost_scene_reset_1200f
```

Risultati:

```text
assigned_rows: 8526
unknown_rows: 6593
assigned_display_ids: 42
unique_display_ids: 85
scene_cuts: [294, 517, 715, 925, 1052]
nr_best_component_count: 89
changed_rows vs audit: 5181
changed_display_ids vs audit: 24
```

Interpretazione:

- il reset per taglio video funziona tecnicamente;
- il comportamento e' visivamente incoraggiante nella prima parte;
- l'aumento di assegnazioni mostra che il reset rimuove la competizione
  globale fra azioni;
- il peso `number_region_bonus_weight: 0.20` resta troppo aggressivo: molti
  candidati arrivano a costo `0.0`.

Decisione corrente: mantenere `assignment_scope: scene_segment`, ma rendere il
costo number-region piu' conservativo prima di promuovere.

Prossima config consigliata:

```yaml
base_config: number_region_cost_scene_reset_int_ata.yaml

identity:
  assignment_scope: scene_segment
  number_region_cost_enabled: true
  number_region_bonus_weight: 0.05
  number_region_mismatch_penalty: 0.0
  number_region_min_votes: 3
  number_region_min_mean_confidence: 0.50
  number_region_min_consecutive_support: 2
```

Obiettivo: usare il numero-region come tie-breaker dentro la scena, non come
segnale che azzera il costo.

### Prossimi controlli

- Revisione visiva dei 24 display IDs cambiati nella run scene-reset.
- Verificare se i cambi in scene 0 sono corretti: esempi critici ricorrenti
  `display 9 -> team1_sub_02`, `display 15 -> team2_08`,
  `display 12: team2_sub_15 -> team2_13`.
- Contare duplicati player-id same-frame dopo scene-reset.
- Confrontare `identity_evidence.assignment_gate` e `components.number_region`
  per tutte le nuove assegnazioni.
- Se la config conservativa mantiene miglioramento visivo e riduce costo `0.0`,
  preparare audit manuale stratificato per promozione.

## Note storiche - Number Region Detector audit-only

Notebook analizzato:

```text
basketball_ai_how_to_detect_track_and_identify_basketball_players.ipynb
```

Il notebook era una sorgente esplorativa esterna da circa 45 MB ed e' stato
rimosso dal worktree locale il 2026-07-14 dopo averne documentato le sole
conclusioni utili in questa sezione. Non e' una dipendenza del progetto.

Contributo utile: separare localizzazione della regione numero, associazione al
giocatore, riconoscimento e validazione temporale. Non copiare direttamente i
modelli basketball o l'assegnazione roster del notebook.

### 1. Baseline diagnostica

- Quantificare su Int-Ata quali false evidenze OCR, soprattutto `#2`, nascono
  fuori dalla reale regione numero.
- Costruire un campione manuale stratificato per team, numero, crop quality,
  assigned/unknown e tipo di errore.
- Conservare A1 come controllo: `5426 assigned`, `9693 unknown`, zero
  duplicati.

### 2. Backend FT-native di localizzazione numero

Implementare un backend disabilitato per default, in modalita' audit-only:

```text
player bbox -> number-region detection -> player/number association
            -> restricted crop -> recognizer -> temporal evidence
```

Prima versione:

- usare bbox player esistenti; non introdurre SAM2;
- rilevare una o piu' regioni numero per frame/player;
- associare regione e player tramite containment/IoS geometrico;
- scartare associazioni ambigue o con overlap insufficiente;
- applicare padding controllato e recognizer soltanto sul crop localizzato;
- mantenere separata questa evidenza dall'OCR legacy.

Interfaccia artifact minima:

```text
frame
raw_track_id
display_track_id
number_region_bbox
number_region_confidence
player_number_overlap
recognition_raw
recognition_confidence
recognition_source
consecutive_support
```

Artifact previsti:

```text
{video_id}_number_region_detections.csv/json
{video_id}_number_region_tracklet_evidence.csv/json
```

### 3. Validazione temporale

- Campionare a intervallo configurabile, inizialmente ogni 5 frame.
- Misurare supporto consecutivo dello stesso valore, ma non usarlo come unico
  gate: tre errori consecutivi possono confermare una lettura falsa.
- Combinare consecutive support, confidence, frame diversity e crop quality.
- Supportare evidenza parziale (`?7`, `1?`) senza completamento generativo.
- Mantenere la distribuzione raw prima del roster-constrained decoding.

### 4. Audit manuale e ablation

- Nessuna identita' applicata nella prima run.
- Generare contact sheet/HTML per tutte le nuove letture candidate.
- Etichettare `correct`, `wrong`, `partial`, `uncertain`.
- Confrontare contro OCR legacy a livello crop e tracklet:
  precision, coverage, false `#2`, roster-compatible accuracy e stabilita'
  temporale.
- Eseguire un'ablation con una sola nuova sorgente, propagation e jersey linker
  disabilitati.

### 5. Criteri di promozione

Promuovere la number-region evidence soltanto se:

- falsi positivi identitari: zero nel campione revisionato;
- precisione jersey superiore all'OCR legacy;
- riduzione misurabile delle false evidenze `#2`;
- nessuna regressione di player rows, scene cut o duplicati;
- ogni nuova assegnazione e' tracciabile a region bbox, recognition raw e
  supporto temporale;
- la nuova sorgente puo' essere disabilitata tornando bit-for-bit alla
  baseline matched.

### 6. Training futuro

- MNIST solo come pretraining/benchmark per singole cifre, mai come modello
  operativo diretto.
- Preferire dati sintetici football-specific: font sportivi, prospettiva,
  motion blur, pieghe, occlusioni e numeri a due cifre.
- Fine-tuning soltanto su ground truth SoccerNet-GSR o annotazioni manuali
  affidabili; non usare output FT come pseudo-ground-truth identitario.
- Per numeri incompleti produrre distribuzioni e wildcard, non ricostruzioni
  forzate.

### Non fare

- non sostituire ByteTrack con SAM2 one-shot: non gestisce bene cut e re-entry;
- non riattivare team clustering PRTReID/KMeans;
- non mappare direttamente `team + numero -> player_id` senza gate e audit;
- non usare VLM/generative completion come evidenza hard;
- non rilassare constraint o Hungarian per compensare errori OCR.

## Decisione corrente

Profilo operativo da preservare:

```text
Int-Ata_number_region_scene_reset_control_1200f
configs/best_known_int_ata_number_region_scene_reset_control.yaml
best_yolo26x_gsr_light.pt
```

Stato operativo:

- Baseline operativa Int-Ata: `assigned_rate=58.4%`, zero duplicati.
- Baseline sperimentale GSR detection/tracking: confidence YOLO `0.12`, config
  `configs/gsr_detection_tracking_conf012_v1.yaml`; validata sul pilot
  validation da 12 sequenze, non ancora default generale.
- Il benchmark GSR mostra che il collo di bottiglia tracking e' association
  (`AssA circa 0.517`); il prossimo audit deve spiegare perche' il candidato
  corretto manca in 22 target su 50, attribuendo gap/distance/team/appearance/
  overlap/tracklet corto.
- TVCalib e' scartata dal percorso operativo: geometria accurata (`1.10 m`
  mediana, `2.71 m` P90), ma gate rigido e reranker morbido non migliorano il
  tracking end-to-end. Lasciare TVCalib, pitch gate e pitch reranking
  disabilitati; non eseguire altri sweep sul pilot12.
- Number-region/scene-reset resta il profilo migliore osservato, ma non va
  promosso oltre Int-Ata senza benchmark manuale.
- Identity Benchmark V1 e' implementato: usare manifest/HTML/evaluator per
  promuovere modifiche identity, non solo assigned rows.
- La normalizzazione `identity_status='' -> unknown` e' fixata e verificata.
- Activity segmentation non e' promossa: i boundary da detection-count sono
  troppo rumorosi per azioni/reset.
- YOLO person+ball full20e e' migliore su validation GSR, ma peggiora la
  pipeline identity su Int-Ata: non usarlo come modello operativo.
- PRTReID resta contesto storico/opt-in sperimentale; non riattivarlo in run
  operative senza una nuova ablation benchmarkata.
- Il filtro jersey main-subject PRTReID e' respinto dal percorso candidato:
  125 outlier su SNGS-023 ma nessun guadagno accuracy track/frame; lasciarlo
  disabilitato per costo e ridondanza con il selettore ResNet34.
- `same_scene`, `identity_propagation`, `jersey_identity_linking`,
  PRT bridge/linking e number-region cost aggressivo devono restare disabilitati
  nel profilo operativo.
- Non promuovere number-region come evidenza hard finche' non supera revisione
  manuale zero-FP; usarlo solo come segnale controllato/auditabile.
- Non usare checkpoint YOLO fine-tuned come default solo perche' hanno mAP
  migliore: serve test end-to-end su tracking/OCR/identity.
- Non usare questo profilo come default generale: Inter-Juve non aveva veri cut
  nella clip testata; Inter-Atalanta/Inter-Juve non hanno ancora validato
  generalizzazione cross-scene.
- La nuova legibility policy migliora OCR jersey su cinque sequenze GSR
  validation, ma resta audit offline: non e' ancora parte del profilo operativo.
- Candidata legibility bilanciata: top5, gap5, score>=0.50, almeno 2 voti,
  margin 0, solo player, gate ruolo per frame. Variante strict: margin 0.10.
- Il consolidamento prefix conservativo `weight=0.25`, `min_long_votes=3` e'
  validato su 12 sequenze nuove: 93->99 track corrette, 61->55 errate,
  frame corretti +1480, coverage invariata e zero regressioni corretto->errato.
  Usare solo `configs/gsr_prefix_consolidation_votes3_propose_v1.yaml`.
- Prossimo controllo legibility/prefix: baseline vs prefix-only sullo split GSR
  `test` congelato, una sola volta e senza tuning. Non includere subject PRTReID.
- Il refactoring di `ft/pipeline.py` e' solo alla prima tranche; prima di altre
  estrazioni confermare lo smoke end-to-end remoto e poi separare persistenza
  artifact ed evidence/identity in passaggi incrementali.
- Per la generalizzazione identity cross-scene resta necessario un secondo
  video reale con cut, audit manuale indipendente e calibrazione
  leave-one-video-out; GSR valida jersey/tracking ma non sostituisce quel test
  roster/player-name.

## Aggiornamento 2026-07-21 - Region CTC SJN->GSR e ablation multi-frame

### Candidato numerico confermato su GSR test

Pipeline valutata:

```text
frame selezionato
-> YOLO26s regione numero GSR
-> padding 0.25
-> ResNet18-BiLSTM CTC preaddestrato SJN-210k
-> fine-tuning GSR
-> aggregazione probabilistica track-level
```

Checkpoint congelati:

```text
models/jersey_number_ctc_sjn_to_gsr_v1.pth
sha256 11c6356895c94e734b795d9ca2403580187d1d726c5aada51273738042c7c1c5

runs/jersey_number_region/yolo26s_smoke_v1/weights/best.pt
sha256 ec3c4d06df75aff92f49779827e6cff96d99b84da496ced8a76ecfbb12c02ae4
```

GSR test ufficiale, singola esecuzione congelata:

```text
sequenze:                  49
tracklet:                 774
crop selezionati:        2954
regioni rilevate:        1803

baseline CTC corretti:    249
SJN->GSR CTC corretti:    442
accuracy all:          32.17% -> 57.11%
accuracy assigned:     41.16% -> 73.06%
top-5 GT rate:         49.87% -> 67.31%
coverage:              78.17% -> 78.17%
wrong->correct:           206
correct->wrong:            13
guadagno netto:           193
paired exact p:          7.54e-46
```

Bootstrap sequence-level, 10000 repliche:

```text
delta accuracy medio: +0.2492, IC95% [0.2222, 0.2776]
delta top-5 medio:    +0.1744, IC95% [0.1480, 0.2022]
```

Non rieseguire o ritoccare GSR test. Non usare il test per nuove soglie.

### Integrazione runtime audit-only

Configurazioni:

```text
configs/gsr_jersey_region_ctc_sjn_to_gsr_audit_v1.yaml
configs/int_juve_jersey_region_ctc_sjn_to_gsr_audit_v1.yaml
```

Vincoli:

```text
mode: audit
audit_only: true
fusion_preview_enabled: false
detector_confidence: 0.25
box_padding: 0.25
max_crops_per_tracklet: 5
min_frame_gap: 5
```

Smoke SNGS-025: tracklets control/audit strutturalmente identiche. L'unica
differenza CSV era `crop_path`, dovuta alle directory artifact diverse;
tracking, jersey e identita' avevano zero differenze. `identity_mutations=0`.

Inter-Juve 1500 frame:

```text
crop selezionati:          201
regioni rilevate:           16
region detection coverage: 7.96%
track con CTC:               7
accordi SAR/CTC:             3
disaccordi:                  4
identity mutations:          0
```

Limite principale cross-domain: copertura del detector, non costo CTC.
Caso qualitativo track 42: numero visivo `95`, SAR `2`, CTC top-1 `55` su
quattro frame e `95` presente nel top-5. Il CTC confonde sistematicamente la
prima cifra `9->5` ed e' overconfident; non abilitare override automatici.

### Ablation multi-frame con nuove teste visuali

Architettura sperimentale:

```text
ResNet18 encoder CTC
-> attention frame-level
-> teste length/tens/units
```

Checkpoint ablation:

```text
models/jersey_multiframe_digits_resnet18_v1.pth
sha256 b743aba7b470b19d83ad90b4a74197f8e1cd88b6b57e580fec8eff8187164160
```

Risultati:

```text
crop readable validation:       55.74% vs CTC 8.20%
regioni manuali validation:     30.43% vs CTC 86.96%
correct->wrong regioni manuali: 26
wrong->correct regioni manuali:  0
ECE regioni manuali:            0.377
```

Decisione: ablation negativa per domain/surface overfitting. Non usare su
frozen, test o runtime. Le teste casuali hanno perso BiLSTM e classificatore
CTC gia' addestrati; 225 track non bastano per reimparare il riconoscimento.

### Ablation fusione digit-level dei logits CTC

Il fusore fattorizza la distribuzione CTC in lunghezza, decine e unita' e la
interpola con la distribuzione originale. Sweep development `0.0..1.0`:

```text
baseline:           40/46 = 86.96%
fusion:             40/46 = 86.96%
top-5:              45/46 = 97.83%
wrong->correct:      0
correct->wrong:      0
peso selezionato:    0.0
```

Decisione: ablation neutra/no-op. Non aggiunge informazione visuale e non va
portata su frozen, test o runtime.

### Direzione successiva

Se si riprende il multi-frame visuale, preaddestrare prima le teste
`length/tens/units` sulle regioni SJN-210k e trasferire l'intero recognizer;
solo dopo fine-tuning attention su bag GSR. Evitare nuove teste casuali su sole
225 track e non riordinare ulteriormente gli stessi logits CTC.

Worktree locale pronto per pubblicazione su `main`. Artifact, dataset, video,
log e checkpoint restano esclusi dal repository.

## Aggiornamento 2026-07-22 - Coverage gap del detector regione: diagnosi e limite strutturale

### Diagnosi del gap sul frozen

Auditato il gap di coverage del detector regione (`yolo26s_smoke_v1`) sul
frozen (11 sequenze, 173 tracklet, 865 crop selezionati):

```text
region_detection_coverage @conf=0.25: 60.7%
confidence_threshold_curve:
  0.25 -> 60.7%
  0.20 -> 64.4%
  0.15 -> 68.9%
  0.10 -> 71.9%
  0.05 -> 75.0%
hard_miss_rate (zero box anche a conf=0.05): 25.0%
fully_blind_tracklets: 12/173
```

Scomposizione dell'hard-miss con il classificatore `clean_back`
(`jersey_back_yolo11s_cls_v1.pt`, uso diagnostico):

```text
pose_explained_not_clean_back: 78.2% (169/216)
unexplained_clean_back_still_missed: 21.8% (47/216)
```

Nuovo segnale più forte, orientamento di schiena via pose estimator generico
(`yolov8x-pose.pt`, regola geometrica su keypoint spalle sinistra/destra COCO,
zero training):

```text
pose_detected: 763/865 (88%) — keypoint affidabili anche su crop 25-70px
coverage crop orientation=back:  81.0% (410/506)
coverage crop orientation=front: 29.0% (71/245)
```

Script: `scripts/audit_jersey_number_region_detector_coverage.py`,
`scripts/analyze_number_region_coverage_csv.py`,
`scripts/inspect_jersey_ocr_tracklet_diagnostics.py`,
`scripts/audit_hard_miss_crop_pose.py`,
`scripts/audit_crop_pose_orientation.py`,
`scripts/aggregate_number_region_coverage_by_sequence.py`.

### Re-selezione frame: due tentativi, entrambi falliti

Ipotesi: se il frame selezionato non mostra la schiena, ri-selezionare fra i
20 candidati grezzi già su disco (non serve nuova estrazione) un frame
migliore recupererebbe i tracklet ciechi.

```text
Re-selezione per clean_back_probability: 0/12 tracklet recuperati
Re-selezione per pose_back_score (piu' discriminante in aggregato): 0/12 recuperati
```

Dettaglio per i 12 tracklet ciechi (pose):

```text
7 tracklet: zero candidati "back" in tutti i 20 frame campionati
  (068::10, 068::13, 068::16, 068::5, 097::14, 098::7, 170::16)
  -> assenza strutturale nel campione, non risolvibile da nessun ranking
5 tracklet: almeno 1 candidato "back" confermato (fino a 6/20 per due
  tracklet, confidence spalle 0.93-0.99), detector fallisce comunque su tutti
  (099::3, 165::17, 068::14, 170::6, 098::11)
  -> vero limite del detector, non un problema di selezione frame
```

Script: `scripts/simulate_clean_back_reselection.py`,
`scripts/simulate_pose_reselection.py`.

Conclusione: la selezione dei frame NON è il collo di bottiglia dominante.

### Retraining del detector: tre varianti, nessuna promossa

Scoperto che il checkpoint operativo (`yolo26s_smoke_v1`) era stato
addestrato per sole 10 epoche (mAP50-95 ancora in salita all'ultima epoca).
Tre esperimenti, stesso dataset `gsr_train_manual_v1` (435 immagini, 337
positive, 37 sequenze; frozen mai incluso, verificato), stesso seed
`20260719`:

```text
                          coverage@0.25  hard_miss  fully_blind  coverage@0.05
smoke (10 epoche)              60.7%       25.0%       12/173        75.0%
full GSR-only (43 epoche)      62.8%       27.2%       13/173        72.8%
SJN->GSR (25 epoche, pretrain  63.7%       28.1%       13/173        71.9%
  esterno SJN-210k gia' pronto,
  runs/jersey_number_region/yolo26s_sjn210k_full_v1)
```

Checkpoint valutati:

```text
runs/jersey_number_region/yolo26s_full_v1/weights/best.pt
sha256 a268eb047dc2be307ec35dae99409315ddc8f631d76f663d5c62448241f9ccee

runs/jersey_number_region/yolo26s_sjn_to_gsr_full_v1/weights/best.pt
sha256 1b050f3fac085d77eb669f9d93907df2c0698fa163a8e2224bcb723c1c152f16
```

Pattern coerente su tutti e tre: piu' training (piu' epoche, poi pretraining
esterno su 8938 regioni SJN-210k) sposta la coverage al threshold operativo
di pochi punti (+3 totali), ma peggiora sistematicamente hard-miss e coverage
alle soglie basse. Il detector diventa piu' "sicuro" quando rileva, non piu'
capace sui casi difficili.

**Decisione**: nessuno dei tre checkpoint promosso. Il collo di bottiglia non
e' la procedura di training (epoche, pretraining esterno) ne' la selezione
frame (pose o classificatore): e' strutturale — le 435 immagini manuali GSR
non coprono la varieta' visiva delle sequenze frozen, e nessun retraining
sulle stesse 435 immagini la recupera. Prossimi passi possibili, non
eseguiti: piu' dati annotati nel dominio target (richiede annotazione
manuale), o cambio architetturale (risoluzione input maggiore,
super-risoluzione dei crop prima del recognizer, detector per oggetti molto
piccoli). Restare su `yolo26s_smoke_v1` come operativo.

Nota operativa: training su questa GPU (RTX 2080, 7.8GB) e' andato in crash
CUDA "illegal memory access" transitorio due volte durante la validazione a
batch=16 (non deterministico, non riproducibile a batch=8). Se ricapita,
riprovare con `--batch 8` prima di altre ipotesi.

## Aggiornamento 2026-07-22 (2) - Roster-reranking implementato; posa come detector bocciata

### Roster-constrained re-ranking del CTC: successo, implementato

Osservazione di partenza: il CTC a volte legge con alta confidenza una cifra
sbagliata quando la vera cifra e' visivamente simile (caso noto Inter-Juve:
`95` letto come `55`), con la cifra corretta comunque presente nel proprio
top-5. Il modulo CTC-regione non consultava mai il roster prima d'ora.

Misura del tetto teorico su frozen con un roster oracolo (costruito dai
`gt_jersey_number` reali per sequenza+lato, solo a scopo diagnostico — GSR
non ha un roster reale in produzione e non va trattato come se lo avesse):

```text
run: ctc_sjn_transfer_frozen_v1/sjn_to_gsr (+ --dump-raw-scores, output
     identico al run originale: correct=124, wrong=35 -> verificato bit-a-bit)
accuracy_assigned: 77.99% -> 84.91% (+6.9 punti)
transitions: correct_to_correct=124, correct_to_wrong=0,
             wrong_to_correct=11, wrong_to_wrong=24
net_gain: +11, changed_rows: 19
```

Zero regressioni (`correct_to_wrong=0`) — il criterio di sicurezza chiave e'
soddisfatto. Script: `scripts/audit_roster_oracle_reranking.py` (usa
`aggregate_frames` di `ft/features/jersey_number_ctc.py`, puro Python, gira
anche in locale senza GPU).

**Implementato** in `ft/features/jersey_region_ctc_audit.py`:
`JerseyRegionCTCAuditor` accetta ora `roster`, `roster_reranking_enabled`,
`roster_reranking_top_k`. Calcola un campo `roster_rerank_preview` per
tracklet (audit-only: non scrive mai in `jersey_number`/`tracklets`,
`identity_mutations` resta 0). Roster costruito da `ft.identity.roster` via
`team_id`/`jersey_number` (stesso formato gia' usato da Hungarian). Team per
display_track_id derivato da `player_rows` (majority vote), passato a
`.run(..., player_rows=player_rows)`. Wiring in `ft/pipeline.py`. Test
unitari in `tests/test_jersey_region_ctc_audit.py` (7 nuovi test). Config di
prova: `configs/int_ata_jersey_region_ctc_roster_rerank_audit_v1.yaml`.

Test di integrazione sulla pipeline reale (Int-Ata, 200 frame, roster
sintetico `SNGS-151_synthetic_roster.json` solo per verifica di cablaggio,
non risultato scientifico):

```text
roster_teams_loaded: [1, 2]
roster_reranking_summary: tracklets=9, changed=5
identity_mutations: 0
```

Meccanismo verificato: gira nella pipeline reale senza errori, campi
popolati correttamente, zero mutazioni. Il tasso di cambio alto (5/9) e'
atteso: il roster sintetico non corrisponde ai giocatori reali di Int-Ata.

**Nessun roster reale disponibile per Int-Ata o altri video custom** (solo
roster sintetici GSR in `evaluation/soccernet_gsr/rosters/` e l'esempio
generico `configs/roster.example.json`). Prossimo passo prima di poter
validare il guadagno reale: reperire le formazioni reali della partita.

### Posa come sostituto del detector regione: bocciata

Ipotesi: dato che l'orientamento da posa correla molto piu' fortemente con
il successo del detector (81% back vs 29% front, vedi sopra) di qualunque
segnale appreso, e la letteratura pubblica (Liu & Bhanu 2019; Koshkina et
al. 2024, crop guidato da ViTPose) deriva il box torso/numero direttamente
dai keypoint invece di addestrare un detector — provato a sostituire
interamente il detector YOLO con una regola geometrica sui keypoint
(spalle/fianchi COCO), stesso identico CTC a valle, stesso `ocr-run` frozen
per confronto diretto.

```text
                          coverage   accuracy_assigned  accuracy_all  gt_in_top5
YOLO detector (sjn_to_gsr)   81.6%        77.99%           71.68%       ~84%
Pose geometria (torso)       90.0%        35.6%            32.9%        56.1%
```

La geometria da posa trova una regione piu' spesso (+8.4 coverage) ma il
recognizer legge malissimo quello che trova (-42 punti accuracy_assigned).
Causa probabile: il rettangolo spalle->fianchi e' troppo largo/impreciso
rispetto al crop stretto sul solo numero su cui il CTC e' stato addestrato
— problema di framing/scala, non di localizzazione generica. Script:
`scripts/evaluate_pose_derived_region_ctc.py`.

**Decisione**: non sostituire il detector con la geometria pura. L'uso
valido della posa resta quello di stamattina — segnale di orientamento per
la selezione frame, non generatore diretto del box. Non ritentare aggiustare
le proporzioni geometriche (`top_fraction`/`bottom_fraction`/
`side_padding_fraction`) sul frozen, gia' molto usato oggi (7-8 valutazioni
diverse) — se ripreso, tarare su development.

### Stato di fine giornata

- Roster-reranking: implementato, testato, in attesa di un roster reale per
  la validazione su video custom. Prossima azione: procurarsi le formazioni
  reali di Int-Ata (o altro video con roster disponibile).
- Miglioramento del detector regione: esplorato ampiamente (epoche,
  pretraining esterno, re-selezione frame per classificatore e per posa,
  sostituzione con geometria), nessuna soluzione a basso costo trovata.
  Restare su `yolo26s_smoke_v1`. Riprendere solo con investimento serio
  (dataset esterno con diversita' di broadcast validata, es. Hockey JNR
  Dataset di Koshkina — transfer zero-shot verso il calcio validato in
  letteratura all'87.4%; oppure annotazione mirata nel dominio target).
- GTA (Global Tracklet Association, post-processing tracking model-agnostic)
  proposto per il bottleneck AssA del tracking, mai testato: prossima pista
  indipendente da detector/jersey.

## Aggiornamento 2026-07-22 (3) - Roster-reranking validato su ground truth reale (Identity Benchmark V1)

Seguito diretto della sezione precedente: procurato il roster reale di
Int-Ata (gia' presente su disco, `costume-video/Int-Ata/Int-Ata.json`, mai
cercato a fondo prima), etichettate a mano le identita' con l'infrastruttura
gia' esistente "Identity Benchmark V1" (`ft/evaluation/identity_benchmark.py`,
`scripts/build_identity_benchmark.py`, `scripts/evaluate_identity_benchmark.py`,
spec in `evaluation/identity_benchmark_v1.yaml`), e misurato il guadagno
reale del roster-reranking — non piu' solo l'oracle sintetico su GSR.

**Scoperta infrastruttura pre-esistente**: il progetto ha gia' un sistema
completo di annotazione/adjudication a doppio revisore per identita' di
tracklet (unit singole + coppie same/different), con split
development/test/external gia' dichiarato per i tre video custom (Int-Ata,
Inter-Juve, Inter-Atalanta) e roster reali dichiarati per spec. Prima di
costruire qualunque tooling nuovo per l'etichettatura richiesta dall'utente,
va sempre controllato se esiste gia' — qui c'era, completo e mai citato
esplicitamente in precedenza in questo handoff.

**Bug scoperto e risolto: crop mancanti per pulizia disco pregressa**. Le
run di produzione usate come `source_runs` nello spec (es.
`Int-Ata_prtreid_conservative_final_1200f`,
`Int-Ata_identity_evidence_v1_resetbytetrack_cutsensitive_1200f`, e le
equivalenti per Inter-Atalanta/Inter-Juve) avevano le cartelle `crops/`
intere cancellate manualmente per liberare spazio disco, pur avendo
`metadata/*_tracklets.csv` (con bbox e frame per ogni crop) intatto. La
build del benchmark falliva silenziosamente producendo migliaia di tile
nere nelle contact sheet (nessuna eccezione Python, solo warning OpenCV
`imread` a livello C++). Soluzione: nuovo script
`scripts/recrop_from_tracklets.py` che rilegge `tracklets.csv` +
`run_manifest.json` (per il `video_path` originale), riapre il video
sorgente, e ri-ritaglia solo i crop mancanti agli stessi bbox/frame/percorso
gia' registrati — zero re-detection/re-tracking/re-identity, solo il
ritaglio. Rieseguito per tutte e 4 le run necessarie (Int-Ata x2,
Inter-Atalanta, Inter-Juve x2), ~68mila crop rigenerati in totale in pochi
minuti (nessuna GPU necessaria). Nota per il futuro: se si ripulisce lo
spazio disco cancellando `crops/`, farlo solo su run che non serviranno mai
piu' a costruire contact sheet/benchmark, o tenere `tracklets.csv` per poter
rigenerare on-demand con questo script.

**Etichettatura**: 202 item totali (159 unit identita' + 43 coppie),
etichettati a mano da un solo revisore via `annotate_reviewer_a.html`
(nessuna doppia revisione/adjudication vera, dato revisore singolo:
`reviewer_b.csv` duplicato identico a `reviewer_a.csv` cosi'
`adjudicate_identity_benchmark.py` produce comunque un `ground_truth.csv`
completo per costruzione, accordo 100% per definizione sulle righe che
richiedono seconda revisione).

**`evaluate_identity_benchmark.py` non e' lo strumento giusto per misurare
il roster-reranking**: quello script valuta la qualita' dell'assegnazione
di identita' (player_id/tracklet), non la lettura del numero di maglia.
Dato che il roster-reranking e' audit-only (`identity_mutations=0`, non
tocca mai l'assegnazione di identita'), baseline e candidate risultano
identici byte per byte su ogni metrica — comportamento atteso, non un bug.
Serve invece confrontare direttamente `gt_jersey_number` (dal ground truth
appena etichettato) contro `original_number`/`preview_number` del
`roster_rerank_preview` nell'audit della run candidata. Nuovo script:
`scripts/evaluate_roster_rerank_ground_truth.py`.

**Bug scoperto durante questa valutazione: riuso di `display_track_id` fra
segmenti di scena diversi**. La run usa `resetbytetrack` (reset degli ID di
tracking a ogni cambio di scena), quindi lo stesso `display_track_id`
grezzo puo' rappresentare giocatori fisici diversi in segmenti di scena
diversi — ma `roster_rerank_preview` nell'audit e' indicizzato solo per
`display_track_id`, senza distinguere il segmento di scena. Nel join con il
ground truth, questo produceva lo stesso ID (es. `"4"`) associato a 6 unita'
di identita' diverse con `gt_jersey_number` diversi (1, 22, 8, 8, 29, 19) ma
con la stessa identica predizione — impossibile per un giocatore reale, e
un artefatto puro di join, non un errore del modello. Fix nello script di
valutazione: individua i `display_track_id` con piu' di un
`gt_jersey_number` distinto associato ed esclude tutte le loro righe dal
computo (18 righe su 7 ID esclusi, tracciati in
`ambiguous_display_track_ids_excluded`). Limite noto da tenere a mente se si
vuole rendere questa valutazione riutilizzabile su scala: la chiave giusta
sarebbe `(display_track_id, scene_segment_id)`, non solo `display_track_id`.

**Risultato reale (non oracle) su Int-Ata, 27 tracklet non ambigue,
etichettate a mano**:

```text
original_accuracy_assigned:   59.3% (16/27)
reranked_accuracy_assigned:   77.8% (21/27)
transitions: correct_to_correct=16 correct_to_wrong=0 wrong_to_correct=5 wrong_to_wrong=6
net_gain: +5
changed_rows: 7 (5 hanno corretto un errore, 2 rimasti sbagliati con un numero diverso)
```

+18.5pp di accuratezza, **zero regressioni** (`correct_to_wrong=0`),
coerente in direzione con il +6.9pp misurato sull'oracle GSR frozen — qui
persino piu' marcato, anche se il campione e' piccolo (n=27) e l'intervallo
di confidenza sarebbe ampio. Prima validazione genuina (non sintetica) del
guadagno del roster-reranking su un video custom con roster reale.

Roster-reranking resta **audit-only** (nessuna mutazione applicata alla
pipeline reale) nonostante questo risultato positivo, in linea con la
disciplina propose->apply del progetto: la promozione ad "apply" andrebbe
decisa solo dopo aver esteso l'etichettatura (campione piu' grande, magari
anche su Inter-Juve/Inter-Atalanta) e aver sistemato la disambiguazione per
`scene_segment_id`.

## Aggiornamento 2026-07-22 (4) - Revisione end-to-end della pipeline identity/jersey: 4 bug corretti, disambiguazione scene-aware validata

Richiesta dall'utente: revisione sistematica di tutta la pipeline
identity/jersey per trovare bug che intaccano il lavoro del progetto, non
solo il fix di oggi. Coperti: CTC/aggregazione numeri, selezione frame,
prefix consolidation, jersey identity linker, identity graph/propagazione,
roster, roster-aware OCR, candidati non autoritativi, evidence, gate audit,
Hungarian, l'intero `constraints.py`, gran parte di `jersey_ocr.py`, team/
goalkeeper/referee assignment.

**Fix scene-aware nella region-CTC audit** (root cause del problema di
disambiguazione descritto nella sezione precedente): `roster_rerank_preview`
e l'aggregazione dei punteggi CTC in `JerseyRegionCTCAuditor.run()` erano
indicizzate solo per `display_track_id` grezzo, fondendo l'evidenza di
giocatori fisici diversi riusati sullo stesso ID dopo un reset di scena.
Nuova chiave composta `tracklet_key(display_track_id, scene_segment_id)` in
`ft/features/jersey_region_ctc_audit.py`, con fallback al solo
`display_track_id` quando `scene_segment_id` non e' disponibile (preserva il
comportamento nei test esistenti). `team_by_display_track_id` e
`collect_selected_crops` aggiornate di conseguenza; il confronto con
`primary` (meccanismo di fusion_preview preesistente, non scene-aware a
monte) resta volutamente indicizzato per ID grezzo per non regredire quel
percorso.

**4 bug reali trovati e corretti** durante la revisione (tutti con test di
regressione, 331 test locali passano, nessuna regressione):

1. Il fix scene-aware sopra descritto.
2. `player_jersey_from_id` (`ft/identity/identity_graph.py`) indovinava il
   numero di maglia dal suffisso del `player_id` (es. `team1_09` -> 9) come
   fallback quando l'OCR abbandona. Funziona per roster sintetici
   (`team1_07`->jersey 7) ma e' rotto per il roster reale di Int-Ata
   (`team1_05` -> Pavard, jersey vero 28). Rimossa la funzione; `build_nodes`
   in `ft/identity/propagation.py` ora riceve `roster_by_player_id` e usa il
   vero `jersey_number` del roster come fallback.
3. `goalkeeper_number_one_prior` (`ft/identity/hungarian.py`) veniva
   saltato silenziosamente ogni volta che un `jersey_score` da distribuzione
   candidati veniva trovato -- il caso comune in produzione, dato che il
   numero dominante di una tracklet appare quasi sempre nella sua stessa
   `raw_jersey_distribution`. L'unico test esistente non copriva questo
   caso. Spostata la logica fuori dal ramo if/elif cosi' si applica sempre
   quando il numero osservato e' 1.
4. `summarize_samples` (`ft/features/referee.py`) contava tutti i campioni
   con un colore riconosciuto (`len(valid)`) invece di solo quelli del
   colore vincente (`len(values)`) per il gate `min_tracklet_frames` -
   indebolendo la protezione contro falsi positivi arbitro descritta nel
   commento della funzione stessa. Nessun test esisteva per questa funzione
   (ne' per `RefereeAppearanceAssigner` in generale). Confrontata con la
   funzione gemella corretta in `goalkeeper.py`, gia' testata.

Inoltre: consolidata una duplicazione (`roster_numbers_by_team` gia'
esistente in `ft/identity/roster.py`, ridefinita per errore oggi in
`jersey_region_ctc_audit.py`). Segnalazioni minori senza impatto reale
(codice morto in `conflict_runs`, un `cost_margin` diagnostico in
`gate_audit.py` potenzialmente fuorviante se il candidato selezionato non e'
il piu' economico) annotate ma non corrette. Un presunto bug nel gate
squadra di `jersey_identity_linker.py` (bypass a bassa confidenza) e' stato
inizialmente segnalato ma poi ritrattato dopo aver trovato lo stesso pattern
deliberato, simmetrico e testato in `ft/linking/tracklet_linker.py`
(`test_linker_team_gate_blocks_confident_team_mismatch`).

**Validazione su GPU**: rilanciata la pipeline reale su Int-Ata
(`int_ata_all_fixes_verify_1200f`) con tutti i fix applicati. Risultato
dell'audit: `tracklets=43 roster_reranking_changed=11 identity_mutations=0`
(prima del fix scene-aware: `tracklets=35 changed=7`). Ripetuta la
valutazione contro il ground truth etichettato a mano
(`scripts/evaluate_roster_rerank_ground_truth.py`, aggiornato per costruire
la stessa chiave composta `display_track_id#scene_segment_id` usando
`ft.features.jersey_region_ctc_audit.tracklet_key`):

```text
ambiguous_display_track_ids_excluded: [] (prima: 7 ID, 18 righe escluse)
assigned_tracks_evaluated: 36 (prima: 27)
original_accuracy_assigned: 61.1% (22/36)
reranked_accuracy_assigned: 72.2% (26/36)
transitions: correct_to_correct=21 correct_to_wrong=1 wrong_to_correct=5 wrong_to_wrong=9
net_gain: +4
```

Zero track esclusi come ambigui (prima: 7 ID / 18 righe) - la disambiguazione
per segmento di scena funziona come previsto. Il guadagno netto scende
leggermente rispetto alla misura precedente non disambiguata (+4 invece di
+5) perche' ora c'e' una vera regressione visibile (`correct_to_wrong=1`,
track `4#5`: GT=19, originariamente letto correttamente 19, il reranking lo
cambia erroneamente in 10) - un limite reale del meccanismo di roster
reranking su top-k, non un artefatto di misura. Il quadro complessivo
(+11.1pp, guadagno netto positivo, singola regressione isolata) resta
comunque un miglioramento validato su dati reali etichettati a mano, ora con
una metodologia di valutazione corretta e priva di ambiguita' di conteggio.

## Aggiornamento 2026-07-22 (5) - Il reset scena e' strutturalmente irrilevante per l'AssA misurato su GSR

Richiesta dell'utente: capire se ByteTrack/il reset scena si possono
migliorare, partendo dal collo di bottiglia gia' noto (`AssA=0.5167` sul
benchmark GSR a 12 sequenze). Prima di toccare qualunque soglia, verificato
se il reset scena si attiva davvero sulle sequenze GSR usate per misurare
quel numero.

**Scoperta chiave**: la config del benchmark GSR
(`gsr_detection_tracking_benchmark_v1.yaml` -> `gsr_evaluation_control.yaml`)
eredita le impostazioni `scene_cuts` dalla config tarata specificamente per
Int-Ata (`best_known_int_ata_number_region_scene_reset_control.yaml`, soglia
sensibile 0.08/0.12/min_gap 40) - mai isolate/validate per GSR
specificamente. Nuovo script diagnostico
`scripts/audit_scene_cuts_on_sequences.py` (legge frame da video O da
cartelle `img1/` in stile GSR/MOT, zero GPU, solo istogrammi HSV) applicato a
due sequenze del pilota a 12 (SNGS-025, SNGS-036, 750 frame ciascuna):

```text
SNGS-025: cuts_found=0, punteggio massimo=0.0190 (soglia=0.08, hard=0.12)
SNGS-036: cuts_found=0, punteggio massimo=0.0162
```

Il punteggio massimo raggiunto in entrambe le sequenze e' 4-5x sotto la
soglia attuale, con correlazione tra istogrammi ancora altissima (0.98-0.99)
anche ai picchi. Non c'e' un problema di taratura soglia da correggere: le
clip GSR (azione continua a camera singola, ~30 secondi, nessun replay o
cambio di camera broadcast) semplicemente non contengono il fenomeno
(discontinuita' visiva forte) che il rilevatore e' progettato a trovare.
Abbassare la soglia inviterebbe solo falsi positivi su rumore in-game
(panning, giocatori in movimento), non rivelerebbe segnale reale.

**Conclusione**: il reset scena e' completamente dormiente sul benchmark
GSR (stesso comportamento gia' documentato per Inter-Atalanta con le
impostazioni di default) - il valore `AssA=0.5167` gia' misurato e'
prodotto interamente da ByteTrack puro (IoU+moto), senza alcun contributo
positivo o negativo del reset scena. Qualunque tentativo di migliorare
quel numero specifico agendo sulla soglia/logica di reset scena e'
strutturalmente destinato a non avere effetto, perche' il meccanismo non
scatta mai su questo dominio. Il reset scena resta rilevante solo per video
broadcast multi-camera reali (Int-Ata, Inter-Juve, dove i tagli esistono
davvero) - li' pero' non esiste un ground truth di tracking frame-per-frame
(HOTA/AssA), solo l'Identity Benchmark V1 appena costruito (etichette di
identita'/jersey, non di bounding box/tracking). Per migliorare l'AssA
misurato su GSR serve una leva diversa dal reset scena - GTA (Global
Tracklet Association) resta la pista piu' diretta, gia' proposta e mai
testata.

## Aggiornamento 2026-07-22 (6) - Analisi diretta della frammentazione ID grezzi su SNGS-025 (no reset scena coinvolto)

Seguito diretto: dato che il reset scena e' irrilevante per GSR, spostata
l'analisi su quanti ID viene formulati da ByteTrack contro il ground truth,
usando artifact gia' esistenti (run `gsr_valid_pilot12_baseline_v1`,
nessuna nuova computazione GPU).

**Conteggio grezzo**: SNGS-025 (748 frame, azione continua, zero tagli
scena). Ground truth: 25 oggetti distinti (~20 giocatori + 2 portieri + 2
arbitri + 1 palla, da `Labels-GameState.json`). Pipeline: 41
`raw_track_id`, 46 `display_track_id`. Per ruolo, i nostri raw:
`{player: 35, goalkeeper: 1, referee: 5}` - notare che gli **arbitri sono
la categoria proporzionalmente piu' frammentata** (5 raw contro ~2 reali,
~2.5x) contro ~1.75x per i giocatori, e un portiere manca del tutto (quasi
certamente mal classificato come "player" dal classificatore di colore).

**`display_track_id` (46) > `raw_track_id` (41)**: controintuitivo, dato
che il linking dovrebbe ridurre i frammenti fondendoli, non aumentarli.
Causa: `_split_persistent_frame_team_conflicts` in `constraints.py`
(`frame_team_split_enabled=True` di default, nessuna config della catena lo
disattiva) ha creato 7 nuovi display id da conflitti di colore-squadra
persistenti. Verificato per tutti e 7 gli split: traiettoria posizionale
completamente fluida (salto massimo frame-a-frame 1-26px, mai una
discontinuita'). Questo esclude uno scambio silenzioso di identita' da
parte di ByteTrack in tutti i 7 casi - il pattern piu' probabile e' che sia
lo **stesso giocatore tracciato correttamente**, con il classificatore di
squadra che deriva (verosimilmente per cambio di luce/distanza dalla camera
mentre il giocatore attraversa il campo) fino a leggere la squadra
sbagliata in modo sostenuto (es. 231/232 frame consistenti, non rumore a
scatti). Il meccanismo di split, in questi 7 casi su SNGS-025, sembra
**auto-infliggere** frammentazione invece di correggere un vero errore.
Implicazione pratica: irrigidire le condizioni di split (piu' frame
richiesti, o migliorare la stabilita' del team classifier lungo il campo)
potrebbe ridurre falsi split senza perdere protezione reale - non ancora
tentato.

**Il gap piu' grande resta 41 raw contro 25 GT (16 in eccesso), indipendente
dal meccanismo sopra**. Analisi di durata di vita dei 41 raw track: alcuni
sono perfetti (`raw_id=3,14`: 748/748 frame, zero interruzioni), altri sono
rumore puro (`raw_id=40,41`: 2-4 frame, quasi certi falsi positivi
istantanei), la maggior parte si interrompe a meta' clip (150-290 frame) e
presumibilmente si ricollega altrove con un nuovo raw id.

**Ricerca di coppie di frammenti candidate** (fine di un raw track entro 60
frame e 150px dall'inizio di un altro, script ad-hoc via CSV, nessuna
nuova run): trovate 6 coppie candidate su 41 track totali. Due casi puliti
e ad alta confidenza:
- `15->22` (player): gap=16 frame, distanza=11px - quasi certamente lo
  stesso giocatore perso e riacquisito.
- `1->23` (referee->referee): gap=33 frame, distanza=67px - spiega in
  parte la frammentazione arbitro osservata sopra.

Un caso ambiguo e istruttivo: **4 track diversi** (9, 12, 18, 21, tutti
"player") risultano candidati per continuare come track 24, tutti vicino al
bordo destro del frame (x~1900+, larghezza probabile ~1920) in una finestra
temporale simile - verosimilmente un momento affollato vicino alla linea
laterale (rimessa/corner) con piu' giocatori reali vicini
contemporaneamente. Solo `12->24` (distanza=12px) e' verosimilmente il vero
collegamento; gli altri 3 (distanza 47-98px) sono probabilmente giocatori
diversi. Questo dimostra che un gate a sola distanza+tempo (quello gia'
usato da `TrackletLinker`) non basta a disambiguare nei momenti affollati -
serve un segnale aggiuntivo (aspetto/jersey), esattamente il tipo di
problema per cui GTA (Global Tracklet Association, gia' proposto,
mai testato) e' pensato.

**Conclusione**: `TrackletLinker` con i gate di default sembra fondere
solo ~2 coppie su 41 raw track in questa sequenza (41 raw -> ~39 componenti
pre-split, +7 split team-conflict = 46 display), nonostante esistano almeno
2-3 coppie recuperabili con alta confidenza identificabili da sola
distanza+tempo. Prossimo passo naturale, non ancora fatto: verificare se
questo pattern (frammentazione concentrata su alcuni ruoli/situazioni,
coppie recuperabili mancate dal linker, split team-conflict auto-inflitti)
si ripete sulle altre 11 sequenze del pilota, prima di investire in GTA o
in una modifica ai gate di `TrackletLinker`.

## Aggiornamento 2026-07-22 (7) - Correzione: il pitch-speed gate e' gia' stato provato e respinto (TVCalib)

Nella lista di tecniche proposte per l'associazione tracking (seguito
dell'analisi SNGS-025 sopra), avevo raccomandato come primo passo economico
di verificare/attivare il `pitch_gate` di `TrackletLinker` (blocco/rerank di
una fusione basato su velocita' implicita in m/s, usando posizione
proiettata sul campo). Verifica preliminare corretta - `pitch_gate.enabled:
false` di default, e comunque bloccato per requisito `require_source_prefix:
"tvcalib:"` mentre GSR usa la calibrazione automatica legacy
`auto:field_quad` - ma **non avevo controllato abbastanza a fondo se
questo fosse gia' stato tentato con la calibrazione buona**. Lo era, in una
sezione handoff precedente (non ancora vista/ricordata a inizio di questo
filone): TVCalib e' stata eseguita su tutte le 12 sequenze del pilota
(errore pitch mediano 1.10m, P90 2.71m, contro 35.78m/58.28m di
`auto:field_quad` - qualita' geometrica confermata buona), e usata per due
esperimenti end-to-end completi:

1. **Gate rigido di velocita'** (12-14 m/s): sembrava perfetto nell'audit
   offline (4 collegamenti sbagliati bloccati, 0 corretti bloccati), ma
   l'ablation end-to-end ha *peggiorato* HOTA/AssA/IDF1 e il tasso di
   switch. Respinto.
2. **Costo morbido di reranking sulla distanza pitch** (peso 0.10): stesso
   esito - leggero miglioramento offline (Recall@1 50%->52%), ma
   peggioramento end-to-end su HOTA/AssA. Respinto.

Decisione gia' congelata (non mia, precedente): mantenere `pitch_gate.
enabled: false`, `pitch_reranking.enabled: false`, `calibration.tvcalib.
enabled: false`; non fare altri sweep di soglia/peso sullo stesso pilota.
**Correggo quindi la mia raccomandazione di oggi**: riattivare o ritarare
il pitch-gate non e' una pista aperta, e riproverei un lavoro gia' fatto
con piu' rigore (calibrazione migliore di quella disponibile ora) e gia'
scartato con evidenza end-to-end chiara.

**Il dato piu' prezioso di quella sezione, pero', rimane rilevante e
rafforza la direzione GTA**: l'audit dei candidati reali del linker
(`scripts/audit_gsr_link_candidate_ranking.py`) ha trovato che in **22
target su 50 valutabili (44%)** il candidato corretto non esiste affatto
nel pool generato dal linker (`max_gap=90 frame`, `max_distance=160px`) -
non un problema di *ranking* dei candidati esistenti (gia' escluso da
entrambi gli esperimenti sopra), ma di **candidate generation**: la finestra
gap/distanza locale del linker attuale non arriva nemmeno a considerare il
vero collegamento in quasi meta' dei casi. Questo e' esattamente il tipo di
limite per cui un'associazione globale offline (GTA), con un pool di
candidati piu' ampio e una funzione di score diversa da un semplice gate
locale, e' pensata - rafforza la priorita' data a GTA rispetto a qualunque
ulteriore tuning di `TrackletLinker`.

**Prossimo passo**: esplorare concretamente GTA (letteratura, requisiti di
implementazione, dati di cui ha bisogno) prima di scrivere codice.

## Aggiornamento 2026-07-22 (8) - GTA: letteratura verificata, ambiente PRTReID rotto, pseudo-labeling verificato per il detector regione

**GTA (Global Tracklet Association)**, letteratura verificata (Sun et al.,
ACCV 2024W, https://arxiv.org/abs/2411.08216, codice
https://github.com/sjc042/gta-link): non un algoritmo unico ma due
componenti separati, entrambi basati su embedding d'aspetto (OSNet,
checkpoint SportsMOT), non movimento:
- **Splitter**: DBSCAN sugli embedding *dentro* una tracklet per rilevare
  se mescola due identita' diverse (parametri paper: eps=0.6,
  min_samples=10, max_k=3).
- **Connector**: fonde coppie di tracklet *diverse* per distanza coseno
  media (merge_dist_thres=0.4, min_len=100).

Rischio esplicito segnalato: il Connector e' concettualmente vicino allo
"shadow tracker" gia' respinto in passato (309 vs 45 ID switch su
SNGS-025, aspetto non discrimina bene compagni di squadra con maglie
quasi identiche). Deciso di partire solo dal **Splitter**, piu' sicuro e
mai provato in questa forma (il nostro split attuale e' basato su
colore-squadra, non aspetto generale).

**Segnale incrociato trovato prima di scrivere codice**: `prtreid_
tracklet_consistency` (gia' calcolato quando `prtreid_linking` e' attivo,
proxy semplice dello Splitter - similarita' media al prototipo) ha
segnalato `display_track_id=4` su Int-Ata (run
`Int-Ata_prtreid_conservative_final_1200f`) come la tracklet con
consistenza piu' bassa (0.8143) - **lo stesso identico track** trovato
stamattina con 6 numeri di maglia GT diversi dopo il fix scene-aware. Due
segnali indipendenti (conflitto GT, incoerenza d'aspetto) puntano allo
stesso track: buona conferma indiretta che l'idea ha fondamento sui dati
reali, prima ancora di costruire il DBSCAN vero.

**Ambiente PRTReID rotto, mai sistemato del tutto**: nessun ambiente conda
dedicato esisteva; clonato `jersey-yolo-ocr` in `prtreid-ft` per isolare il
lavoro. `pip install monai` senza vincoli ha aggiornato in cascata anche
torch (2.0.0+cu117 -> 2.13.0) e setuptools (rimuovendo `pkg_resources`,
necessario a `prtreid`) - ambiente ricreato da zero con
`pip install --no-deps monai`, che pero' ha installato monai 1.6.0
(sintassi tipizzazione `X | None` non supportata dal parser TorchScript di
torch 2.0.0). **Risolto**: `pip install --no-deps monai==1.2.0` (versione
precedente all'adozione della sintassi `X | None`) - `check_prtreid_env.py`
conferma ambiente `prtreid-ft` pienamente funzionante (tutti i moduli e i
checkpoint OK). Ambiente pronto; lo Splitter reale (DBSCAN sugli embedding
per-campione, non solo lo scalare di consistenza) resta da costruire come
prossimo passo.

**Dataset esterni valutati oggi per aumentare i dati del detector regione
(tutti scartati per il detector specificamente)**:
- DeepSport/DeepSportradar: nessuna annotazione sul numero, task diversi
  (palla 3D, calibrazione, segmentazione, re-ID).
- Paper NFL/Seahawks a dati sintetici: stesso dominio gia' scartato,
  localizza via posa (gia' respinto oggi), risolve la classificazione non
  la localizzazione.
- TeamTrack: nessun numero di maglia annotato, camere fisheye/drone
  sostanzialmente statiche - gli stessi autori evitano il confronto con
  dataset broadcast per questo motivo.
- Single-Stage Uncertainty-Aware JNR (Grad, CVPR2025W): idea
  architetturale interessante (elimina il detector regione), ma allenato
  su SoccerNet Jersey Number (probabile overlap con GSR, vedi sopra), zero
  codice rilasciato, e un modello end-to-end richiede *piu'* dati non
  meno - i nostri 435 reali gia' insufficienti per il detector piu'
  semplice.
- SoccerSynth-Detection (Unreal Engine, stesso gruppo TeamTrack): la base
  tecnica piu' credibile vista oggi (motion blur simulato, transfer
  validato per la detection giocatore), ma zero numeri di maglia
  renderizzati, e il proprio limite dichiarato (fallisce a rilevare le
  teste per "diversita' insufficiente nei modelli") e' un campanello
  d'allarme diretto per dettagli fini come un numero di maglia. Base di
  partenza sensata solo per un investimento serio futuro, non ora.
- Idea "registrare video stile FIFA": scartata - gap sim-to-real
  probabilmente peggiore di quanto gia' visto oggi con dati reali di
  dominio diverso, piu' problema di licenza (gioco commerciale protetto).

**Pseudo-labeling verificato - la strada scelta per oggi**: invece di
self-training ingenuo (rischio di rinforzare i punti ciechi del modello),
raccolti solo crop dove: (1) il detector ha trovato una regione, (2) la
predizione CTC del singolo crop combacia col consenso multi-frame della
tracklet, (3) quel consenso combacia col ground truth reale (Identity
Benchmark V1) - la label e' quindi ancorata a una fonte indipendente, non
solo al modello che si conferma da solo. Nuovo script
`scripts/harvest_verified_region_crops.py`.

Risultato su Int-Ata (run `int_ata_all_fixes_verify_1200f`): **43
candidati verificati su 98 crop con regione rilevata**, distribuiti su
~19 tracklet diverse - dati reali del dominio broadcast (non GSR),
verificati indipendentemente. Contributo modesto (~10% dei 435 originali)
ma a costo quasi zero.

**Estensione a Inter-Juve/Inter-Atalanta**: nessuna run esistente aveva
l'audit regione-CTC attivo. Nuovo script
`scripts/build_region_ctc_audit_config_from_run.py`: legge la config gia'
risolta dal `run_manifest.json` di una run sorgente esistente (quelle
dichiarate in `evaluation/identity_benchmark_v1.yaml` per ciascun video,
necessarie per allineare la numerazione `display_track_id`/
`scene_segment_id` col ground truth gia' etichettato) e aggiunge solo il
blocco `jersey_region_ctc_audit` sopra, senza toccare altro. Bug trovato e
corretto: la config risolta della run Inter-Juve conteneva una chiave di
tracking obsoleta (`strongsort`) non piu' accettata dall'attuale
`YoloByteTracker.__init__` - probabile drift dello schema di config nel
tempo. Fix: lo script ora filtra le chiavi di `tracking` per introspezione
contro i parametri realmente accettati dal tracker corrente, scartando
quelle obsolete con un avviso esplicito (invece di cambiare qualunque
altro comportamento di detection/tracking/linking). Run di raccolta
lanciate per entrambi i video a fine giornata, esito non ancora noto.

## Aggiornamento 2026-07-23 - GTA Splitter: calibrazione eps e validazione cross-video

### Splitter reale costruito e calibrato

`scripts/audit_tracklet_identity_splitter.py` (gia' esistente ma mai
validato con parametri corretti): DBSCAN su embedding PRTReID per-campione
dentro ogni tracklet, per rilevare due identita' diverse mescolate sotto
lo stesso `display_track_id`. Primo run su Int-Ata con i parametri di
default del paper GTA (`eps=0.6`, `max_samples_per_tracklet=24`,
sotto-campionamento uniforme): **0 tracklet flaggati su 89**, incluso il
`display_track_id=4` gia' noto per certo come misto (6 numeri di maglia
GT diversi tra segmenti di scena, e gia' il proxy scalare
`prtreid_tracklet_consistency` lo aveva isolato come il piu' basso).
Falso negativo totale del parametro di default.

Diagnosticata la causa con uno script dedicato
(`scripts/diag_track4_distances.py`, non versionato - one-off): estratti
tutti i 1159 embedding del track 4 (non sotto-campionati) e calcolata la
matrice di distanza coseno completa. Distanze intra-identita' (stesso
blocco temporale): ~0.01-0.03. Distanze inter-identita' (blocchi
temporali lontani, presumibilmente giocatori diversi): ~0.29-0.41. Con
`eps=0.6` nessuna distanza, nemmeno tra identita' diverse, supera mai la
soglia - il DBSCAN colla tutto in un cluster unico per costruzione, non
per mancanza di segnale. Causa doppia: `eps` tarato sul paper originale
(OSNet + SportsMOT, dominio diverso) troppo permissivo per PRTReID su
crop broadcast soccer, e sotto-campionamento a 24 frame troppo aggressivo
per catturare la struttura a blocchi su tracklet lunghe.

Fix: `--max-samples-per-tracklet` default `24 -> 0` (nessun limite, usa
tutti i crop disponibili), `--dbscan-eps` default `0.6 -> 0.15`,
`--dbscan-min-samples` default `3 -> 5`. Rilanciato su Int-Ata,
`--only-display-track-id 4`: **isolato correttamente in 2 cluster**
(291 + 868 campioni) - split quasi esatto sul confine reale tra segmenti
di scena.

### Validazione cross-video

Sweep completo con i parametri calibrati sui tre video con run
PRTReID-enabled (Int-Ata gia' esistente; Inter-Juve/Inter-Atalanta
costruite oggi via nuovo script `scripts/add_prtreid_to_config.py`, che
inietta il blocco `prtreid` - identico a quello di
`Int-Ata_prtreid_conservative_final_1200f` - sopra una config
tracking/linking gia' allineata alla source run del ground truth, senza
toccare altro):

```text
video           tracklet valutati   flaggati   confermati da GT reale
Int-Ata               89                5      1 vero positivo (5 identita'
                                                mescolate su 5 segmenti di
                                                scena), 4 inconclusivi
                                                (GT troppo rado in quei punti,
                                                non falsi positivi accertati)
Inter-Juve            39                0      0 attesi - coerente
                                                (nessun display_track_id ha
                                                >1 gt_jersey_number distinto
                                                nel benchmark)
Inter-Atalanta        20                0      0 attesi - coerente
```

Cross-check contro il ground truth eseguito con nuovo script
`scripts/crosscheck_splitter_flags_against_gt.py` (per ogni
`display_track_id` flaggato, elenca tutti gli item GT che lo referenziano
e i `gt_jersey_number` distinti attraverso i segmenti di scena).

**Conclusione onesta**: lo Splitter con i parametri calibrati non produce
falsi positivi grossolani (zero flag dove il GT non conosce nessun
mixing) ed e' capace di isolare correttamente l'unico caso di mixing
reale osservato finora, con uno split quasi esatto sul confine vero. Ma
un solo vero positivo non e' sufficiente per stimare un tasso di
richiamo (recall) affidabile, e i 4 casi inconclusivi di Int-Ata restano
da etichettare a mano se si vuole un secondo punto dato. L'assenza di
casi positivi su Inter-Juve/Inter-Atalanta e' plausibilmente dovuta a
tracklet piu' continue in quei clip (Inter-Juve: quasi tutte le tracklet
coprono l'intero segmento di 1200 frame senza interruzioni;
Inter-Atalanta: clip piu' corto, max ~292 campioni per tracklet), quindi
meno occasioni strutturali di mixing indotto da reset di scena rispetto a
Int-Ata. Non ancora deciso se/come integrare lo Splitter nel percorso
operativo (resta audit-only, non applica nessuno split).

### Pseudo-labeling verificato: raccolto completato su Inter-Juve e Inter-Atalanta

Eseguito `scripts/harvest_verified_region_crops.py` sulle run di raccolta
lanciate ieri sera:

```text
video            crop con regione   verificati   tracklet distinti
Int-Ata                  98              43              ~19
Inter-Juve               11               5               1 (bassa diversita')
Inter-Atalanta           22              13               8
totale                  131              61              ~28
```

Inter-Juve ha una resa molto piu' bassa (5/11, tutti dallo stesso
tracklet `1#0`) - contributo quasi nullo in termini di varieta' per il
training del detector, anche se non e' un errore: la config aveva
`max_crops_per_tracklet=5` e questo video ha prodotto meno tracklet
idonei nella run sorgente. Inter-Atalanta ha resa migliore (13/22, 8
tracklet diversi: numeri 6, 7, 8, 10, 23, 28, 36, 59).

Pool complessivo: 61 nuovi crop verificati contro i 435 originali del
training set del detector regione (`gsr_train_manual_v1`) - circa +14%.
Contributo reale ma modesto; non ancora deciso se sufficiente per
giustificare un nuovo retraining del detector (i tre retraining gia'
tentati con piu' epoche/pretraining esterno non erano stati promossi,
vedi sezione precedente - il pool aggiuntivo qui e' dati reali nuovi, non
piu' training sugli stessi dati, quindi il confronto non e' direttamente
lo stesso esperimento).

## Aggiornamento 2026-07-23 (2) - Allargare la finestra di candidate generation del linker: respinto

### Motivazione

Seguito diretto del dato piu' rilevante emerso ieri (sezione "correzione
pitch-gate"): in 22 target su 50 (44%) il candidato corretto non entra
nemmeno nel pool valutato dal linker, per via della finestra locale
`max_gap=90` frame / `max_distance=160px` (`ft/linking/tracklet_linker.py`).
Prima di investire su GTA, testato il fix piu' economico possibile:
allargare semplicemente questi due parametri, lasciando invariati
team-gate e appearance-gate a valle.

### Esperimento

Nuova config one-variable `configs/gsr_detection_tracking_conf012_widelinker_v1.yaml`
sopra la baseline congelata `conf012` (stesso pilot12, stesso modello):
`linking.max_gap: 90 -> 180`, `linking.max_distance: 160.0 -> 250.0`.

Nota laterale utile: aggiunta in questa occasione una var booleana
`export.save_video` (default `true`, invariato ovunque tranne il profilo
benchmark GSR dove ora e' `false`) in `ft/pipeline.py`
(`render_output_phase` ritorna subito se disabilitato) - il benchmark
detection/tracking legge solo i CSV di detection/tracklet, mai il video
annotato, quindi disattivarlo risparmia tempo di encoding senza toccare
nessuna metrica.

### Risultato: respinto

```text
delta (widelinker - baseline conf012), 12 sequenze:
  display HOTA:  -0.00183 (3 migliorate, 2 pari, 7 peggiorate)
  display AssA:  -0.00242 (3 migliorate, 2 pari, 7 peggiorate)
  display IDF1:  +0.00129 (5 migliorate, 2 pari, 5 peggiorate, CI attraversa lo zero)
  display MOTA:  -0.00001 (11 pari, 1 peggiorata)
  frammentazioni/1000 GT: 0.0 in tutte le 12 sequenze, nessun cambiamento
  raw HOTA/IDF1: 0.0 in tutte le 12 sequenze (atteso, il linker non tocca il raw tracker)
```

Il dato piu' informativo e' che le **frammentazioni sono identiche su
tutte e 12 le sequenze**: allargare la finestra rende eleggibili piu'
candidati, ma team-gate/appearance-gate a valle li scartano quasi sempre
comunque; i pochi che passano peggiorano leggermente l'associazione
(falsi collegamenti accettati) invece di recuperare quelli veri mancanti.

**Conclusione**: coerente con l'esito gia' visto su TVCalib (gate fisico)
e sul reranking morbido sulla distanza pitch - qualunque intervento
*locale* del linker (guardare solo la coppia di tracklet candidata, in
ordine temporale greedy, indipendentemente da quanto ampia sia la
finestra) non recupera il gap del 44%. Rafforza ulteriormente la
priorita' di un'associazione *globale* offline (GTA Connector) rispetto a
qualunque altro tuning locale di `TrackletLinker`. Non ripetere sweep di
soglia su questi due parametri sullo stesso pilot12.

## Aggiornamento 2026-07-23 (3) - GTA Connector: respinto, precisione 7.3%

### Motivazione e metodologia

Seguito diretto del punto precedente: prima di scrivere qualunque logica
di merge nella pipeline reale, costruito un audit offline puro (nuovo
script `scripts/audit_gta_connector.py`, zero mutazione) che replica
l'idea del Connector di GTA - fondere coppie di raw tracklet *diverse*,
non sovrapposte nel tempo, la cui distanza coseno tra prototipi PRTReID
media e' sotto soglia (`merge_dist_thres=0.4`, `min_len=100`, parametri
del paper) - e confronta ogni proposta con il ground truth GSR offline
(mai usato nella logica di proposta). Rischio esplicito da verificare:
il Connector e' concettualmente vicino allo "shadow tracker"
appearance-based gia' respinto in passato (309 vs 45 ID switch su
SNGS-025, causa nota: l'aspetto non discrimina bene compagni di squadra
con maglie quasi identiche).

Eseguito su tutte le 12 sequenze pilot (run congelata `conf012`, nessuna
nuova inferenza - solo tracklets.csv + frame gia' estratti + PRTReID):

```text
totale coppie proposte:        1307
coppie con GT scorabile:        1307
fusioni corrette:                 96
fusioni sbagliate:              1211
precisione:                    7.3%
sbagliate stesso team:           531
sbagliate team diverso:          345
sbagliate team ignoto:           335
```

### Risultato: respinto, e piu' severo del previsto

Precisione **7.3%** - inutilizzabile cosi' com'e'. Ancora piu' rilevante:
delle fusioni sbagliate, **345 sono tra squadre diverse**, non solo tra
compagni di squadra. L'ipotesi di partenza (aspetto confuso solo tra
maglie simili) spiega parte del problema (531 casi) ma non tutto - la
soglia `0.4` e' semplicemente troppo permissiva in questo dominio per
l'embedding medio-tracklet di PRTReID, indipendentemente dalla somiglianza
di maglia. Causa probabile: mismatch di dominio del checkpoint
(allenato su SoccerNet, non sui nostri crop) o limite intrinseco della
media di un embedding su tracklet lunghe con pose/illuminazione molto
variabili (a differenza dello Splitter, che confronta campioni
*all'interno* della stessa tracklet e ha funzionato bene con lo stesso
embedding).

**Decisione**: Connector respinto. Non fare uno sweep di soglia piu'
stretta come prossimo tentativo automatico: con quasi un terzo degli
errori gia' cross-team a soglia 0.4, restringere la soglia rischia solo
di eliminare anche le poche fusioni corrette senza risolvere il problema
di fondo. GTA nel suo complesso chiude qui con un bilancio misto onesto:
Splitter validato su un caso reale confermato (vedi sopra), Connector
respinto con evidenza chiara. Riprendere l'idea di associazione globale
solo con un embedding diverso/fine-tuned sul dominio target, non con
altro tuning di soglia sullo stesso PRTReID pretrained.

## Aggiornamento 2026-07-23 (4) - Pseudo-labeling: allargata la finestra di raccolta e recuperati i "misread" corretti dal contesto

### Allargata la finestra di export per tracklet

`jersey_region_ctc_audit` limitava l'export a `max_crops_per_tracklet=5`
per tracklet, indipendentemente da quanto fosse lunga (alcune tracklet
Int-Ata superano 1000 frame). Reso parametrico in
`scripts/build_region_ctc_audit_config_from_run.py`
(`--max-crops-per-tracklet`, `--min-frame-gap`, default invariati a 5/5)
e creata una config dedicata `configs/int_ata_jersey_region_ctc_pseudolabel_harvest_v1.yaml`
(30/3) separata da quella "congelata" usata per il confronto scientifico
CTC (`int_ata_jersey_region_ctc_roster_rerank_audit_v1.yaml`, lasciata
invariata a 5/5 - non toccare quel profilo, e' il riferimento del test
ufficiale). Stesso allargamento applicato anche a Inter-Juve/Inter-Atalanta.

Nota laterale: incontrato di nuovo il crash CUDA transitorio "illegal
memory access" gia' documentato per questa GPU (RTX 2080, 7.8GB),
questa volta durante l'inferenza del detector principale (non
training) e con due processi `ft.cli run` concorrenti sulla GPU - la
VRAM non era satura (2.9/8GB), quindi non e' un OOM pulito. Non
approfondito oltre; workaround usato: aspettare che l'altra run finisca
e rilanciare in sequenza.

### Recuperati i "misread" corretti dal contesto (non solo gli accordi puliti)

Domanda del giorno che ha portato al fix: ha senso ri-analizzare
un'informazione (es. un numero letto male dal detector) alla luce di
conclusioni acquisite piu' avanti (consenso multi-frame, roster-rerank)?
Risposta: si', ed esisteva gia' un punto preciso dove questo veniva
scartato invece che sfruttato.

`scripts/harvest_verified_region_crops.py` teneva solo i crop dove la
lettura CTC del singolo frame (`ctc_top1`) **concordava gia'** col
consenso finale della tracklet (`ctc_disagrees_with_consensus` scartava
tutto il resto). I crop scartati per questo motivo sono pero' i casi
piu' interessanti: frame dove il CTC ha letto male quel singolo crop, ma
il consenso finale (verificato contro il ground truth reale, non contro
se stesso) e' comunque noto per certo - errori del recognizer gia'
risolti dal contesto multi-frame/roster. Rietichettarli con l'etichetta
corretta li rende esempi verificati e potenzialmente piu' preziosi
(hard example), senza rischio di auto-rinforzo perche' l'ancora resta il
GT indipendente, non l'accordo del modello con se stesso - stessa
disciplina del pseudo-labeling gia' in uso.

Aggiunto flag opt-in `--include-corrected-misreads` (default `False`,
comportamento pre-esistente invariato se omesso) che include questi casi
taggati distintamente (`verification_type: corrected_misread` vs
`clean_agreement`), per trasparenza - non li confondiamo con gli accordi
puliti nell'output.

Risultato su Int-Ata (stesso run `int_ata_all_fixes_verify_1200f`, nessuna
nuova inferenza):

```text
total_crops_with_region=98
verified_candidates=63 (prima: 43)
  clean_agreement=43
  corrected_misread=20
rejected_reasons: no_gt_for_track=7, no_tracklet_consensus=16,
                  consensus_not_ground_truth=12
```

**+20 crop recuperati (+46% sul pool Int-Ata)**, a costo computazionale
zero (letti dagli stessi dati gia' esportati, nessuna nuova run).
Prossimo passo naturale: ri-eseguire l'harvest con lo stesso flag su
Inter-Juve/Inter-Atalanta una volta completate le run allargate lanciate
oggi (`inter_juve_pseudolabel_harvest_wide_1200f`,
`inter_atalanta_pseudolabel_harvest_wide` - quest'ultima interrotta dal
crash CUDA, da rilanciare in sequenza).

Non ancora testato (deliberatamente rimandato, rischio piu' alto e
vicino a percorsi gia' respinti - linker allargato, Connector): ri-
eseguire detection/tracking stesso usando le conclusioni della prima
passata come prior (es. per recuperare frame dove il detector non ha
trovato nessuna regione). Valutare solo dopo aver misurato il guadagno
reale di questo fix piu' economico.

### Risultato finale su Inter-Juve e Inter-Atalanta: pool quadruplicato

Rilanciato l'harvest con `--include-corrected-misreads` sulle run
allargate (30 crop/tracklet, gap minimo 3) completate per entrambi i
video (Inter-Atalanta rilanciata con successo dopo il crash CUDA
transitorio, nessun errore alla seconda esecuzione in sequenza):

```text
video            crop con regione   verificati (clean + corrected)
Int-Ata                  98              63 (43 + 20)
Inter-Juve              109              92 (70 + 22)
Inter-Atalanta          151             122 (72 + 50)
totale                  358             277
```

**Pool totale: 277 crop verificati contro i 435 originali del training
set del detector (`gsr_train_manual_v1`) - +64%**, un contributo molto
piu' sostanziale del +14% misurato ieri con la finestra stretta
(5 crop/tracklet, solo accordi puliti). I due fattori si sono
rinforzati: allargare `max_crops_per_tracklet` ha aumentato il numero
grezzo di crop disponibili, e includere i `corrected_misread` ha quasi
raddoppiato la resa su Inter-Atalanta (50 dei 122 verificati, il 41%)
e Inter-Juve (22/92, il 24%) - segno che il recognizer sbaglia spesso
sul singolo frame anche quando il contesto (multi-frame + roster) lo
corregge senza ambiguita'.

**Decisione da prendere**: con 277 crop (quasi il doppio del training
set originale di 435), la domanda "vale la pena un retraining del
detector?" e' ora concreta e non piu' marginale come con il pool da 61.
Prossimo passo naturale: costruire il training set unito (originale +
pseudo-labeled, mantenendo lo split sequenza-disgiunto gia' usato) e
ripetere l'esperimento di retraining gia' fatto tre volte ieri (stavolta
con piu' dati reali nuovi, non solo piu' epoche/pretraining esterno sugli
stessi dati) - misurare hard-miss-rate e coverage@0.05/0.25 come nelle
tre run precedenti, stesso protocollo, per confronto diretto.

## Aggiornamento 2026-07-23 (5) - Retraining con pseudo-labeling: coverage isolata migliora, end-to-end peggiora

### Dataset unito e training

Aumentato ulteriormente il pool di pseudo-labeling (config
`max_crops_per_tracklet=30`, `min_frame_gap=3`, invece di 5/5, sulle
run gia' completate) e aggiunto il recupero dei "corrected_misread"
(vedi sezione precedente) su tutti e tre i video: **640 crop verificati
totali** (Int-Ata 426, Inter-Juve 92, Inter-Atalanta 122) - molto piu'
dei 61/277 stimati con le config strette. Nuovo script
`scripts/merge_pseudolabel_region_dataset.py`: unisce questi 640 crop
nel solo partizionamento `images/train`/`labels/train` del dataset
originale (`gsr_train_manual_v1`, 435 immagini), lasciando la
validation (110 immagini, sequenze GSR) byte-identica - i video custom
non fanno parte dello split train/val/frozen GSR, quindi nessun rischio
di leakage. Dataset unito: **1075 immagini di training** (435 + 640).

Training: stessi parametri delle tre run precedenti (seed `20260719`,
`yolo26s.pt`, batch 16, patience 10). Fermato per early stopping dopo
24 epoche (best a epoca 14). Checkpoint:
`runs/jersey_number_region/yolo26s_pseudolabel_v1/weights/best.pt`
sha256 `977d7f500896cf0b5dd5574254c849b1a058ccc05ff8a4dc4e8087829f11c50c`.

### Coverage isolata: pattern diverso, apparentemente migliore

Stesso audit delle tre run precedenti
(`scripts/audit_jersey_number_region_detector_coverage.py`, ocr-run
`readability_finetuned_surface`, confidence 0.25):

```text
                          coverage@0.25  hard_miss  fully_blind  coverage@0.05
smoke (10 epoche)              60.7%       25.0%       12/173        75.0%
full GSR-only (43 epoche)      62.8%       27.2%       13/173        72.8%
SJN->GSR (25 epoche)           63.7%       28.1%       13/173        71.9%
pseudolabel (24 epoche)        62.9%       15.5%       15/173        84.5%
```

A differenza delle tre run precedenti (dove piu' training peggiorava
sistematicamente hard-miss/coverage a soglie basse), qui hard-miss
crolla quasi a meta' e coverage@0.05 sale al valore piu' alto misurato
finora. Unico dato in controtendenza: `fully_blind_tracklets` sale
leggermente (15 vs 12-13).

### End-to-end (CTC fisso, stesso ocr-run "finetuned_sar" della tabella originale): PEGGIORE su ogni metrica

Prima di promuovere il checkpoint, misurato l'impatto reale con
`scripts/evaluate_jersey_number_region_ctc_ocr_run.py`, CTC fisso
(`jersey_number_ctc_sjn_to_gsr_v1.pth`), stesso protocollo esatto della
tabella frozen originale (ocr-run
`jersey_ocr_finetune/recognizer_ablation_frozen_v1/finetuned_sar`,
confidence 0.25, box-padding 0.25):

```text
                         SJN->GSR (yolo26s_smoke_v1)   pseudolabel (yolo26s_pseudolabel_v1)
region coverage          0.8162                        0.8132
track coverage           0.9191                        0.9017
correct                   124                            117
wrong                      35                             39
accuracy all             0.7168                         0.6763
accuracy assigned        0.7799                         0.7500
GT in top-5              84.4%                          81.5%
```

**Peggiore su ogni singola metrica**, nonostante il miglioramento netto
sulla coverage isolata. Contraddizione solo apparente: l'audit di
coverage misura se il detector trova *una* regione, non quanto bene la
localizza (IoU/precisione del box). Il nuovo detector, allenato anche
sui crop "corrected_misread" (regioni corrette dal contesto ma piu'
"rumorose" come esempi di training), sembra aver imparato a rilevare
piu' spesso ma con box meno precisi - lo stesso identico pattern gia'
visto con la geometria da posa (trova la regione piu' spesso ma il
recognizer legge quello che trova molto peggio, li' il crollo era
drastico 78%->36%, qui piu' contenuto ma stessa dinamica).

**Decisione**: checkpoint `yolo26s_pseudolabel_v1` **non promosso**.
Resta operativo `yolo26s_smoke_v1`. Risultato comunque prezioso: conferma
che aumentare volume/diversita' dei dati di training non basta se non si
controlla anche la qualita' di localizzazione (non solo presenza/assenza
del box) - prossimo passo, se si vuole insistere su questa direzione,
sarebbe un audit IoU per-crop tra il box del nuovo detector e quello
del vecchio sugli stessi campioni rilevati da entrambi, per confermare
l'ipotesi della localizzazione piu' rumorosa prima di provare a
correggerla (es. filtrando i crop "corrected_misread" piu' estremi dal
training set, o pesandoli meno).

## Aggiornamento 2026-07-23 (6) - Nuovo video Por-Uzb (Mondiale 2026) e nuova feature: escalation OCR sui tracklet senza numero

### Setup del video

Aggiunto un nuovo video custom `Por-Uzb` (Portogallo-Uzbekistan,
broadcast Mondiale 2026). Roster costruito a mano da formazioni Sky
Sport (fetch web) nello stesso formato di `Int-Ata.json`
(`player_id`/`team_id`/`jersey_number`/`role`/`position_prior`), ma
**senza inventare dati biometrici** (altezza, piede, nazionalita') per
i singoli giocatori, trattandosi di persone reali identificabili - a
differenza del roster Int-Ata (anch'esso persone reali, non fittizie:
correzione fatta a valle, quei dati biometrici lì erano gia' presenti
da una sessione precedente, non verificati da fonte indipendente in
questa sessione). 52 giocatori (26 per squadra, 11 titolari + 15
riserve), nessun numero duplicato per squadra.

Due bug incontrati e risolti sulla prima run:
1. `max_frames: 300` ereditato silenziosamente da una config a monte
   della catena (probabile residuo di uno smoke-test) - troncava il
   video (554 frame reali) senza errori o avvisi. Fix: passare sempre
   `--max-frames` esplicito e verificare nel `run_manifest.json` prima
   di fidarsi dell'output.
2. Crash CUDA "illegal memory access" transitorio (stessa instabilita'
   nota della GPU) - risolto rilanciando lo stesso comando.

### Diagnosi bassa coverage jersey

Con soglie di default (`min_votes=2`, `min_winner_margin=0.15`), solo
17/27 tracklet ottenevano un numero. Causa: crop giocatore piccoli
(mediana 55px lato minimo, sotto la soglia dove negli audit del
detector regione la coverage crolla) - ma **nota bene**: Por-Uzb usa il
backend `mmocr+easyocr` (lettura diretta del testo), non la pipeline
YOLO-regione+CTC auditata nei giorni precedenti - le soglie
44px/55px/63px di quegli audit non si applicano direttamente qui,
il meccanismo (testo piccolo piu' difficile da leggere) e' solo
analogo, non lo stesso sottosistema.

Allargare le soglie OCR (`min_confidence 0.25->0.10`,
`min_winner_margin 0.15->0.08`, `min_votes 2->1`) ha portato la coverage
a 27/27 tracklet, ma indebolendo il criterio di accettazione per tutti.

### Nuova feature: escalation on-demand, non tuning globale

Domanda del giorno: ha senso, se un tracklet non ottiene un numero dal
campione iniziale, andare a leggere altri crop invece di abbassare la
soglia per tutti? Risposta: si', e il motivo per cui l'OCR non guarda
gia' tutti i frame e' puro costo (campionare ~86 crop per tracklet
costa gia' 15-45 minuti con mmocr+easyocr; leggerli tutti costerebbe
~6x).

Implementato in `ft/features/jersey_ocr.py`: nuovo parametro
`escalate_on_unassigned` (default `false`, nessun cambiamento per le
run esistenti). Se un tracklet non raggiunge `min_votes` con il
campione iniziale, la pipeline pesca fino a `escalation_max_extra_crops`
(default 20) crop **non ancora usati** dal pool rimanente di quel
tracklet specifico, li fa leggere, e ri-vota includendoli. Il costo
extra si paga solo per i tracklet che ne hanno bisogno. Diagnostica:
ogni tracklet riporta `escalated: true/false` in `*_jersey_ocr.json`.

Bug trovato e corretto durante il primo test: passato erroneamente la
stringa `"escalation"` come `pass_index` (il codice lo usa per
formattare nomi di file di debug con `{pass_index:02d}`, causando
`ValueError: Unknown format code 'd' for object of type 'str'`) - fix:
usare `self.temporal_passes` (un intero, sempre maggiore di qualunque
indice di pass normale) come indice dedicato per i crop di escalation.

Risultato su Por-Uzb, stesse soglie strette originali (non allargate):

```text
                                          tracklet assegnati
baseline (soglie strette, no escalation)       17/27
escalation (soglie strette, con retry)         21/27
loose (soglie allargate, no escalation)        27/27
```

L'escalation da sola recupera 4 tracklet su 10 che fallivano (40%)
mantenendo i criteri di accettazione originali (piu' severi, minor
rischio di falsi positivi) - miglioramento isolato e reale, non
compensato abbassando la barra per tutti. Non ancora confrontato contro
ground truth reale su Por-Uzb (nessun benchmark manuale costruito per
questo video finora) - il numero misura solo *quanti* tracklet ottengono
un'assegnazione, non se il numero assegnato sia corretto.

## Aggiornamento 2026-07-24 - Nuovi video (Por-Uzb, Esp-Arg, Usa-Bel), fix portiere per colore, bug critico jersey_frame_selection mode=propose

### Roster per tre nuovi video Mondiale 2026

Aggiunti `Por-Uzb` (Portogallo-Uzbekistan), `Esp-Arg` (Spagna-Argentina),
`Usa-Bel` (USA-Belgio). Roster costruiti da fetch web (Sky Sport /
Eurosport). Per persone reali identificabili: nessun dato biometrico
inventato (altezza, piede, nazionalita'), solo numero/nome/ruolo
verificati dalla fonte; dove la fonte non riportava i numeri delle
riserve (Eurosport per Esp-Arg/Usa-Bel), roster limitato ai soli 11
titolari per squadra piuttosto che inventare numeri.

### Bug del colore portiere: il verde puro si confonde col prato

Il rilevatore di ruolo portiere (`GoalkeeperAppearanceAssigner`,
`ft/features/goalkeeper.py`) si attiva solo se il roster fornisce
`metadata.kit_color` per il portiere (letto da
`goalkeeper_color_ranges_by_team_from_roster`). Aggiunto inizialmente
verde fosforescente (`#39FF14`) per il portiere Uzbekistan (Nematov):
risultato, quasi tutti i 27 tracklet di Por-Uzb sono stati classificati
come "goalkeeper" (falsi positivi diffusi, punteggi 0.2-0.7 su quasi
ogni tracklet). Causa: il matching e' basato solo sulla tonalita' HSV
(`h_tolerance=12`), e il verde di una maglia da portiere e' strutturalmente
troppo vicino alla tonalita' del prato per essere distinto in modo
affidabile - non e' un problema della sfumatura scelta, e' un limite
strutturale del classificatore per QUALSIASI verde. Fix: rimosso il
colore per quel portiere (resta senza rinforzo di ruolo, come gia' fatto
deliberatamente per il portiere Belgio su richiesta esplicita "verde ma
non lo mettiamo"). Il teal/azzurro (`#20B2AA` Diogo Costa, `#00BFFF`
Unai Simon, `#FFD700` giallo per Freese) non ha mostrato lo stesso
problema - tonalita' lontane dal verde prato restano sicure.

### Bug critico: `jersey_frame_selection.mode: propose` non scrive mai il jersey_number

Provato il selettore di leggibilita' gia' validato
(`legibility_resnet34_soccer_20240215.pth`, checkpoint verificato via
sha256) per sostituire il campionamento cieco (86+ crop/tracklet) con
al massimo `top_k=5` crop scelti per leggibilita' reale - motivazione:
ridurre drasticamente il tempo di calcolo (lo stage OCR impiegava
15min-2h25 a run). Copiata la config di riferimento gia' validata su
GSR (`gsr_legibility_selector_balanced.yaml`), incluso `mode: propose`.

Risultato: **0 tracklet su 23 con jersey_number nel CSV finale**,
nonostante il diagnostico interno (`*_jersey_ocr.json`) mostrasse
correttamente 10/23 tracklet con `status: assigned` e voti validi.
Causa identificata in `ft/pipeline.py` (righe ~588-606):

```python
observational_propose = any(
    cfg.get("enabled", False) and str(cfg.get("mode", "audit")).lower() == "propose"
    for cfg in (selector_cfg, subject_cfg, prefix_cfg)
)
if not observational_propose and (...):
    _apply_jersey(jersey_assignments, player_rows, tracks, ...)
```

Con `mode: propose` su un modulo abilitato, `observational_propose=True`
e **`_apply_jersey` non viene mai chiamato per nessun tracklet della
run** - non solo per quelli toccati dal selettore. Non e' un bug del
codice: e' il comportamento voluto quando si usa `propose` per fare
audit di un selettore *sopra* un profilo operativo gia' esistente e
funzionante (`propose` = "calcola e confronta, non toccare l'output
reale"). Il nostro caso era diverso: il selettore era l'**unica** fonte
di jersey number nel profilo, quindi serviva `mode: apply`. Fix:
cambiato `mode: propose -> apply` in
`configs/custom_video_legibility_selector_v1.yaml`. Non ancora
verificato il risultato con il fix (run rilanciata, esito pendente).

**Lezione generale**: prima di riusare una config "propose"-only gia'
validata come base per un nuovo caso d'uso, verificare cosa significa
`propose` in quello specifico punto della pipeline - il nome e' condiviso
da piu' moduli (`jersey_frame_selection`, `jersey_subject_filter`,
`jersey_prefix_consolidation`) con la stessa convenzione ("non mutare
l'output reale"), ma l'impatto pratico dipende da cosa altro sta gia'
scrivendo `jersey_number` in quella run.

### Pulizia disco macchina remota

`~/FT` era arrivato a 55G. Investigazione mostra che il grosso non era
nei crop immagine (`crops/`, solo 3.7G totali) ma in export JSON
ridondanti gia' duplicati in CSV: **125 file `*_tracklets.json`** (9.7G,
CSV gemello sempre presente, nessuno script letto oggi li usa) cancellati
in sicurezza. Attenzione: **`*_identity_assignments.json` NON ha un CSV
gemello** (125 file, 2.7G) - probabile unica copia delle decisioni di
assegnazione identita', non cancellare senza verificare se serve
all'evaluator del benchmark. Cancellate anche le 66 cartelle `SNGS-*`
in `artifacts/costume-video` (5.3G, esperimenti GSR di meta' luglio gia'
riassunti in handoff.md, nessuna nelle `source_runs` protette). Lista
delle run intoccabili (source_runs dichiarate in
`evaluation/identity_benchmark_v1.yaml` + run citate per nome nei
risultati definitivi di handoff.md):
`Int-Ata_identity_evidence_v1_resetbytetrack_cutsensitive_1200f`,
`Int-Ata_prtreid_conservative_final_1200f`,
`Inter-Juve_scenecuts_loose_noreset_gkfallback_spreadgate_propagation_strict_linker_legacyappearance_1200f`,
`Inter-Juve_identity_evidence_patch_1200f`,
`Inter-Atalanta_identity_evidence_v1_resetbytetrack_full`,
`int_ata_all_fixes_verify_1200f`, `inter_juve_pseudolabel_harvest_wide_1200f`,
`inter_atalanta_pseudolabel_harvest_wide`.

### Secondo giro di pulizia: sprechi strutturali fermati alla fonte, non solo puliti

Cercato cosa la pipeline continua a *creare* senza che nessuno lo legga
mai, non solo cosa cancellare una tantum:

- **`jersey_ocr_debug/`**: fino a 12 varianti di immagine (torso,
  torso_clahe, upper_back, number_band, ecc.) per ogni crop OCR, mai
  ispezionate in nessuna sessione. Trovate **16G** accumulate solo nelle
  run di questa sessione - piu' grande di `tracklets.json` (9.7G) e
  `SNGS-*` (5.3G) messi insieme. Causa: `jersey_ocr.debug_crops: true`
  di default in `configs/default.yaml`. Cambiato a `false`.
- **`*_tracklets.json`**: stessa causa strutturale del ritrovamento
  precedente (9.7G) - `export.save_final_json: true` di default lo
  rigenera ad ogni run anche se nessuno script lo legge mai (tutti usano
  il `.csv` gemello). Cambiato a `false`.
- **`output_videos/*.mp4`**: overlay di visualizzazione, mai letto da
  nessuno script, generato anche per run puramente diagnostiche (sweep
  di soglia) dove nessuno guarda il video. Non disattivato di default
  (a volte serve davvero per ispezione visiva, es. verifica del fix
  portiere) - da disattivare per-run con `export.save_video: false`
  quando si sa gia' che non servira'.
- **`.ft_cache/`, cache pip/conda/torch**: non strutturale (si
  rigenerano da sole quando servono), ma ~15G liberati con
  `conda clean --all`, `pip cache purge`, rimozione `~/.cache/torch`.
- **`*_pre_identity.json/csv`**: controllato, dimensione trascurabile
  (0 byte trovati nelle run attuali) - non serve intervenire.

Bilancio pulizia disco totale di oggi: **~46G liberati** su ~55G di
partenza, con due sprechi strutturali (debug crops, tracklets.json)
disattivati alla fonte per tutte le run future, non solo ripuliti una
volta.

## Aggiornamento 2026-07-24 (2) - Segnale `full_body_sufficient`: filtri ROI sui crop OCR, validato e implementato

### Domanda di partenza

Non i filtri logici (roster_filter/goalkeeper_filter/gate Hungarian, gia'
investigati oggi), ma i filtri *sull'immagine* usati dall'OCR: le regioni
ROI (`full_body`, `back_number_wide`, `upper`, `torso`, `number_band`,
ecc. - vedi `legacy_ocr_regions()`/`number_roi_regions()` in
`ft/features/jersey_ocr.py`) ritagliate dal crop prima di leggere il
numero. Troppi filtri/regioni potrebbero peggiorare il problema invece di
aiutarlo?

### Metodologia: ablation offline, zero inferenza GPU nuova

Ogni detection gia' salvata in `*_jersey_ocr.json`/`ocr_diagnostics.json`
porta il campo `variant` (nome regione + trasformazione, es.
`roi_number_band_clahe`). Riusando direttamente le funzioni reali
`aggregate_detections_by_crop()`/`vote_numbers()` di
`ft/features/jersey_ocr.py` (non una riscrittura), e' possibile
ri-simulare la decisione di voto di un tracklet sotto un sottoinsieme di
detection diverso (es. solo regione `full_body`) senza rileggere nessun
crop. Testato contro due run di valutazione OCR gia' esistenti su
SoccerNet-GSR con ground truth reale, mai ispezionate prima in questa
sessione:

- `evaluation_outputs/gsr_jersey_ocr_mmocr_50s/gsr_val_mmocr_easyocr_50s_5000t`
  (830 tracklet, 50 sequenze).
- `evaluation_outputs/gsr_jersey_ocr_mmocr_10s_v2/gsr_val_mmocr_easyocr_10s_1000t`
  (157 tracklet, 10 sequenze indipendenti, usato come validazione).

Baseline della simulazione riprodotto esattamente il `metrics.json` gia'
salvato (`accuracy_assigned=0.4305`, `coverage=0.7024` sul campione da 50
sequenze) - conferma che la metodologia e' fedele.

### Risultato: `full_body` da solo e' molto piu' preciso, ma con meno coverage

| Policy | 50 seq: coverage | 50 seq: accuracy | 10 seq: coverage | 10 seq: accuracy |
|---|---|---|---|---|
| baseline (tutte le regioni) | 70.2% | 43.1% | 66.9% | 54.3% |
| solo `full_body` | ~29% | **75.2%** | ~32% | **84.3%** |

Un cascade a due stadi (prova prima `full_body` da solo, ripiega su tutte
le regioni solo se non raggiunge il quorum voti) NON migliora
l'accuracy pooled, perche' le decisioni di stadio 2 (fallback) sono
sistematicamente solo ~20% accurate su **entrambi** i campioni
indipendenti (20.0% e 20.4%) - troppo vicino al rumore per compensare la
perdita di coverage.

Il punto utile non e' il cascade come sostituto della decisione, ma la
**stratificazione**: sapere se la decisione (quella con tutte le
regioni, la stessa gia' in uso oggi) sarebbe *anche* raggiungibile dalla
sola regione `full_body` e' un segnale di affidabilita' forte e
riproducibile, confermato su due campioni GSR indipendenti prima di
fidarsene (richiesta esplicita: "prima validalo su un altro dataset
GSR").

### Implementazione in produzione (non solo ablation)

Segnale chiamato `full_body_sufficient` (booleano), implementato in tre
punti:

1. **`ft/features/jersey_ocr.py`**: nuova funzione `is_full_body_variant()`
   (stesso stile di `is_template_variant()`/`should_direct_recognize_variant()`
   gia' presenti, substring-check su `"full_body"` nel nome variant - tutte
   le regioni strette non contengono questa substring, verificato). Nel
   loop principale `recognize()`, dopo il voto finale (incluso l'eventuale
   escalation), viene ricalcolato un secondo voto usando solo le detection
   `full_body` (stessa aggregazione per-crop, stessi parametri
   `min_raw_confidence`/`digit_confusion_overrides` della decisione reale).
   Se questo secondo voto raggiunge da solo `min_votes`, `full_body_sufficient
   = True`. Zero costo aggiuntivo: nessun nuovo crop letto, nessuna nuova
   inferenza OCR - solo un secondo voto sulle detection gia' calcolate.
   Il campo finisce sia nei diagnostics per-tracklet sia nel dict `voted`
   (quindi nelle assignments quando accettate).
2. **`ft/pipeline.py`**: `_apply_jersey()` promuove il segnale a campo
   piatto `jersey_full_body_sufficient` su row e track, stesso pattern di
   `jersey_votes`/`jersey_confidence`/`jersey_winner_margin`.
3. **`ft/identity/hungarian.py`**: nuovo parametro
   `narrow_region_jersey_score_penalty` (default `0.15`,
   `identity.narrow_region_jersey_score_penalty` in `configs/default.yaml`).
   Quando `jersey_full_body_sufficient is False`:
   - `_has_reliable_jersey_match()`/`_has_reliable_jersey()`: alza di 0.15
     sia la soglia di `candidate_score` che quella di `winner_margin`
     richieste per concedere `reliable_jersey` nel gate di assegnazione -
     evidenza raggiunta solo con regioni strette deve essere piu' forte per
     passare, invece di essere scartata a priori;
   - `jersey_reliability()`: dimezza il peso soft-cost usato in `_score()`
     (influenza sia il bonus per match che la penalita' per mismatch, oltre
     al prior "numero 1 = portiere").

Scelta deliberata di non trasformare il segnale in un veto secco
(`reliable_jersey = False` sempre quando narrow-region): non c'e' ancora
una validazione end-to-end sul benchmark identity reale (solo sulla
metrica OCR isolata), quindi il segnale abbassa la fiducia in modo
graduale invece di bloccare, seguendo lo stesso principio "unknown
invece di identita' debole" gia' applicato altrove, ma senza introdurre
un cambio di comportamento troppo aggressivo senza misura diretta
sull'identity benchmark.

### Verifica sul codice effettivamente spedito (non solo sulla logica derivata)

Rieseguita la stratificazione con le funzioni *reali* appena modificate
(`is_full_body_variant`, `aggregate_detections_by_crop`, `vote_numbers`)
sugli stessi due dataset GSR, usando pero' la decisione baseline (tutte
le regioni, quella realmente usata come `jersey_number`) come risposta
scorata, e il flag `full_body_sufficient` solo come segnale di
affidabilita' (esattamente come viene consumato in produzione, non come
decisione alternativa):

| Dataset | `full_body_sufficient=True` | `full_body_sufficient=False` |
|---|---|---|
| 50 sequenze | 76.5% (182/238) | 20.0% (69/345) |
| 10 sequenze (indipendente) | 90.2% (46/51) | 20.4% (11/54) |

Pattern confermato, anzi il bucket `True` risulta ancora piu' forte
misurato cosi' (76.5%/90.2%) che nell'ablation con cascade a se stante
(75.2%/84.3%) - coerente: qui la decisione scorata e' quella con tutte
le detection, che tende ad avere piu' evidenza totale a supporto quando
`full_body` da solo bastava comunque.

Nessun bug di trascrizione tra logica validata e codice shippato.
`min_votes`/`min_raw_confidence`/`max_candidates_per_crop`/
`min_crop_candidate_ratio` letti dal run stesso (`ocr_diagnostics.json`),
non hardcoded.

File nuovo per riprodurre la verifica: `scripts/verify_full_body_signal.py`
(importa direttamente da `ft.features.jersey_ocr`, va eseguito da `~/FT`).

### Bug reale trovato al primo run end-to-end: il segnale non arrivava mai al gate

Prima run di verifica end-to-end su un video reale (non GSR):
`Usa-Bel`, config `usa_bel_jersey_ocr_all_crops_v1.yaml` (stesso profilo
"tutti i crop" di Por-Uzb, con `number_region.enabled: false` per
evitare un doppio caricamento MMOCR/EasyOCR nello stesso processo - una
combinazione che aveva gia' causato un crash CUDA "illegal memory
access" al primo tentativo, non legato al nuovo segnale).

Al primo run pulito (v2): OCR grezza assegna 7/23 tracklet, ma solo
**3 identita' finali confermate** (Richards, Courtois, Adams) invece
delle ~7 attese. Prima di concludere che la nuova penalita'
(`narrow_region_jersey_score_penalty`) fosse troppo aggressiva, verificato
il funnel completo (stesso principio "misurare prima di cambiare soglie"
gia' applicato altrove in questo handoff): per il tracklet 15
(jersey=1, votes=33, winner_margin=0.319 - ampiamente sopra anche la
soglia alzata di 0.25), il gate segnava comunque `reliable_jersey=False`.
Causa trovata in `ft/identity/hungarian.py`: la funzione `summarize()`
- che costruisce l'oggetto "tracklet" su cui gira davvero il gate
Hungarian, raggruppando le righe per tracklet - **non ricopiava mai** il
campo `jersey_full_body_sufficient` scritto da `pipeline.py` sulle righe
grezze. Segue lo stesso pattern gia' usato per
`jersey_votes`/`jersey_confidence`/`jersey_head_confidence`/
`jersey_winner_margin` (aggregati con `mean()`/`max_int()` sulle righe
che condividono il `jersey_number` vincente), ma il nuovo campo non era
stato aggiunto a quella lista. Risultato pratico: `tracklet.get(
"jersey_full_body_sufficient")` era sempre `None` durante l'assegnazione
reale, quindi la penalita' non poteva mai scattare - il calo a 3
identita' non era causato dalla nuova penalita' (di fatto inattiva), ma
da un problema pre-esistente (probabile bias "OCR legge 1 quando
incerta", gia' documentato sopra, con 3 tracklet diversi che leggevano
tutti "1").

Fix: aggiunta la riga mancante in `summarize()`,
`mode([row.get("jersey_full_body_sufficient") for row in items if
row.get("jersey_number") == jersey_number and
row.get("jersey_full_body_sufficient") is not None])`, stesso pattern
delle altre metriche jersey aggregate per tracklet.

Rerun (v3) dopo il fix: **7 identita' finali confermate** (Freese #24,
Courtois #1, Richards #3, Adams #4, Dest #2, Trossard #10, Robinson #5),
tutte plausibili rispetto al roster reale. Nota a margine: nello stesso
rerun l'OCR grezza e' salita a 21/23 assegnati (da 7/23) pur non avendo
toccato `jersey_ocr.py` tra le due run - spiegazione piu' probabile:
cache OCR condivisa (`~/FT/.ft_cache`, non legata alla cartella
artifacts) popolata nel frattempo da un'altra run rimasta attiva sullo
stesso video con config quasi identica. Non e' quindi un confronto
pulito v2-vs-v3 sull'effetto isolato del gate, ma il risultato finale
(7 identita' reali, nessun crash, segnale confermato attivo nel gate)
e' l'esito atteso.

**Lezione**: dopo aver aggiunto un nuovo campo per-tracklet in
`ft/pipeline.py`, verificare sempre se esiste un punto di aggregazione
a monte (qui `HungarianPlayerIdentifier.summarize()`) che ricostruisce
l'oggetto "tracklet" da zero raggruppando le righe - un campo scritto
solo sulle righe/track non arriva automaticamente li' se non viene
esplicitamente ricopiato, anche se il resto del codice (gate,
`jersey_reliability()`) lo legge correttamente per nome.

## Aggiornamento 2026-07-24 (3) - Roster arbitro, frammentazione ID chiarita, e trasparenza del gate identita'

### Roster: le particolarita' di ogni partita vanno nel file roster, non nei config di run

Richiesta di aggiungere l'arbitro (rosso, pantaloncini neri, confermato da
un frame reale del video) a Usa-Bel. Primo tentativo: config-level override
di `referee.color_ranges` in `usa_bel_jersey_ocr_all_crops_v1.yaml`.
Corretto dall'utente: la convenzione di questo progetto e' che le
particolarita' di una singola partita (colore portiere, colore arbitro)
vanno nel roster (`metadata.kit_color` su un'entry con `role: goalkeeper`
o `role: referee`), non in un config di run - stessa logica gia' seguita
per i portieri, gia' usata altrove nel roster (`referee_yellow` in
Inter-Atalanta.json). Annullato l'override di config, aggiunta invece
un'entry `referee_red` a `Usa-Bel.json` (`role: referee`,
`metadata.kit_color: "red"`). Un solo colore basta (il rosso da solo e'
gia' distintivo, dato che nessuna delle due squadre gioca in rosso;
il nero avrebbe rischiato di confondersi con la maglia scura USA).

### Frammentazione ID: 33 visti nel video, 23 identita' reali - nessun bug

L'utente ha notato ID fino a "33" nel video renderizzato per un video con
solo ~22 giocatori + arbitro, temendo una frammentazione grave del
tracking. Misurato invece di ipotizzare: `Usa-Bel_linking.json` mostra
**25 track grezzi -> 23 tracklet finali** (solo 2 fusioni necessarie,
entrambe accettate), con 295/298 candidati di fusione scartati per
"gap" - cioe' track che coesistono nello stesso momento, quindi
*correttamente* non fusi (persone diverse viste insieme). 18 dei 27 track
nel CSV durano 450+ frame su 470 totali - tracking stabile.

La spiegazione del "33" visto a occhio: `display_track_id` di default
**e' semplicemente l'ID grezzo del tracker** (`ft/linking/
tracklet_linker.py:194`, `track.setdefault("display_track_id",
int(raw_track_id))`), mai rinumerato da 1 a N. Il tracker assegna un
numero sempre crescente ogni volta che rileva/ritrova qualcosa, anche per
frammenti di pochi frame poi scartati dall'export finale (durata minima).
Vedere "ID33" nel video non significa 33 identita' reali - il contatore
interno era arrivato li', ma diversi numeri intermedi appartenevano a
rilevamenti troppo corti per contare. Nessun intervento necessario su
tracker/linker per questo video.

### Nuovo strumento: `scripts/identity_gate_report.py`

L'utente ha messo in discussione se l'assegnazione Hungarian sia utile o
"faccia solo perdere tempo/risorse" data la complessita' accumulata
(reliable_jersey, goalkeeper_singleton, strong_combined,
narrow_region_jersey_score_penalty...). Prima di ripensare l'architettura,
costruita trasparenza sull'esistente: script che incrocia
`{video}_identity_assignments.json` (assignments, chiavi codificate
`str(-(display_track_id*100000+1))`), `{video}_jersey_ocr.json` e
`{video}_candidate_scores.csv` in un report leggibile per-tracklet (stato,
quale criterio del gate ha deciso, valori esatti contro soglie, evidenza
OCR incluso `full_body_sufficient`, miglior candidato roster). Zero nuova
run necessaria, riusa dati gia' prodotti.

Eseguito su Usa-Bel, due scoperte concrete e azionabili:

1. **`visual_similarity` e' sempre `None`, su tutti i 23 tracklet, senza
   eccezioni.** Il componente "somiglianza visiva" del costo Hungarian e
   l'intero percorso `strong_combined` (che lo richiede) sono
   **strutturalmente inerti**, non solo in questa run - causa: ogni
   giocatore nel roster ha `"visual_embedding": null` (nessun roster
   finora fornisce profili visivi), quindi `visual_distance()` non ha
   nulla con cui confrontare l'embedding del tracklet. Questo non e' un
   componente "poco utile", e' un componente morto per qualsiasi video
   che non fornisca profili visivi nel roster - probabilmente tutti i
   video processati finora.
2. **La vera causa della bassa copertura (7/23 assegnati) e' il bias OCR
   gia' documentato ("legge 1 quando incerta"), non l'architettura del
   gate**: 8 tracklet su 23 (35%) leggono il numero "1" (track 1, 5, 8,
   11, 13, 15, 19, 20). Solo uno (track 17, Courtois) trova un candidato
   roster reale con jersey=1; gli altri 7 restano correttamente
   "insufficient_assignment_evidence" perche' nessun altro giocatore ha
   quel numero - il gate li rifiuta invece di assegnarli a caso, che e'
   il comportamento voluto, non uno spreco.

Anomalia isolata non ancora spiegata: track 10 ha `assignment_gate` mai
valutato (`pass=None`, tutti i campi `None`, `confidence=0.0`) nonostante
un'evidenza OCR forte (numero=7, 22 voti) - probabilmente escluso dal
matching Hungarian per qualche motivo strutturale (matrice non quadra,
ruolo non-player), da investigare separatamente se rilevante.

**Conclusione provvisoria sulla domanda "le varie parti aiutano o
sprecano risorse?"**: il gate (reliable_jersey/goalkeeper_singleton)
sembra funzionare correttamente e conservativamente dato il bias OCR
presente; il componente visivo/strong_combined e' morto per mancanza di
dati roster, non per un difetto della logica stessa. Prossimo passo
concordato con l'utente: investigare il bias OCR sul numero "1" prima di
decidere se investire nel popolare profili visivi nel roster.

### Bias OCR sul numero "1": diagnosticato, correzione testata e RESPINTA

Su Usa-Bel, 8/23 tracklet (35%) hanno vinto il voto con numero "1".
Ispezionata la distribuzione completa dei candidati (non solo il
vincitore) per tutte e 8: track 1 mostra il pattern gia' noto di
troncamento `number_band` ("1": 51 voti vs "10": 34 voti, margine 0.025 -
quasi un pareggio, "10" letto parzialmente come "1"); le altre 7 mostrano
"1" vincere su alternative diverse ogni volta (7, 4, 3, 6...) senza un
pattern fisso di confusione tra due cifre specifiche - coerente col bias
gia' documentato su GSR (`confusion.csv`: molte cifre vere ->1) come
"risposta di ripiego" del modello quando incerto, non una confusione
visiva specifica.

Testata una correzione mirata (`scripts/test_distrust_one_bias.py`,
stessa metodologia offline zero-nuova-inferenza gia' validata per
`full_body_sufficient`) su entrambi i campioni GSR: se il vincitore e' 1
con `winner_margin` sotto soglia, o (a) scartarlo (`demote`) o (b) usare
il secondo candidato al suo posto (`promote_runner_up`).

- `demote`: su entrambi i campioni il numero di risposte corrette resta
  **identico a ogni soglia** (251/830 e 57/157) - tutti gli "1" scartati
  erano comunque sbagliati, quindi `accuracy_assigned` sale solo perche'
  il denominatore (copertura) scende; `accuracy_all` resta piatta. Nessun
  guadagno reale, solo un miglioramento cosmetico della metrica.
- `promote_runner_up`: sul campione da 50 sequenze guadagna **+10
  risposte corrette** (251->261, `accuracy_all` 30.2%->31.4%),
  sembrava confermare l'ipotesi di troncamento. Ma sul campione
  indipendente da 10 sequenze **il guadagno e' zero a qualsiasi soglia**
  (57->57, invariato) - non si riproduce, a differenza di
  `full_body_sufficient` che aveva replicato in modo quasi identico su
  entrambi i campioni.

**Decisione: RESPINTA, non implementata.** Non supera la stessa barra di
validazione (conferma su un secondo campione indipendente) gia' applicata
con successo a `full_body_sufficient` in questa stessa sessione. Il
miglioramento visto sul campione da 50 sequenze e' piu' probabilmente un
artefatto di quel campione specifico che un effetto generalizzabile. Non
riprovare senza nuova evidenza.

### Portiere: il bug "squadra sbagliata" non si riproduce su Usa-Bel v3

Controllato `Usa-Bel_goalkeeper_colour.json`: solo il track 18 ha uno
score sostanziale (0.499, `is_goalkeeper_palette: True`) con
`team_id: 1` (USA) - **gia' corretto**, non invertito su Belgio come
temuto. Il rilevamento colore/squadra del portiere funziona bene in
questa run.

Il comportamento confuso osservato (Freese assegnato a un tracklet
diverso da quello visivamente portiere) ha una spiegazione diversa e piu'
semplice: track 18 (il vero portiere) legge dall'OCR il numero "4"
(sbagliato, comune per i portieri: crop piccoli/angolazioni difficili),
quindi il gate lo lascia "unknown". Track 21 (un tracklet diverso, non
marcato come portiere per colore) legge correttamente "24" e vince
l'assegnazione a Freese tramite `reliable_jersey`, perche' l'assegnazione
Hungarian e' uno-a-uno globale: chi ha il costo piu' basso su un giocatore
si prende quello slot, lasciando l'altro tracklet senza identita' invece
di due tracklet con lo stesso nome. Non e' un bug della logica - il bug
originario "squadra sbagliata" temuto dall'utente era probabilmente
specifico di una run precedente (v1/v2, prima dei fix di oggi) o una
confusione con il caso analogo gia' documentato su Por-Uzb (verde
fosforescente/prato).

### Nota per il futuro: `escalate_on_unassigned` e' inerte sotto il profilo "tutti i crop"

Prima di riabilitare `escalate_on_unassigned` su
`usa_bel_jersey_ocr_all_crops_v1.yaml` per provare a recuperare copertura,
notato che il profilo ha gia' `max_crops_per_tracklet: 100000` (legge
gia' tutti i crop disponibili al primo giro). L'escalation pesca crop
*non ancora usati* quando il primo voto fallisce - con "tutti i crop"
gia' letti, non resta nulla da pescare in piu': il flag non avrebbe
nessun effetto in questa combinazione (stesso motivo per cui era gia'
lasciato disattivato di proposito in `por_uzb_jersey_ocr_all_crops_v1.yaml`).
Il test corretto dell'escalation richiede un budget di crop limitato
(es. il default) con l'escalation attiva sopra - verifica se recupera
una copertura simile a "tutti i crop" ma a costo inferiore, non un
miglioramento aggiuntivo sopra "tutti i crop" gia' letto. Decisione
rimandata all'utente: rilanciare Usa-Bel identico (v3, verifica
riproducibilita') oppure con budget limitato + escalation (vero test
dell'ipotesi).

### Stato a fine sessione 2026-07-24

Lavoro completato e verificato: segnale `full_body_sufficient`
implementato end-to-end (jersey_ocr.py -> pipeline.py ->
hungarian.py, bug di propagazione trovato e corretto); arbitro
aggiunto al roster Usa-Bel (rosso, via `metadata.kit_color`, non
config); frammentazione ID chiarita (23 identita' reali, non 33 -
l'ID visto a video e' quello grezzo del tracker, mai rinumerato);
bias OCR sul numero "1" diagnosticato e correzione testata e respinta;
bug "portiere squadra sbagliata" non riprodotto nella run attuale;
nuovo strumento `scripts/identity_gate_report.py` per trasparenza
del gate identita' (riusabile su qualsiasi run futura). Scoperta
aperta e non ancora sfruttata: `visual_similarity`/`strong_combined`
sono strutturalmente inerti per mancanza di profili visivi nel roster
- potenziale leva piu' alta per aumentare la copertura identita', ma
richiede capire da dove ottenere le foto di riferimento dei giocatori.
Decisione pendente sulla prossima run Usa-Bel: ripetizione identica
(v3) vs budget crop limitato + escalation attiva (test vero
dell'ipotesi di risparmio).

## Aggiornamento 2026-07-25 - Budget crop stretto su Usa-Bel, revisione hungarian.py, soglia reliable_jersey gia' al pavimento

### Usa-Bel: `max_crops_per_tracklet=5` collassa la coverage OCR, non e' un bug di overlay

Lanciata la run pendente (`usa_bel_jersey_ocr_top5_crops_v1.yaml`, unica
variabile cambiata vs il profilo "tutti i crop" gia' validato:
`max_crops_per_tracklet: 5` invece di `100000`, resto identico incluso
`escalate_on_unassigned: false`). Al primo sguardo il video renderizzato
non mostrava numeri; verificato con `identity_gate_report.py` che non e'
un problema di overlay ma un collasso reale a monte, nell'OCR:

```text
                                   baseline (tutti i crop)   top-5
tracklet assegnate                        7/23 (30%)         2/22 (9%)
tracklet con zero rilevazioni OCR            0/23            12/22 (55%)
tracklet con voti sotto soglia (1 voto)      0/23             5/22 (23%)
```

17/22 tracklet (77%) non producono nemmeno una cifra rilevata con budget
5, anche quelle con 400-1200 frame disponibili. Causa probabile, coerente
con un risultato gia' documentato (sezione 7.1 del brief): la selezione
dei 5 crop (`select_spread_crops`, spread temporale + `crop_quality`) non
e' un proxy affidabile per "questo crop mostra un numero leggibile" -
stesso motivo per cui i classificatori di legibility testati in passato
(clean-back AP 0.932, readability accuracy 0.981) miglioravano poco o
nulla le decisioni OCR track-level quando collegati. Con "tutti i crop" il
problema non si pone; con un budget stretto, *quali* crop leggere diventa
il collo di bottiglia, e l'euristica attuale non e' adatta a questo scopo.
Non testato `escalate_on_unassigned` sopra il budget stretto: il problema
sembra "quali crop", non "quanti", quindi probabilmente non avrebbe
aiutato granche' - da verificare solo se si decide di investire in una
selezione migliore a monte.

### Revisione di `ft/identity/hungarian.py`: due soglie "reliable_jersey" quasi omonime, una delle due strutturalmente morta

Richiesta esplicita dell'utente di rileggere il file perche' "confusionario".
Letto tutto (900 righe), trovati questi punti concreti:

1. **`_has_reliable_jersey(tracklet)`** (riga 455, 4 parametri costruttore:
   `reliable_jersey_min_votes=5`, `_min_confidence`, `_min_head_confidence`,
   `_min_winner_margin=0.10`) controlla la qualita' intrinseca della lettura
   OCR della tracklet. **`_has_reliable_jersey_match(tracklet, player)`**
   (riga 520, soglia diversa: `reliable_jersey_min_candidate_score=0.45`)
   controlla se l'evidenza supporta *quel* giocatore specifico. Il gate
   finale (`assignment_gate`, riga 478) chiama **solo la seconda**. La prima
   viene usata altrove per un compito diverso (bonus/penalita' nella cost
   matrix, riga 372) e raggiunta dalla seconda solo come fallback raro
   (riga 542, quando la distribuzione grezza OCR e' vuota). Risultato
   pratico: i 4 parametri di `_has_reliable_jersey` sembrano governare "cosa
   serve per fidarsi del numero nel gate identita'" ma quasi mai lo fanno -
   stesso tipo di scoperta gia' fatta per `visual_similarity`/`strong_combined`
   (morti per mancanza di embedding nel roster).
2. **Ipotesi plausibile (non ancora confermata su `candidate_scores.csv`)**
   per il mistero del track 1 Zielinski su Inter-Juve (assegnato a Zielinski
   nonostante l'OCR avesse votato "2", il numero di Dumfries, come vincitore):
   `identity_gate_report.py` stampa solo il numero *vincitore* della
   tracklet, ma il gate valuta `jersey_candidate_score` sul candidato
   specifico per il numero atteso (7, Zielinski) dentro la distribuzione
   grezza - che puo' avere voti minoritari sufficienti anche se non e' il
   vincitore assoluto. Non un bug, ma un buco di trasparenza nel report (non
   mostra lo score del candidato realmente valutato dal gate, solo il
   vincitore).
3. La costante di saturazione voti (`min(1.0, votes/3.0)`) e' hardcoded e
   duplicata identica in 4 punti (righe 836, 789, 824, 875), mai
   fattorizzata ne' esposta come parametro di config.
4. Gestione di `preserved_dropped_jersey_evidence` duplicata in due forme
   diverse (flag booleano threaded a parametro nella cost matrix, early
   return hardcoded nel gate) per la stessa regola.

Deciso di non fare un rewrite ampio subito (le soglie sono il prodotto di
sessioni di misura reale, commenti a riga 65-69 e 877-881 citano percentuali
validate). Proposto un primo giro a basso rischio (rinominare/documentare le
due funzioni, fattorizzare la costante voti/3, marcare esplicitamente inerte
il ramo `visual_similarity`) ma l'utente ha preferito prima un test a
impatto diretto sulla coverage (vedi sotto) prima di tornare al cleanup.

### Sweep offline `reliable_jersey_min_candidate_score` su Int-Ata: leva gia' al pavimento pratico

Ipotesi: se il vero gate che decide l'assegnazione e' `_has_reliable_jersey_match`
(soglia `reliable_jersey_min_candidate_score`, non i 4 parametri "morti"
sopra), forse abbassarla recupera coverage reale. Test offline, zero nuova
run GPU, stessa metodologia gia' validata per `full_body_sufficient`/il bias
"1": nuovo script `scripts/sweep_reliable_jersey_min_candidate_score.py`,
riusa direttamente `evaluate_identity_units()`/`identity_metrics()` da
`ft/evaluation/identity_benchmark.py` (lo stesso codice di una promozione
ufficiale) e `jersey_candidate_score()` da `hungarian.py`, patchando solo
`player_id` per le tracklet che il gate accetterebbe a soglia piu' bassa.
Eseguito solo su **Int-Ata** (unico dei 3 video del benchmark identita' in
split `development` - stessa regola "non tarare sul test" gia' seguita per
GSR, Inter-Juve/Inter-Atalanta non toccate).

**Bug di metodologia trovato e corretto nello script stesso, prima di
fidarsi dei numeri**: la prima versione derivava "gia' assegnato" da
`identity_assignments.json` (solo stadio Hungarian grezzo) e trovava 0/78
tracklet gia' assegnate, mentre il baseline reale (`tracklets.csv`, output
finale) ne mostrava 30 - segno che una parte consistente delle identita' di
Int-Ata arriva da uno stadio successivo di propagazione/linking, non dal
gate Hungarian isolato. La prima versione dello script stava percio'
sovrascrivendo tracklet gia' correttamente identificate dalla propagazione
con ipotesi di flip proprie. Corretto: "gia' assegnato" ora si legge
direttamente da `tracklets.csv` (verita' finale), non da
`identity_assignments.json`. Dopo il fix: 26/78 gia' assegnate (vicino alle
30 unit del baseline benchmark, scarto residuo plausibile per via
dell'aggregazione in "unit" del benchmark).

**Scoperta preliminare**: la soglia realmente usata in produzione su
Int-Ata e' **0.2**, non 0.45 come da `config.py`/`default.yaml` (default
ormai superato/non rappresentativo - nessuna run reale lo usa). Controllato
anche Inter-Juve, Inter-Atalanta, Usa-Bel (x2 run): **0.2 ovunque, nessun
drift reale**. Le uniche differenze apparenti (`narrow_region_jersey_score_penalty`
None su Int-Ata/Inter-Juve/Inter-Atalanta vs 0.15 esplicito su Usa-Bel;
`assignment_scope` None su Int-Ata/Inter-Atalanta vs valore esplicito
altrove) sono quasi certamente solo "non sovrascritto nel manifest", non
comportamento diverso a runtime.

Risultati sweep (soglie sotto 0.2 = piu' permissivo del reale; baseline
reale: 30 unit assegnate, 11 corrette, 19 sbagliate, precision 36.7%):

```text
soglia   flip unknown->assigned   nuove corrette   nuove sbagliate   precision
0.30              16                    5                12           34.0%
0.25              21                    5                16           31.4%
0.20 (reale)      29*                   5*               ...          -
0.15              37                    5                35           22.9%
```

*a soglia 0.2 (= produzione) la simulazione offline non trova 0 flip come
dovrebbe teoricamente, confermando che lo script e' un'approssimazione del
gate reale (non verifica `unknown_threshold=0.55` sul cost-matrix ne' le
collisioni Hungarian quando piu' tracklet flippate competono per lo stesso
giocatore) - i numeri assoluti vanno presi come indicativi, non esatti.

Pattern qualitativo pero' robusto su tutto il range testato: `newly_correct`
si ferma a 5 dopo soglia 0.3 e non cresce piu', mentre `newly_wrong`
continua a salire senza limite (12 -> 16 -> 27 -> 35). **Conclusione:
questa leva e' gia' stata sfruttata fino al suo pavimento pratico** -
abbassare ulteriormente `reliable_jersey_min_candidate_score` sotto il
valore gia' in uso (0.2) non aggiunge identita' corrette, solo errori.
Non e' la via per un guadagno di coverage significativo. Prossimo passo
concordato con l'utente: da decidere tra tornare al cleanup di
`hungarian.py` o investigare un'altra leva (es. profili visivi nel roster
per sbloccare `strong_combined`, gia' individuata come scoperta aperta il
24/07).
