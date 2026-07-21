# Handoff FT - aggiornato 2026-07-21

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
