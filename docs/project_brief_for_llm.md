# Football Player Identification — briefing per un nuovo LLM

## 1. Finalità del progetto

Il progetto studia la ricostruzione automatica dello stato di una partita di
calcio a partire da video broadcast. L'obiettivo non è soltanto rilevare e
seguire persone e pallone, ma associare alle track dei calciatori informazioni
semantiche affidabili:

- ruolo: giocatore, portiere o arbitro;
- squadra;
- numero di maglia;
- identità reale nel roster;
- posizione sul terreno di gioco, quando la calibrazione lo consente.

L'output desiderato è una sequenza di osservazioni utilizzabile per analisi
tattica e Game State Reconstruction, accompagnata da artifact che permettano di
ricostruire perché una decisione è stata presa.

Il principio guida è conservativo: una identità `unknown` è preferibile a una
identità errata. Detection, tracking, numero di maglia e identità sono problemi
distinti; il successo di uno non implica il successo degli altri.

## 2. Domanda scientifica attuale

La parte oggi più avanzata del lavoro riguarda il riconoscimento del numero di
maglia. La domanda è:

> Dati più frame appartenenti alla stessa track, è possibile localizzare la
> regione del numero e riconoscerla in modo sufficientemente accurato e
> generalizzabile da contribuire, in sicurezza, all'identificazione del
> giocatore?

Il problema è difficile perché nei video televisivi il numero è piccolo,
sfocato, deformato dalla posa, parzialmente occluso e visibile soltanto quando
il giocatore mostra la schiena. Una track può contenere centinaia di frame, ma
solo pochi sono realmente informativi.

Il miglior risultato corrente nasce da tre idee:

1. scegliere pochi frame promettenti per ogni track;
2. rilevare esplicitamente la regione del numero nel crop del giocatore;
3. riconoscere la sequenza numerica con un CTC preaddestrato su un dataset
   specifico di numeri di maglia e poi adattato a SoccerNet-GSR.

## 3. Pipeline end-to-end

```text
video broadcast
  -> detector YOLO di persone/pallone
  -> ByteTrack
  -> tracklet linking conservativo
  -> team, arbitro e portiere
  -> selezione dei frame per ogni track
  -> detector YOLO della regione del numero
  -> recognizer numerico ResNet18 + BiLSTM + CTC
  -> aggregazione probabilistica multi-frame
  -> evidenza numero di maglia
  -> assegnazione roster con Hungarian e vincoli
  -> video annotato + CSV/JSON di provenance
```

### Detection e tracking

- YOLO produce detection di giocatori e pallone.
- ByteTrack è il tracker operativo validato.
- Il linking successivo può ricomporre tracklet interrotte, ma deve rispettare
  distanza temporale, geometria, squadra e compatibilità visuale.
- I cambi di inquadratura sono trattati esplicitamente; non si deve propagare
  una identità attraverso un cut senza evidenza sufficiente.

### Evidenze semantiche

- La squadra viene stimata soprattutto dai colori del crop.
- Arbitri e portieri sono gruppi semantici separati.
- OCR e numero di maglia sono evidenze, non identità definitive.
- Il roster restringe i numeri plausibili per squadra.
- L'assegnazione finale usa Hungarian e vincoli duri contro duplicati e
  incompatibilità di ruolo.

### Audit, propose e apply

- `audit`: calcola ed esporta evidenze senza cambiare l'output operativo;
- `propose`: produce candidati da revisionare, senza applicarli;
- `apply`: modifica realmente jersey o identità.

I nuovi moduli devono partire in `audit`. Una promozione richiede miglioramento
misurato, provenance completa e assenza di mutazioni collaterali.

## 4. Dati e protocollo sperimentale

### SoccerNet-GSR

SoccerNet-GSR fornisce immagini, bounding box, track ID, ruolo, squadra e numero
di maglia. È il dominio principale del progetto.

Gli esperimenti jersey sono separati per sequenza, non casualmente per crop,
per evitare che frame quasi identici della stessa partita finiscano in train e
validation.

```text
GSR train OCR/readability: 37 sequenze, 311 track, 624 immagini annotate
development interno:        9 sequenze
frozen interno:            11 sequenze, 173 track
GSR test ufficiale:        49 sequenze, 774 track, 2954 crop selezionati
```

Il frozen è stato usato una volta dopo la selezione della configurazione. Il
test ufficiale è stato eseguito una sola volta con configurazione bloccata. Non
deve essere usato per scegliere nuove soglie o checkpoint.

### SJN-210k

SJN-210k è un dataset esterno specifico per numeri di maglia. È stato scelto al
posto di MNIST o CIFAR perché riduce il domain gap: contiene numeri su maglie,
pose e sfondi sportivi.

```text
split sorgente train:          14956 immagini
split sorgente official test:   6454 immagini
regioni localizzate train:       8938
regioni localizzate validation:  3789
```

Lo split denominato `test` dagli autori è stato usato come validation per il
pretraining e non viene presentato come test finale indipendente. Il vero test
finale del progetto resta GSR.

## 5. Configurazione jersey migliore e congelata

```text
frame selection: configurazione congelata
detector regione: YOLO26s fine-tuned su regioni GSR
detector confidence: 0.25
padding regione: 0.25
recognizer: ResNet18 + BiLSTM + CTC
pretraining recognizer: SJN-210k
fine-tuning recognizer: regioni GSR manuali
aggregazione: probabilistica a livello di track
runtime: audit-only
```

Checkpoint:

```text
models/jersey_number_ctc_sjn_to_gsr_v1.pth
sha256 11c6356895c94e734b795d9ca2403580187d1d726c5aada51273738042c7c1c5

runs/jersey_number_region/yolo26s_smoke_v1/weights/best.pt
sha256 ec3c4d06df75aff92f49779827e6cff96d99b84da496ced8a76ecfbb12c02ae4
```

Nel confronto finale baseline e candidato condividono frame, detector,
regioni, padding e aggregazione. Cambia soltanto il checkpoint CTC. Questo rende
attribuibile al pretraining SJN e al fine-tuning GSR il miglioramento osservato.

## 6. Risultati principali

### Development con regioni manuali, 46 track

```text
                         CTC GSR precedente   CTC SJN->GSR
crop corretti                 61/92              77/92
track corrette                31/46              40/46
track accuracy                67.39%              86.96%
GT nel top-5                  40/46              45/46
```

### Frozen, 11 sequenze e 173 track

```text
                         CTC GSR precedente   CTC SJN->GSR
region coverage                81.62%             81.62%
track coverage                 91.91%             91.91%
corrette                       84                 124
errate                         75                  35
accuracy all                   48.55%             71.68%
accuracy assigned              52.83%             77.99%
GT nel top-5                  124/173            146/173
```

Transizioni paired: 45 wrong-to-correct, 5 correct-to-wrong, guadagno netto
`+40`; test binomiale esatto `p = 4.21e-09`.

### Test ufficiale GSR, 49 sequenze e 774 track

```text
                         CTC GSR precedente   CTC SJN->GSR
regioni rilevate             1803               1803
region coverage                61.04%             61.04%
track coverage                 78.17%             78.17%
corrette                      249                442
errate                        356                163
accuracy assigned              41.16%             73.06%
accuracy all                   32.17%             57.11%
GT nel top-5                  386/774            521/774
```

Transizioni paired:

```text
correct->correct 236
wrong->correct   206
correct->wrong    13
guadagno netto  +193
```

Bootstrap paired per sequenza, 10000 repliche:

```text
delta accuracy all medio: +0.2492, IC95% [0.2222, 0.2776]
delta GT top-5 medio:      +0.1744, IC95% [0.1480, 0.2022]
test binomiale esatto:     p = 7.54e-46
```

Conclusione: il trasferimento SJN->GSR migliora molto il riconoscimento senza
aumentare la coverage. La parte non osservata rimane un problema di selezione e
localizzazione, non di classificazione CTC.

## 7. Esperimenti svolti e decisioni

### 7.1 Selezione dei frame

Sono stati confrontati:

- ResNet34 legacy di legibility;
- YOLO11s classification `clean_back`;
- YOLO26s classification `number_readable`.

Il classificatore clean-back ha ottenuto AP `0.932` e top-5 hit `1.0` su 38
track validation. Il readability classifier ha ottenuto accuracy `0.981`, AP
`0.999` e top-5 hit `1.0` su 61 track. Tuttavia, quando collegati allo stesso
OCR, il miglioramento delle decisioni track-level è stato piccolo o negativo.
La selezione visiva è utile, ma non era il principale collo di bottiglia.

### 7.2 SAR fine-tuned

Un recognizer SAR MMOCR è stato fine-tuned su crop GSR leggibili. Ha migliorato
nettamente il SAR pretrained, ma ha mostrato bias verso numeri frequenti, in
particolare `10`, e nuove emissioni errate. È rimasto recognizer diagnostico e
non è stato promosso automaticamente.

### 7.3 Pseudo-label e PU learning per la qualità dei crop

Le pseudo-label basate sull'accordo di più OCR sono fallite perché quasi tutti
i crop avevano soltanto EasyOCR disponibile. Consentendo una singola sorgente
contro il GT sono state ottenute label positive e hard negative, poi usate per
un classificatore PU ResNet18. Il modello tendeva a predire quasi tutto come
positivo e non ha superato la soluzione supervisionata YOLO. Percorso
abbandonato.

### 7.4 Detector della regione del numero

Sono stati addestrati YOLO26s su regioni GSR manuali e su SJN-210k. Il detector
SJN ha buona capacità nel proprio dominio, ma il trasferimento diretto a GSR ha
prodotto regressioni. Il detector GSR smoke resta quello congelato perché nel
confronto controllato il collo di bottiglia decisivo era il recognizer.

### 7.5 Preprocessing delle regioni

Sono stati provati padding `0.10`, `0.25`, `0.40`, upscale 4x, colore,
grayscale e CLAHE. Il compromesso congelato è padding `0.25`, colore, upscale
come parte della configurazione. CLAHE e selezione dinamica delle varianti non
hanno fornito un vantaggio sufficientemente stabile.

### 7.6 CTC preaddestrato su SJN e fine-tuned su GSR

È l'esperimento riuscito principale. Un primo smoke SJN di tre epoche ha
raggiunto accuracy validation `0.8659`; il training completo è arrivato a
`0.9211`. Il successivo fine-tuning GSR ha prodotto il checkpoint corrente e
ha superato la baseline su development, frozen e test ufficiale.

### 7.7 Modello multi-frame con attention e teste digit-level

È stato provato:

```text
encoder ResNet18 trasferito dal CTC
-> attention sui frame
-> teste length / tens / units
```

Su una superficie di crop readable ha ottenuto `55.74%`, contro `8.20%` della
baseline CTC valutata su quella superficie non equivalente. Sulle regioni
manuali corrette, però, è crollato a `30.43%` contro `86.96%` del CTC, con 26
regressioni e nessun recupero. Le nuove teste erano inizializzate casualmente e
225 track GSR non bastavano per imparare il riconoscimento. Esperimento
respinto; non deve essere usato su frozen, test o runtime.

### 7.8 Fusione digit-level dei logits CTC

La distribuzione CTC è stata rifattorizzata in lunghezza, decine e unità e
interpolata con l'originale. Lo sweep ha selezionato peso `0.0`: nessuna
correzione e nessuna regressione. La procedura riordinava la stessa informazione
senza aggiungere nuova evidenza visuale. Esperimento neutro, non promosso.

### 7.9 PRTReID e associazione appearance-based

PRTReID è stato integrato e verificato: coverage embedding `1.0`, dimensione
256 e norma corretta. Uno shadow tracker che combinava appearance, IoU e moto
ha però peggiorato nettamente ByteTrack su SNGS-025:

```text
ByteTrack ID switches: 45
shadow ID switches:   309
ByteTrack purity:      0.9312
shadow purity:         0.8740
```

Il prototipo non replica StrongSORT e non deve essere chiamato così. ByteTrack
resta il tracker; PRTReID rimane una evidenza sperimentale opt-in.

### 7.10 Tracking benchmark

Sul benchmark GSR controllato di 12 sequenze, la baseline display-track ha
ottenuto circa:

```text
HOTA  0.6018
AssA  0.5167
IDF1  0.7074
```

Il collo di bottiglia è l'associazione (`AssA`), non soltanto la detection.
Sweep di confidence e varianti di linking hanno migliorato leggermente DetA o
MOTA, ma HOTA, AssA e IDF1 sono rimasti neutri o sono peggiorati. MOTA non va
usata da sola per dichiarare un miglioramento.

## 8. Verifica su video reale Inter–Juve

Il CTC SJN->GSR è stato integrato in runtime in modalità audit-only e provato
su 1500 frame di Inter–Juve:

```text
crop selezionati:          201
regioni rilevate:           16
region detection coverage:  7.96%
track con predizione CTC:     7
accordi SAR/CTC:              3
disaccordi:                   4
identity mutations:           0
```

Il risultato evidenzia un forte domain gap nel detector della regione. Nel caso
qualitativo della track 42, il numero è visivamente `95`, ma il CTC produce
`55` come top-1 su più frame, con `95` nel top-5. Il modello confonde `9` e `5`
ed è talvolta overconfident. Per questo non è abilitato alcun override runtime.

## 9. Cosa è validato e cosa non lo è

### Validato

- ByteTrack come tracker operativo.
- Protocollo sequence-disjoint e confronto paired.
- Detector GSR della regione con soglia `0.25` e padding `0.25` per la
  configurazione congelata.
- CTC SJN->GSR come miglior recognizer numerico offline su GSR.
- Integrazione runtime audit-only senza mutazioni di jersey o identità.
- Provenance di crop, detector, recognizer, score e checkpoint.

### Non validato per apply

- sostituzione automatica del numero SAR con il CTC;
- utilizzo del CTC per cambiare direttamente la real identity;
- generalizzazione robusta ai video custom;
- shadow tracker appearance-based;
- modello multi-frame con nuove teste digit-level;
- fusione dei logits CTC;
- threshold derivati dal test ufficiale.

## 10. Collo di bottiglia attuale

Su GSR il recognizer è migliorato molto, ma la coverage delle regioni è circa
`61%`; su Inter–Juve scende a circa `8%`. Quindi il prossimo limite è ottenere
regioni informative nel dominio reale, non spremere ulteriormente gli stessi
logits CTC.

I failure mode principali sono:

- nessun frame con schiena e numero leggibile;
- detector della regione che non trova il numero;
- crop troppo piccolo o sfocato;
- confusione tra cifre simili, soprattutto `5` e `9`;
- aggregazione dominata da predizioni ripetute ma errate;
- domain gap fra GSR/SJN e riprese custom;
- contaminazione della track o cambio di inquadratura.

## 11. Direzione di lavoro raccomandata

La configurazione testata non deve essere ritoccata sul GSR test. Le nuove idee
vanno sviluppate su train/development e confrontate prima sul frozen.

Priorità suggerite:

1. migliorare la copertura del detector di regione su domini custom, creando un
   piccolo validation set separato per video reali;
2. calibrare confidenza e astensione del CTC, soprattutto per `5`/`9`;
3. se si riprende il multi-frame, preaddestrare su SJN anche le teste
   `length/tens/units`, poi trasferire l'intero recognizer e apprendere soltanto
   l'aggregazione su bag GSR;
4. valutare aggregazione multi-frame che conservi il CTC addestrato, invece di
   sostituirlo con teste casuali;
5. mantenere il risultato runtime in audit finché un esperimento paired non
   dimostra che gli override aggiungono più correzioni che regressioni;
6. per il tracking, lavorare su errori di associazione misurati da AssA/IDF1,
   non riattivare automaticamente PRTReID o linking aggressivi.

## 12. Regole metodologiche per chi prosegue

- Leggere gli artifact reali prima di proporre cambiamenti.
- Cambiare una sola variabile per ablation quando possibile.
- Confrontare metodi sugli stessi frame e sulle stesse track.
- Fare split per sequenza, mai per crop casuale.
- Calcolare bootstrap per sequenza, non per crop.
- Riportare coverage, accuracy assigned e accuracy all-track.
- Riportare sempre transizioni paired `correct->wrong` e `wrong->correct`.
- Non presentare lo split SJN official test come test finale indipendente.
- Non usare il GSR test per tuning successivo.
- Non confondere GT nel top-5 con decisione top-1 corretta.
- Non interpretare confidence alta come correttezza garantita.
- Non applicare nuove evidenze all'identità prima di un audit senza mutazioni.
- Preferire `unknown` a una assegnazione non sufficientemente supportata.

## 13. File di riferimento

```text
handoff.md
README.md
docs/thesis/jersey_experiments_results.tex
docs/gsr_detection_tracking_benchmark.md

ft/features/jersey_number_ctc.py
ft/features/jersey_region_ctc_audit.py
ft/features/jersey_frame_selector.py
ft/identity/hungarian.py
ft/validation.py

scripts/train_jersey_number_ctc.py
scripts/evaluate_jersey_number_ctc.py
scripts/evaluate_jersey_number_region_ctc_ocr_run.py
scripts/aggregate_jersey_thesis_benchmark.py
```

`handoff.md` contiene la cronologia completa e i comandi operativi. Questo
briefing contiene invece la finalità, lo stato scientifico e le decisioni che
un nuovo LLM deve comprendere prima di intervenire.
