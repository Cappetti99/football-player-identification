# Benchmark GSR di detection e tracking

## Obiettivo

Questo benchmark misura detector e tracker sugli stessi video annotati di
SoccerNet-GSR, prima di qualsiasi lavoro su OCR o identità. Il ground truth non
entra mai nella pipeline FT: viene letto soltanto dall'evaluator offline dopo
che il CSV delle tracklet è stato congelato.

Il tracking viene misurato su due superfici distinte:

- `raw_track_id`: uscita del tracker, prima del linking FT;
- `display_track_id`: identità temporale finale, dopo il linking FT.

La differenza fra le due indica se il linker FT aiuta o danneggia il tracker.
Il target GT primario comprende `player`, `goalkeeper` e `referee`, cioè le
classi che FT deve tracciare; il ruolo SoccerNet `other` è escluso e la lista
effettiva viene registrata nel `summary.json`.

## Protocollo

1. Selezionare un pilot deterministico di 12 sequenze da `valid`, senza leggere
   il contenuto delle annotazioni.
2. Congelare modello, config e lista di sequenze nel provenance.
3. Eseguire una baseline completa sui 750 frame.
4. Aggregare risultati macro, micro e intervalli bootstrap al 95%.
5. Modificare una sola variabile per esperimento e rieseguire esattamente le
   stesse sequenze.
6. Confrontare baseline e candidata con delta appaiati per sequenza.
7. Estendere all'intero split `valid` solo dopo aver stabilizzato il workflow.
8. Lasciare `test` congelato fino alla scelta finale.

Sia il builder del manifest sia il runner rifiutano lo split `test` senza il
flag esplicito `--allow-test`.

## Metriche prodotte

Detection:

- le box arrivano da `*_detections.csv`, esportato subito dopo YOLO e prima di
  ByteTrack; le tracklet non vengono usate come surrogato del detector;
- precision, recall, F1 e mean IoU a IoU 0.50 e 0.75;
- precision/recall/F1 usano la soglia operativa del detector (`0.05` nel
  profilo corrente); AP50 e AP75 usano invece tutte le detection esportate
  dal floor di inferenza YOLO (`ball_confidence`, attualmente `0.002`);
- falsi positivi e falsi negativi per frame;
- recall e precision per bbox piccola (`<0.5%` dell'immagine), media
  (`0.5–2%`) e grande (`>=2%`);
- breakdown per ruolo predetto/annotato.

Tracking:

- HOTA, DetA, AssA e LocA con global alignment compatibile TrackEval, mediati
  sulle soglie IoU `0.05:0.95`;
- MOTA e MOTP a IoU 0.50;
- ID precision, ID recall e IDF1 tramite assegnazione globale delle identità;
- ID switch e frammentazioni, anche per 1.000 detection GT;
- mostly tracked, partially tracked, mostly lost, coverage e association purity.

Nota: MOTA può essere negativa quando `FP + FN + IDSW` supera il numero di
detection GT. Non va quindi forzata nell'intervallo 0–1.

## Artifact

Per ogni sequenza:

- `*_detections.csv` negli artifact FT: box YOLO pre-tracker e confidence;
- `summary.json`: tutte le metriche;
- `frame_matches.csv`: associazioni GT→box del tracker a IoU 0.50;
- `detection_frame_matches.csv`: associazioni GT→detection YOLO a IoU 0.50;
- `frame_metrics.csv`: TP, FP, FN e qualità per frame;
- `tracking_frame_metrics.csv`: qualità delle box emesse dal tracker per frame;
- `precision_recall.csv`: curve PR a IoU 0.50/0.75.

Aggregati:

- `per_sequence.csv`;
- `aggregate.json` e `metric_table.csv`;
- grafici PNG e SVG in `charts/`.

## Audit calibrazione e gate fisico del linker

La ground truth `bbox_pitch` SoccerNet e' centrata sul centrocampo; prima del
confronto viene convertita nel sistema FT `[0,105] x [0,68]` metri. L'evaluator
esporta anche `pitch_frame_metrics.csv`. Il gate velocita' e' solo audit: non
modifica `display_track_id` e non usa GT per calcolare la decisione.

Riutilizzo della run congelata confidence `0.12`, senza nuova inferenza:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/cappetti/FT \
python3 scripts/run_gsr_pitch_link_audit.py \
  --manifest evaluation/detection_tracking_manifests/valid_pilot12_v1.json \
  --run-name gsr_valid_pilot12_conf012_v1 \
  --detection-confidence 0.12 \
  --max-gap 90 \
  --max-speed-mps 12.0

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/cappetti/FT \
python3 scripts/aggregate_gsr_pitch_link_audit.py \
  --audit-root evaluation_outputs/calibration_link_audit/gsr_valid_pilot12_conf012_v1 \
  --output-dir evaluation_outputs/calibration_link_audit/gsr_valid_pilot12_conf012_v1_aggregate
```

Artifact:

- `evaluation/summary.json` e `pitch_frame_metrics.csv` per sequenza;
- `link_audit/pitch_link_candidates.csv`;
- `link_audit/pitch_link_audit.json`;
- aggregato `per_sequence.csv`, `aggregate.json` e grafici `pitch_error.png`,
  `blocked_links.png`.

Non applicare il gate se blocca anche un solo link GT corretto. Una omografia
automatica singola e' diagnostica; per l'uso operativo serve calibrazione
per-frame affidabile e astensione sui frame senza supporto.

## Esecuzione remota del pilot

Da `/home/cappetti/FT`:

```bash
mkdir -p evaluation/detection_tracking_manifests \
  artifacts/detection_tracking output_videos/detection_tracking \
  evaluation_outputs/detection_tracking logs

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/cappetti/FT \
python3 scripts/build_gsr_detection_tracking_manifest.py \
  --gsr-dir /media/data-lie/cappetti/dataset/SoccerNet-GSR \
  --split valid \
  --count 12 \
  --seed 20260713 \
  --output evaluation/detection_tracking_manifests/valid_pilot12_v1.json
```

Avvio completo in background:

```bash
nohup env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/cappetti/FT \
python3 scripts/run_gsr_detection_tracking_benchmark.py \
  --manifest evaluation/detection_tracking_manifests/valid_pilot12_v1.json \
  --config configs/gsr_detection_tracking_benchmark_v1.yaml \
  --model-path /home/cappetti/FT/best_yolo26x_gsr_light.pt \
  --run-name gsr_valid_pilot12_baseline_v1 \
  --resume \
  > logs/gsr_valid_pilot12_baseline_v1.log 2>&1 &

echo $! > logs/gsr_valid_pilot12_baseline_v1.pid
```

Monitoraggio:

```bash
tail -f logs/gsr_valid_pilot12_baseline_v1.log
```

Aggregazione dopo `DETECTION/TRACKING BENCHMARK COMPLETE`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/cappetti/FT \
python3 scripts/aggregate_gsr_detection_tracking_benchmark.py \
  --manifest evaluation/detection_tracking_manifests/valid_pilot12_v1.json \
  --evaluation-root evaluation_outputs/detection_tracking/gsr_valid_pilot12_baseline_v1 \
  --output-dir evaluation_outputs/detection_tracking/gsr_valid_pilot12_baseline_v1_aggregate
```

## Smoke test prima del pilot

Usare un manifest esplicito di una sequenza e limitare sia inferenza sia
valutazione agli stessi 20 frame:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/cappetti/FT \
python3 scripts/build_gsr_detection_tracking_manifest.py \
  --gsr-dir /media/data-lie/cappetti/dataset/SoccerNet-GSR \
  --split valid \
  --sequences SNGS-023 \
  --output evaluation/detection_tracking_manifests/smoke_sngs023.json

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/cappetti/FT \
python3 scripts/run_gsr_detection_tracking_benchmark.py \
  --manifest evaluation/detection_tracking_manifests/smoke_sngs023.json \
  --config configs/gsr_detection_tracking_benchmark_v1.yaml \
  --model-path /home/cappetti/FT/best_yolo26x_gsr_light.pt \
  --run-name gsr_detection_tracking_smoke_20f \
  --max-frames 20
```

## Confronto con una candidata

La candidata deve usare lo stesso manifest. Dopo aver aggregato entrambi i run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/cappetti/FT \
python3 scripts/compare_gsr_detection_tracking_benchmarks.py \
  --baseline-dir evaluation_outputs/detection_tracking/gsr_valid_pilot12_baseline_v1_aggregate \
  --candidate-dir evaluation_outputs/detection_tracking/gsr_valid_pilot12_candidate_v1_aggregate \
  --output-dir evaluation_outputs/detection_tracking/comparisons/baseline_vs_candidate_v1
```

Nel confronto, `improvement > 0` significa sempre miglioramento: per ID switch e
frammentazioni il segno viene invertito automaticamente perché valori più bassi
sono migliori.
