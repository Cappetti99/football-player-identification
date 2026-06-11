# Handoff FT - Stato al 2026-06-09

## Contesto

I dati e i modelli pesanti sono sulla macchina remota `/home/cappetti/FT`; in locale abbiamo modificato il codice/config del progetto in `FT/`.

Obiettivo del lavoro: migliorare tracking/identity/ball handling senza perdere la baseline stabile. Le run remote sono state fatte soprattutto su:

- `Int-Ata`
- `Inter-Atalanta`
- `Inter-Juve`

## Baseline da preservare

La linea stabile per identity/tracking persone e OCR resta:

```yaml
configs/default_realvideo_ocr_nopromote_rawgate020_scenecuts_loose_noreset_gkfallback_spreadgate_propagation_linker_legacyappearance.yaml
```

Questa usa:

- ByteTrack baseline invariato.
- Palla temporale disattivata di default.
- Linker nuovo con ordinamento temporale, ma appearance legacy HSV.
- Identity propagation strict.

Su `Int-Ata` 1200/full, `legacyappearance` e baseline strict sono identiche:

- `rows`: `16423`
- `assigned_rows`: `6197`
- `candidate_rows`: `2096`
- `jersey_rows`: `6261`
- `propagated_rows`: `336`
- duplicati: `0`

Conclusione: il fix del linker temporale e' safe se resta in modalita' legacy appearance.

## Modifiche implementate

### Linker

File:

- `ft/linking/tracklet_linker.py`
- `ft/pipeline.py`
- `ft/config.py`
- `configs/default.yaml`
- `ft/validation.py`
- `tests/test_identity.py`

Cambi:

- Candidate iteration ora privilegia tracklet temporalmente vicini.
- Aggiunto `max_temporal_candidates`.
- Aggiunto `embedding_mode`.
- Aggiunto `appearance_min_similarity_hsv`.
- Se `embedding_mode=hsv`, soglia effettiva puo' essere rilassata.
- Config `legacyappearance` mantiene soglia HSV a `0.72`.

Config utili:

```yaml
configs/default_realvideo_ocr_nopromote_rawgate020_scenecuts_loose_noreset_gkfallback_spreadgate_propagation_linker_legacyappearance.yaml
configs/default_realvideo_ocr_nopromote_rawgate020_scenecuts_loose_noreset_gkfallback_spreadgate_propagation_linker_legacyappearance_full.yaml
```

### StrongSORT

File:

- `ft/pipeline.py`
- `ft/validation.py`
- `ft/config.py`
- `configs/default.yaml`
- `configs/strongsort.yaml`
- `configs/strongsort_tracking_debug.yaml`
- `tests/test_validation.py`

Cambi:

- StrongSORT resta disponibile ma richiede opt-in:

```yaml
tracking:
  backend: strongsort
  strongsort:
    allow_experimental: true
```

Motivo: la versione in codebase usa feature colore leggere, non un vero ReID model. Non va confrontata con ByteTrack come tracker equivalente.

### OCR

File:

- `ft/validation.py`
- `tests/test_validation.py`

Cambi:

- Aggiunto warning se si usa multi-backend OCR con `aggregate_by_crop=false`.
- Il codice attuale con `aggregate_by_crop=true` non gonfia i voti MMOCR+EasyOCR come osservazioni indipendenti dello stesso crop.

### OSNet ReID

File:

- `ft/features/reid_extractor.py`
- `ft/features/visual.py`
- `ft/validation.py`
- `tests/test_identity.py`

Cambi:

- Aggiunto `OSNetReIDExtractor` opzionale con `torchreid`.
- Nuovo `visual.embedding_mode: osnet`.
- Se `torchreid` manca, fallback automatico a HSV.

Config:

```yaml
configs/default_realvideo_ocr_nopromote_rawgate020_scenecuts_loose_noreset_gkfallback_spreadgate_propagation_linker_osnet.yaml
configs/default_realvideo_ocr_nopromote_rawgate020_scenecuts_loose_noreset_gkfallback_spreadgate_propagation_linker_osnet_widedistance.yaml
```

Risultati:

- Su `Int-Ata`, OSNet e legacy danno risultati finali identici.
- Su `Inter-Juve` 1200f, OSNet e legacy danno risultati finali identici.
- OSNet era attivo davvero:
  - `reid_status: ok`
  - `computed: 21611` su Inter-Juve
- Il collo di bottiglia non era appearance ma `distance` nel linker.

Conclusione: OSNet e' integrato ma non promosso. Serve solo per esperimenti con gate spaziale piu' largo.

### ByteTrack tuning

Config aggiunta:

```yaml
configs/default_realvideo_ocr_nopromote_rawgate020_scenecuts_loose_noreset_gkfallback_spreadgate_propagation_broadcast_bytetrack_tuned.yaml
```

Contenuto:

```yaml
tracking:
  track_activation_threshold: 0.25
  lost_track_buffer: 90
  minimum_matching_threshold: 0.85
  minimum_consecutive_frames: 3
```

Non promossa. In esperimenti precedenti l'aumento di `track_activation_threshold` aveva ridotto evidenza OCR/assegnazioni.

### Palla Kalman

File:

- `ft/tracking/ball_kalman.py`
- `ft/tracking/yolo_bytetrack.py`
- `ft/config.py`
- `configs/default.yaml`
- `ft/pipeline.py`
- `ft/validation.py`
- `tests/test_ball_kalman.py`
- `tests/test_ball_tracking.py`

Cambi:

- Aggiunto `BallKalmanTracker` con stato `[x, y, dx, dy, ddx, ddy]`.
- Disattivato di default.
- Se attivo:
  - predice palla durante occlusioni brevi;
  - marca frame con `kalman_predicted`;
  - usa la predizione come gate contro detection lontane;
  - aumenta dinamicamente `ball_max_area_ratio` quando velocita' stimata e' alta.

Config:

```yaml
configs/default_realvideo_ocr_nopromote_rawgate020_scenecuts_loose_noreset_gkfallback_spreadgate_propagation_ballkalman.yaml
```

Risultati su `Inter-Atalanta` 500f, confronto fair:

Legacy:

- `detected_frames`: `292`
- `interpolated_frames`: `2`
- `p95_detected_jump_px`: `556`
- `max_reacquisition_jump_px`: `1035`

BallKalman gate:

- `detected_frames`: `249`
- `interpolated_frames`: `45`
- `p95_detected_jump_px`: `21`
- `max_reacquisition_jump_px`: `82`

Identity invariata tra legacy e ballkalman:

- `rows`: `3761`
- `assigned_rows`: `2178`
- `candidate_rows`: `971`
- `propagated_rows`: `159`
- `jersey_rows`: `2178`
- duplicati: `0`

Conclusione: `ballkalman` e' una buona config sperimentale per overlay/analisi palla, ma non ancora default. Serve valutazione visiva full video.

### YOLO training augmentations

File:

- `scripts/train_yolo_gsr_full.py`

Cambi:

- Aggiunte augmentation YOLO native:
  - `degrees`
  - `perspective`
  - `mosaic`
  - `copy_paste`
  - `mixup`
- Aggiunta augmentation offline opzionale train-only:
  - motion blur direzionale;
  - bbox occlusion 20-40%.

Non va mischiato con le run pipeline correnti: e' un ramo separato per riallenare YOLO.

## Run principali gia' fatte

### Int-Ata

Baseline strict:

```text
Int-Ata_scenecuts_loose_noreset_gkfallback_spreadgate_propagation_strict_1200f
```

Legacyappearance:

```text
Int-Ata_scenecuts_loose_noreset_gkfallback_spreadgate_propagation_strict_linker_legacyappearance_1200f
Int-Ata_scenecuts_loose_noreset_gkfallback_spreadgate_propagation_strict_linker_legacyappearance_full_v2
```

Esito: identico alla baseline strict.

OSNet:

```text
Int-Ata_scenecuts_loose_noreset_gkfallback_spreadgate_propagation_strict_linker_osnet_1200f
```

Esito: OSNet attivo, risultato finale identico.

### Inter-Juve

OSNet:

```text
Inter-Juve_scenecuts_loose_noreset_gkfallback_spreadgate_propagation_strict_linker_osnet_1200f
```

Legacyappearance:

```text
Inter-Juve_scenecuts_loose_noreset_gkfallback_spreadgate_propagation_strict_linker_legacyappearance_1200f
```

Esito: identici.

### Inter-Atalanta

Legacy control:

```text
Inter-Atalanta_scenecuts_loose_noreset_gkfallback_spreadgate_propagation_legacyappearance_500f
```

BallKalman gate:

```text
Inter-Atalanta_scenecuts_loose_noreset_gkfallback_spreadgate_propagation_ballkalman_gate_500f
```

Esito: identita' invariata, palla molto migliorata.

## Prossimo passo consigliato

Lanciare full video `Inter-Atalanta` con BallKalman gate.

Attenzione: molte config ereditano `max_frames: 300` da `configs/default.yaml`. Per full serve una config con `max_frames: null`.

Se non esiste sul remoto, crea:

```bash
cat > configs/default_realvideo_ocr_nopromote_rawgate020_scenecuts_loose_noreset_gkfallback_spreadgate_propagation_ballkalman_full.yaml <<'YAML'
base_config: default_realvideo_ocr_nopromote_rawgate020_scenecuts_loose_noreset_gkfallback_spreadgate_propagation_ballkalman.yaml

max_frames: null
YAML
```

Run:

```bash
RUN=Inter-Atalanta_scenecuts_loose_noreset_gkfallback_spreadgate_propagation_ballkalman_gate_full
mkdir -p logs output_videos/costume-video artifacts/costume-video

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/cappetti/FT \
nohup python3 -m ft.cli run \
  --config configs/default_realvideo_ocr_nopromote_rawgate020_scenecuts_loose_noreset_gkfallback_spreadgate_propagation_ballkalman_full.yaml \
  --video-path costume-video/Inter-Atalanta/Inter-Atalanta.mp4 \
  --model-path best_yolo26x_gsr_light.pt \
  --output-path output_videos/costume-video/${RUN}.mp4 \
  --artifacts-dir artifacts/costume-video/${RUN} \
  --roster-path costume-video/Inter-Atalanta/Inter-Atalanta.json \
  --wandb-name ${RUN} \
  > logs/${RUN}.log 2>&1 &

tail -f logs/${RUN}.log
```

Audit palla dopo run:

```bash
python3 - <<'PY'
import json
from pathlib import Path

runs = [
    "Inter-Atalanta_scenecuts_loose_noreset_gkfallback_spreadgate_propagation_legacyappearance_500f",
    "Inter-Atalanta_scenecuts_loose_noreset_gkfallback_spreadgate_propagation_ballkalman_gate_full",
]

for run in runs:
    root = Path("artifacts/costume-video") / run / "metadata"
    print("\n==", run)
    path = root / "Inter-Atalanta_ball_tracking.json"
    print("ball_json_exists", path.exists())
    if path.exists():
        data = json.loads(path.read_text())
        for key in [
            "total_frames",
            "ball_frames",
            "detected_frames",
            "interpolated_frames",
            "mean_detection_confidence",
            "max_detected_jump_px",
            "p95_detected_jump_px",
            "max_gated_detected_jump_px",
            "p95_gated_detected_jump_px",
            "max_reacquisition_jump_px",
        ]:
            print(key, data.get(key))
        print("largest_jumps", data.get("largest_detected_jumps", [])[:8])
PY
```

## Cose da non promuovere per ora

- `hsv_lab_gradient` come linker default: su `Int-Ata` ha peggiorato `assigned_rows`.
- OSNet come default: integrato e funzionante, ma non migliora i risultati nei test fatti.
- ByteTrack tuned con `track_activation_threshold: 0.25`: rischio perdita evidenza OCR.
- BallKalman come default: promettente, ma serve validazione visiva full.

## Test locali eseguiti

Comandi rilevanti:

```bash
PYTHONPATH=. python3 tests/test_ball_tracking.py
PYTHONPATH=. python3 tests/test_ball_kalman.py
python3 -m py_compile ft/tracking/yolo_bytetrack.py ft/tracking/ball_kalman.py ft/features/visual.py ft/features/reid_extractor.py ft/pipeline.py ft/config.py ft/validation.py
```

Nota: l'ambiente locale non ha sempre `cv2`, `yaml` o `torchreid`; per run complete usare il remoto `mmocr`.
