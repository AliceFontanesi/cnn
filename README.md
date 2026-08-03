# SegFormer-B0 + LiDAR Fusion — ERC Terrain Segmentation

Questo repository è un prototipo di pipeline per la segmentazione semantica del terreno marziano usando un modello SegFormer che fonde immagini RGB e dati LiDAR.

Il focus attuale è sull’uso del dataset AI4MARS e sulla riduzione della tassonomia originale a 3 classi operative:
- `background`
- `traversable_soil`
- `bedrock`

> Nota: il dataset AI4MARS non è incluso nel repository. Usa lo script `segformer_lidar_fusion/scripts/prepare_ai4mars_hf.py` per esportarlo localmente.

---

## 🧠 Panoramica dell’algoritmo

Il modello combina due rami di feature extraction:
- un backbone `MiT-B0` per l’elaborazione delle immagini RGB,
- un encoder convoluzionale per l’elaborazione delle mappe LiDAR height/intensity.

Le feature vengono quindi fuse con un modulo di attenzione multi-scala e passate a una testa decoder in stile SegFormer.

### Schema del modello

```
RGB Image -> [MiT-B0 Encoder] ---┐
                                 +--> [Multi-scale Fusion] --> [SegFormer Head] --> Segmentation Map
LiDAR H/I -> [LiDAR Conv Encoder] -┘
```

### Flusso dei dati

```
1) Input:
   - immagine RGB (3 canali)
   - mappa LiDAR (2 canali: height, intensity)

2) Estrazione feature:
   - ramo immagini -> MiT-B0
   - ramo LiDAR -> encoder conv

3) Fusione:
   - feature multi-scala unite con attenzione

4) Decodifica:
   - SegFormer Head genera logits per pixel
   - logits upsampled a risoluzione immagine

5) Output:
   - mappa semantica 3-classi (background, traversable_soil, bedrock)
```

### Componenti principali

1. `MiT-B0 Encoder`
   - Estrae rappresentazioni visive multi-scala da immagini RGB.
   - È leggero e adatto a scenari in cui si vuole inferenza rapida.

2. `LiDAR Conv Encoder`
   - Riceve input LiDAR con due canali: height e intensity.
   - Codifica informazioni geometriche e topografiche che non sono facilmente visibili nella sola immagine.

3. `Multi-scale Fusion`
   - Allinea e combina le feature della camera e del LiDAR a più risoluzioni.
   - Questo aiuta il modello a disambiguare superfici simili visivamente ma differenti nella pendenza o nell’ostruzione.

4. `SegFormer Head`
   - Aggrega le feature fuse e produce logits per pixel.
   - I logits vengono interpolati alla risoluzione originale dell’immagine.

### Vantaggi del design

- L’integrazione camera + LiDAR migliora la robustezza su terreno complesso.
- Il decoder SegFormer è adatto per segmentazioni dense e multi-classe.
- Il modello supporta un flusso end-to-end: immagine + LiDAR → mappa semantica.

---

## 🎯 Significato delle 3 classi

La tassonomia ridotta è pensata per scenari di navigazione e per valutare il terreno con priorità operativa.

| ID | Classe | Cosa rappresenta |
|---:|---|---|
| 0 | `background` | Zona non utile per navigazione: sky, ombre, strutture non terreno o aree non classificabili. |
| 1 | `traversable_soil` | Terreno potenzialmente percorribile e sicuro per un rover: suolo compatto, sabbia aggredibile, pavimentazioni morbide. |
| 2 | `bedrock` | Superfici rocciose o massicce che potrebbero ostacolare o danneggiare un rover. |

### Mapping della tassonomia AI4MARS → 3 classi

Per ridurre le classi originali, lo script di preparazione applica un remapping come questo:
- `0 -> 0` background
- `1 -> 1` traversable_soil
- `2 -> 2` bedrock
- `3 -> 2` bedrock
- `4 -> 2` bedrock
- `5 -> 1` traversable_soil
- `6 -> 0` background

Questo significa che classi originali come piccoli massi o pendenze vengono ricondotte alle categorie operative.

---

## 📁 Struttura del progetto

### `segformer_lidar_fusion/models`

- `mit.py`
  - Implementa il backbone MiT-B0 usato come encoder visivo.
- `lidar_encoder.py`
  - Encoder convoluzionale per le mappe LiDAR 2-canale.
- `fusion.py`
  - Modulo di fusione multi-scala tra feature visive e LiDAR.
- `segformer_head.py`
  - Head decoder in stile SegFormer che genera logits pixel-wise.
- `segformer_lidar.py`
  - Classe principale che unisce camera, LiDAR, fusione e testa di output.

### `segformer_lidar_fusion/data`

- `erc_dataset.py`
  - Implementa `torch.utils.data.Dataset` per coppie immagine / LiDAR / maschera.
  - Legge immagini RGB (`.png`, `.jpg`), maschere semantiche (`.png`) e LiDAR (`.npy`).
  - Esegue resize uniforme, normalizzazione e augmentazioni base (flip orizzontale/verticale).
  - Tratta `255` come `ignore_index` durante il training.

### `segformer_lidar_fusion/scripts`

- `prepare_ai4mars_hf.py`
  - Esporta AI4MARS da HuggingFace in una struttura locale compatibile con `ERCDataset`.
  - Supporta il remapping delle etichette, il detection automatico delle colonne immagine/maschera e la generazione di LiDAR dummy.
- `train.py`
  - Script di training che carica config YAML, costruisce dataset e modello, e allena con checkpointing.
  - Usa `AdamW`, learning rate poly decay, warmup e mixed precision quando disponibile.
- `evaluate.py`
  - Valuta un checkpoint su validation/test e stampa IoU per classe e mIoU.
- `infer.py`
  - Esegue inferenza su una singola immagine + LiDAR e salva una visualizzazione con overlay e legenda.
- `diagnose.py`
  - Calcola mIoU per esempio di validazione e salva i peggiori `topk` come immagini side-by-side.

### `segformer_lidar_fusion/utils`

- `visualization.py`
  - Converte maschere di classe in colori e crea overlay visivi.
- `metrics.py`
  - Calcola IoU per classe e mean IoU.

### `segformer_lidar_fusion/configs`

- `erc_config_3class.yaml`
  - Configurazione principale per il training a 3 classi con batch size, optimizer, scheduler e class weights.
- I file YAML definiscono anche nome e colore delle classi.

---

## 🛠️ Workflow consigliato

### Quick Start

```bash
# 1) prepara i dati AI4MARS in locale
python3 segformer_lidar_fusion/scripts/prepare_ai4mars_hf.py \
  --out data/ai4mars_partial \
  --max-examples 100 \
  --generate-dummy-lidar \
  --mapping "0:0,1:1,2:2,3:2,4:2,5:1,6:0"

# 2) addestra il modello 3-class
python3 -m segformer_lidar_fusion.scripts.train --config segformer_lidar_fusion/configs/erc_config_3class.yaml

# 3) valuta il checkpoint migliore
python3 -m segformer_lidar_fusion.scripts.evaluate \
  --config segformer_lidar_fusion/configs/erc_config_3class.yaml \
  --checkpoint checkpoints/erc_3class/best.pth \
  --split val

# 4) inferisci e salva un overlay
python3 segformer_lidar_fusion/scripts/infer.py \
  --checkpoint checkpoints/erc_3class/best.pth \
  --image data/ai4mars_partial/val/images/frame_00003.png \
  --lidar data/ai4mars_partial/val/lidar/frame_00003.npy \
  --output out.png \
  --config segformer_lidar_fusion/configs/erc_config_3class.yaml \
  --num-classes 3
```

### Schema del workflow

```
AI4MARS HF -> prepare_ai4mars_hf.py -> data/local_dataset/
           -> train.py -> checkpoints/best.pth
           -> evaluate.py -> mIoU / per-class IoU
           -> diagnose.py -> worst-k visualizations
           -> infer.py -> result overlay.png
```

### 1) Preparare il dataset AI4MARS

```bash
python3 segformer_lidar_fusion/scripts/prepare_ai4mars_hf.py \
  --out data/ai4mars_partial \
  --max-examples 100 \
  --generate-dummy-lidar \
  --mapping "0:0,1:1,2:2,3:2,4:2,5:1,6:0"
```

- `--out` crea la struttura dati locale.
- `--max-examples` è utile per prototipi rapidi.
- `--generate-dummy-lidar` produce file LiDAR placeholder quando non sono disponibili dati reali.
- `--mapping` converte le etichette originali in 3 classi.

Per esportare l’intero dataset senza limite:

```bash
python3 segformer_lidar_fusion/scripts/prepare_ai4mars_hf.py \
  --out data/ai4mars_hf_full \
  --generate-dummy-lidar \
  --mapping "0:0,1:1,2:2,3:2,4:2,5:1,6:0"
```

### 2) Addestramento

```bash
python3 -m segformer_lidar_fusion.scripts.train --config segformer_lidar_fusion/configs/erc_config_3class.yaml
```

Il training utilizza:
- `AdamW`
- schedulazione polinomiale del learning rate
- warmup nei primissimi step
- mixed precision se disponibile
- checkpoint `best.pth` e salvataggi periodici ogni 10 epoche

### 3) Valutazione

```bash
python3 -m segformer_lidar_fusion.scripts.evaluate \
  --config segformer_lidar_fusion/configs/erc_config_3class.yaml \
  --checkpoint checkpoints/erc_3class/best.pth \
  --split val
```

### 4) Diagnostica dei casi peggiori

```bash
python3 segformer_lidar_fusion/scripts/diagnose.py \
  --config segformer_lidar_fusion/configs/erc_config_3class.yaml \
  --checkpoint checkpoints/erc_3class/best.pth \
  --output-dir data/erc_3class/diagnosis \
  --topk 5
```

Salva immagini comparando:
- immagine originale,
- predizione overlay,
- ground truth overlay.

### 5) Inferenza singola immagine

```bash
python3 segformer_lidar_fusion/scripts/infer.py \
  --checkpoint checkpoints/erc_3class/best.pth \
  --image data/ai4mars_partial/val/images/frame_00003.png \
  --lidar data/ai4mars_partial/val/lidar/frame_00003.npy \
  --output out.png \
  --config segformer_lidar_fusion/configs/erc_config_3class.yaml \
  --num-classes 3
```

`infer.py` applica normalizzazione, esegue la rete e salva un overlay RGB con legenda.

---

## 📦 Struttura dei dati richiesta

```
data/<split>/
├── images/    # RGB frames (*.png, *.jpg)
├── lidar/     # .npy files shape [2,H,W]
└── masks/     # .png uint8 class ids
```

- I file devono avere lo stesso stem tra `images/`, `lidar/` e `masks/`.
- Il dataset configurato usa `train`, `val` e `test`.
- Il valore `255` è riservato a `ignore_index` e non viene valutato.

---

## 🔧 Dettagli tecnici utili

- Il dataset usa `torch.nn.functional.interpolate` per resize immagini, LiDAR e maschere.
- Le maschere vengono ridimensionate con interpolazione `nearest` per preservare gli id di classe.
- Le immagini RGB sono normalizzate con mean/std standard ImageNet.
- Il LiDAR è normalizzato con media 0 e deviazione 1 di default.
- Il modello attuale aspetta input LiDAR con shape `(2, H, W)`:
  - canale 0 = height,
  - canale 1 = intensity.

---

## 📌 Limiti e note

- La pipeline attuale funziona come prototipo; la parte LiDAR reale va verificata su dati reali.
- `--generate-dummy-lidar` è utile per debugging ma non sostituisce dati reali di profondità.
- Il remapping 7→3 è applicato nello script di preparazione, quindi il training e l’inferenza usano solamente la tassonomia ridotta.
- È consigliato valutare l’equilibrio delle classi e aggiornare `training.class_weights` se `bedrock` o `background` risultano troppo sbilanciati.

---

## ✅ Cosa è già presente nel repository

- Export AI4MARS a formato locale compatibile con PyTorch.
- Training 3-class con SegFormer+B0 + fusione LiDAR.
- Inferenza visuale con overlay e legenda.
- Valutazione IoU e diagnostica dei casi peggiori.

---

## 📝 License

MIT
