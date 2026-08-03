# TASKS: suddivisione lavoro per 4 persone

Questo file descrive le aree chiave del progetto e le parti da sistemare, in modo che un gruppo di quattro persone possa lavorare in parallelo.

## 1. Esportazione e preparazione dataset AI4MARS

### Obiettivi
- Garantire che `segformer_lidar_fusion/scripts/prepare_ai4mars_hf.py` esporti correttamente immagini, maschere e LiDAR in formato locale.
- Verificare remapping 7→3 label.
- Assicurarsi che `--generate-dummy-lidar` produca `.npy` validi.

### Task
- [ ] Controllare e documentare i formati delle colonne del dataset HF.
- [ ] Aggiungere test sul parsing delle colonne immagine/maschera.
- [ ] Gestire eventuali nuovi split in AI4MARS e fallback in caso di colonne diverse.
- [ ] Validare che `train`, `val` e (opzionale) `test` vengano creati correttamente.

### Assegnabile a
- Persona A: dataset export + remapping

## 2. Modello e inferenza

### Obiettivi
- Documentare e sistemare il modello in `segformer_lidar_fusion/models`.
- Assicurare inferenza robusta con checkpoint mismatched e remapping 3-class.

### Task
- [ ] Verificare la classe `SegFormerLiDAR` e i parametri `num_classes`.
- [ ] Controllare `segformer_lidar_fusion/scripts/infer.py` per compatibilità con checkpoint 3/7 classi.
- [ ] Migliorare i commenti sulle normalizzazioni immagine/LiDAR e sui comportamenti `torch.device`.
- [ ] Aggiungere test di inferenza su una immagine sintetica.

### Assegnabile a
- Persona B: modello + inferenza

## 3. Training e valutazione

### Obiettivi
- Rendere il training più stabile e facile da estendere.
- Documentare i parametri sul YAML e le metriche.

### Task
- [ ] Controllare `segformer_lidar_fusion/scripts/train.py` e i config YAML.
- [ ] Valutare la possibilità di aggiungere scheduler/optimizer alternativi.
- [ ] Verificare la gestione di `class_weights`, `ignore_index` e `mixed_precision`.
- [ ] Aggiornare i log e i checkpoint periodici.
- [ ] Aggiungere test di fine-tuning rapidi con dataset minimo.

### Assegnabile a
- Persona C: training + config

## 4. Diagnosi, visualizzazione e documentazione

### Obiettivi
- Documentare l’intero workflow e assicurare che i risultati diagnostici siano corretti.
- Rendere il README e il nuovo `TASKS.md` utili per il team.

### Task
- [ ] Controllare `segformer_lidar_fusion/scripts/diagnose.py` per correttezza delle immagini create.
- [ ] Aggiungere spiegazioni dettagliate nel README su ciascun modulo e su come eseguire il progetto.
- [ ] Documentare il layout atteso del dataset e i passaggi principali.
- [ ] Creare un file `TASKS.md` con le parti da sistemare (questo file).

### Assegnabile a
- Persona D: diagnosi + documentazione

## 5. Migliorie generali e refactoring

### Obiettivi
- Migliorare la manutenibilità del progetto.
- Individuare punti di fragilità e bug noti.

### Task
- [ ] Uniformare l'uso del percorso `data/ai4mars_*` e dei nomi dei file.
- [ ] Verificare se il codice salva correttamente i checkpoint in `checkpoints/`.
- [ ] Standardizzare la gestione dei config YAML.
- [ ] Scrivere piccoli test o script di validazione.

### Assegnabile a
- Persona A/B/C/D su base rotazionale

## Problemi noti / Cose da sistemare

1. Remapping etichette 7→3 può ancora nascondere errori se la maschera contiene valori non mappati.
2. `prepare_ai4mars_hf.py` usa `--generate-dummy-lidar`; se si desidera LiDAR reale, serve un modulo di conversione aggiuntivo.
3. `infer.py` gestisce ancora checkpoint con `num_classes` diversi in modo non completamente esplicito.
4. La documentazione iniziale era sbilanciata verso Docker; ora va pulita e organizzata.
5. Manca un test automatico di integrazione per pipeline export→train→eval.


