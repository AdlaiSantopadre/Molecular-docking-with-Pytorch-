# Sviluppo del VAE basato su SELFIES per Modellazione Molecolare

## Obiettivo del progetto
Costruire un *Variational Autoencoder* (VAE) per la rappresentazione e generazione di molecole a partire da sequenze SELFIES, con applicazioni future nell'ottimizzazione dei docking score. Il progetto evolve da una prima implementazione basata su RNN a una versione con decoder Transformer, motivata dall'esigenza di evitare il collasso dello spazio latente.

---

## Fase 1: VAE con Decoder RNN

### Architettura
- **Encoder**: `nn.LSTM` bidirezionale che produce `z_mean` e `z_logvar`.
- **Latent sampling**: `z = z_mean + ε * exp(0.5 * z_logvar)`.
- **Decoder**: LSTM autoregressivo che genera SELFIES token per token.
  - Inizializzazione dello stato LSTM con `z`.
  - Concatenazione di `z` a ogni input del decoder.

### Obiettivo: forzare il decoder a utilizzare il bottleneck latente `z`.

---

## Fase 2: Esperimenti di training con decoder RNN

### Dataset
- Molecole convertite in SELFIES, tokenizzate e numerizzate.
- Split train/validation/test e padding dinamico via `collate_fn`.

### Hiperparametri testati
- Dimensione `z`: {16, 32, 64}
- Hidden size RNN: {128, 256}
- Beta (peso KL): {0.001, 0.01, 0.1, 1.0}
- Teacher forcing ratio: fisso a 1 inizialmente

### Tecniche di regolarizzazione
- KL annealing: incremento graduale di β per evitare collasso precoce
- Dropout nel decoder

### Problema riscontrato: **collasso dello spazio latente**
- Analisi dello spazio `z` mostrava distribuzioni poco differenziate.
- Le molecole ricostruite risultavano simili tra loro, indicando che il decoder LSTM ignorava `z` durante la generazione.

---

## Fase 3: Introduzione del Decoder Transformer

### Motivazione
- Il decoder RNN, anche con iniezione esplicita di `z`, tende a usare solo l’informazione locale del teacher forcing.
- Il decoder Transformer consente una dipendenza globale dall’intera sequenza e dal vettore `z`.

### Modifiche architetturali
- Il decoder è ora un **Transformer standard (GPT-like)**:
  - Ogni token in input viene concatenato (o sommato) a una proiezione lineare del vettore latente `z`.
  - `z` può essere anche inserito come token speciale o come bias nei layer di attenzione.
- Masked self-attention per garantire generazione autoregressiva.

### Benefici osservati
- Maggiore variabilità nelle molecole generate.
- `z` influente nella generazione (visibile anche tramite interpolazioni nello spazio latente).
- Collasso drasticamente ridotto.

---

## Fase 4: Stato Attuale

- Modello VAE funzionante con encoder LSTM e decoder Transformer.
- Training stabile con β=0.1 e KL annealing fino a 1.0.
- Ricostruzione accurata e generazione plausibile.
- Spazio latente significativo: si osservano cluster molecolari distinti.

---

## Prossimi step

- Condizionamento sul docking score: passaggio a CVAE.
- Visualizzazione con t-SNE e analisi semantica dello spazio latente.
- Addestramento su dataset più ampio (es. ZINC).
- Implementazione di una loss guidata anche da proprietà molecolari (multi-task).

