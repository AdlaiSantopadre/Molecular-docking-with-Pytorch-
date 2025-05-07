# Sviluppo di un VAE per SELFIES

## Obiettivo del progetto
Sviluppare un *Variational Autoencoder* (VAE) in PyTorch per rappresentazioni molecolari SELFIES, con lo scopo di esplorare e generare nuovi composti tramite uno spazio latente continuo. L’architettura evolve da un RNN usato per il solo ligand, ed è progettata per essere successivamente condizionata su proprietà target (es. docking score).

---

## Architettura generale

### 1. **Encoder**
- Input: sequenze SELFIES numerizzate.
- RNN bidirezionale (`nn.LSTM`) che produce una rappresentazione compatta.
- Output: due vettori (`z_mean`, `z_logvar`) che definiscono la distribuzione normale multivariata latente.
- Output latente:  
  `z = z_mean + ε * exp(0.5 * z_logvar)`

### 2. **Decoder (evoluto rispetto al RNN originario)**
#### Problema riscontrato: *collasso dello spazio latente*
- Durante il training iniziale il decoder RNN tendeva a ignorare il vettore latente `z`, basandosi unicamente sull'informazione proveniente da teacher forcing.

#### Soluzione implementata:
- Il decoder è stato progettato per **ricevere esplicitamente `z` concatenato a ogni input token** del decoder RNN:
  ```python
  input_token = embedding[token]  
  decoder_input = torch.cat([input_token, z_expanded], dim=-1)
  ```
- Inoltre, `z` viene **usato per inizializzare lo stato nascosto del decoder LSTM** tramite una proiezione lineare:
  ```python
  h0 = self.z_to_h(z).unsqueeze(0)
  c0 = self.z_to_c(z).unsqueeze(0)
  ```
- Queste strategie forzano il decoder a **dipendere realmente da `z`**, evitando il collasso.

---

## Funzione di perdita

### 1. **Reconstruction Loss**
- CrossEntropy tra la sequenza target e quella generata.

### 2. **KL Divergence**
- Calcolata tra la distribuzione latente e una gaussiana standard:
  ```python
  kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
  ```
- Strategia di annealing: incremento graduale del peso β per bilanciare le due componenti:
  `loss = recon_loss + β(t) * kl_loss`

---

## Dettagli implementativi

- **SELFIES**: Tokenizzazione robusta per molecole.
- **Padding dinamico**: con `collate_fn` nei `DataLoader`.
- **Sampling**: Reparametrization trick.
- **Teacher forcing**: utilizzato solo durante il training.
- **Decoder autoregressivo**: RNN che produce token per token.

---

## Stato attuale

- VAE funzionante su dataset molecolare SELFIES.
- Il decoder utilizza `z` in input + stato iniziale, evitando il collasso.
- Embedding latente pronto per future analisi (e.g. t-SNE, clustering).
- Loss convergente e generazione plausibile nei primi esperimenti.

---

## Prossimi sviluppi

- Esplorazione dello spazio latente: interpolazioni, vettorializzazione.
- Condizionamento sul *docking score* per trasformare il modello in un CVAE.
- Aggiunta di rumore o dropout per rendere il decoder più robusto.
- Fase 2: ottimizzazione molecolare tramite gradiente nello spazio latente.
