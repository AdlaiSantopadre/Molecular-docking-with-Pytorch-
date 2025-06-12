# Metriche di Valutazione per Modelli di Regressione

Queste metriche sono usate per valutare le prestazioni del modello di regressione nel predire il docking score.

---

## 1. Coefficiente di Determinazione (R²)

Formula:

    R² = 1 - [ Σ(yᵢ - ŷᵢ)² ] / [ Σ(yᵢ - ȳ)² ]

Dove:
- yᵢ = valore reale
- ŷᵢ = valore predetto
- ȳ  = media dei valori reali

Interpretazione:
- R² = 1: predizione perfetta
- R² = 0: modello non migliore della media
- R² < 0: modello peggiore della media

---

## 2. Errore Assoluto Medio (MAE)

Formula:

    MAE = (1/n) * Σ |yᵢ - ŷᵢ|

Proprietà:
- Robusto agli outlier
- Misura l'errore medio in valore assoluto

---

## 3. Radice dell’Errore Quadratico Medio (RMSE)

Formula:

    RMSE = sqrt[ (1/n) * Σ(yᵢ - ŷᵢ)² ]

Proprietà:
- Penalizza errori grandi
- Misura l'errore medio nella stessa unità del target
