# Power Spectral Analysis; Optical Variability of Southern TESS Blazars

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11111111.svg)](https://doi.org/10.3847/1538-4357/ad4f87) 
[![arXiv](https://img.shields.io/badge/arXiv-2406.10346-B31B1B.svg)](https://arxiv.org/abs/2406.10346)
[![License: ISC](https://img.shields.io/badge/License-ISC-blue.svg)](https://opensource.org/licenses/ISC)

This repository contains the complete Power Spectral Density (PSD) analysis pipeline and all figures from the paper **["Optical Variability Properties of Southern TESS Blazars"](https://doi.org/10.3847/1538-4357/ad4f87)** by Dingler & Smith (2024).

We analyze high-cadence TESS light curves of 67 blazars to characterize their variability. This repository provides a reproducible, figure-by-figure breakdown of the PSD analysis and best‑fit determination process.

---

## 🚀 Key Findings

- **Significant Variability:** 15 BL Lacertae objects (BLLs) and 18 Flat Spectrum Radio Quasars (FSRQs) exhibit statistically‑significant variability.  
- **PSD Slopes:** Both subpopulations show power‑law slopes of $\alpha \sim 2$, consistent with common variability mechanisms.  
- **Timescales:** Characteristic variability timescales range from ~0.8–8 days, supporting the interpretation that the optical emission originates in the jet.

---

## 📂 Repository Structure

The repository is organised by individual blazar objects. Each directory contains the **chi‑squared distributions**, **final PSD figures**, and **simulated power spectra** for that source.

```text
.
├── 1ES2322-409/                     # Example blazar directory
│   ├── chi_squared_distributions/   # Goodness-of-fit histograms & contours
│   ├── final_PowerSpectra_figures/  # Best‑fit PSD models & residuals
│   └── simulated_power_spectra_figures/  # Simulated vs observed periodograms
├── 1RXSJ054357.3-553206/
├── 1RXSJ120417.0-070959/
├── 2MASSJ02271658+0202005/
├── 2MASXJ04390223+0520443/
├── ... (directories for all 67 blazars)
├── sim_ps_analysis.ipynb            # Main Jupyter Notebook: PSD simulation & fitting
├── sim_ps_analysis.py               # Standalone Python version of the notebook
└── LICENSE                          # ISC License
```

> **Note:** The notebook `sim_ps_analysis.ipynb` contains the full PSD simulation and fitting pipeline. It can be exported to `.py` or run interactively.

---

## ⚙️ Analysis Methods

The pipeline in `sim_ps_analysis.ipynb` implements:

1. **Power Spectral Density (PSD) Estimation:**  
   - Periodogram calculation via Lomb–Scargle (for unevenly sampled data) or FFT (for evenly sampled data).  
2. **Model Fitting:**  
   - Fits three PSD models to the periodogram:  
     *Simple power‑law*  
     *Broken power‑law*  
     *Bending power‑law*  
   - Uses a maximum‑likelihood estimator in logarithmic space.  
3. **Goodness‑of‑fit:**  
   - Monte Carlo simulations of the light curves (using the Emmanoulopoulos et al. 2013 method) to generate model periodograms.  
   - Chi‑squared distributions are computed to determine the best‑fitting model and its significance.  
4. **Characteristic Timescales:**  
   - For models with a bend/break, the corresponding timescale is extracted ( $T_{\text{break}} = 1 / \nu_{\text{break}}$ ).

---

## 🔧 Requirements

- Python 3.8+ (recommended: 3.10)
- Packages:  
  ```bash
  numpy scipy pandas matplotlib astropy seaborn jupyter
  ```

Install all dependencies with:
```bash
pip install numpy scipy pandas matplotlib astropy seaborn jupyter
```

---

## 💻 Usage

### 1. Reproduce the Analysis

Launch the Jupyter Notebook:
```bash
jupyter notebook sim_ps_analysis.ipynb
```

The notebook is self‑contained: it reads the TESS light curves (CSV files) from the respective blazar directories, computes periodograms, fits the PSD models, runs Monte Carlo simulations, and produces the figures.

### 2. Run the Pipeline for a Single Object

The analysis can be re‑run for any object by modifying the `target_name` variable at the top of the notebook (e.g., `"1ES2322-409"`). The code will automatically locate the corresponding subdirectory.

### 3. Use the Python Script

For non‑interactive use (e.g., batch processing), convert the notebook to a script:
```bash
jupyter nbconvert --to script sim_ps_analysis.ipynb
```
Then run `sim_ps_analysis.py`.

---

## 📖 Citation

If you use this code or the data in your research, please cite both the original paper and this software repository.

**Paper (Dingler & Smith 2024):**
```bibtex
@article{Dingler2024,
    author = {Dingler, Ryne and Smith, Krista Lynne},
    title = {Optical Variability Properties of Southern TESS Blazars},
    journal = {The Astrophysical Journal},
    volume = {973},
    number = {1},
    pages = {44},
    year = {2024},
    doi = {10.3847/1538-4357/ad4f87},
    url = {https://doi.org/10.3847/1538-4357/ad4f87}
}
```

**Software (this repository):**
```bibtex
@misc{Dingler2024code,
    author = {Dingler, Ryne},
    title = {PowerSpectralAnalysis\_TESSBlazars: PSD Analysis of TESS Blazar Light Curves},
    year = {2024},
    doi = {10.5281/zenodo.11111111},
    url = {https://github.com/RyneD314/PowerSpectralAnalysis_TESSBlazars}
}
```

The arXiv preprint is available at [2406.10346](https://arxiv.org/abs/2406.10346).

---

## 📜 License

This project is distributed under the **ISC License**. See the [LICENSE](LICENSE) file for details.

---

## 📬 Contact & Feedback

- **Author:** [Ryne Dingler](https://github.com/RyneD314)  
- **Questions or issues?** Please open an [issue](https://github.com/RyneD314/PowerSpectralAnalysis_TESSBlazars/issues) on GitHub.

```

This `README.md` is structured to make the repository clear and actionable for researchers who want to inspect, reproduce, or build upon the PSD analysis presented in your paper. It includes a DOI badge placeholder; you can replace the example DOI with the actual DOI of the software release (if you have one) or remove that badge.
