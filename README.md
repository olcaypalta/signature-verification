# Offline Signature Verification — Python & MATLAB

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![MATLAB](https://img.shields.io/badge/MATLAB-R2021a%2B-0076A8)
![License](https://img.shields.io/badge/license-MIT-green)

Decides whether a scanned signature belongs to a person, given **3–5 genuine reference signatures**
of that person. No training on other writers is required.

## Method
- **Preprocessing:** Otsu binarization, crop to ink, scale into a 256×128 canvas
- **Features:** HOG (stroke direction) + uniform LBP on a 4×8 grid (stroke texture) + aspect ratio,
  ink density and centroid
- **Decision:** mean cosine distance *d* to the references; accept if (*d* − μ)/σ ≤ *k* (default 2),
  where μ, σ are the pairwise distances among the references — so the threshold adapts to how
  consistently each person signs.

## Quick start
```bash
cd python
pip install -r requirements.txt
python make_synthetic.py
python signature_verify.py --refs ../examples/genuine --query ../examples/questioned
```
```matlab
cd matlab
signature_verify      % uses ../examples/genuine and ../examples/questioned
```

## Results on the bundled synthetic examples
| Questioned | Python z | MATLAB z | Decision |
|---|---|---|---|
| q1 genuine | −0.23 | −0.27 | genuine |
| q2 genuine | +1.49 | +1.03 | genuine |
| q3 skilled forgery | +2.65 | +3.31 | forgery |
| q4 random forgery | +14.77 | +15.77 | forgery |

The skilled forgery is close to the threshold. Static (offline) features cannot capture pen
pressure or speed, so skilled forgeries are the hard case. Evaluate on CEDAR, GPDS or MCYT and
report FAR/FRR or EER before practical use.

## Türkçe
Kişinin 3–5 gerçek imzasıyla kayıt yapılır, sorgulanan imzanın aynı kişiye ait olup olmadığına karar verilir.
Eşik, kişinin kendi imzaları arasındaki farka göre otomatik ayarlanır.

**Author:** Asst. Prof. Dr. Olcay Palta, Bitlis Eren University · [ORCID 0000-0002-7496-2101](https://orcid.org/0000-0002-7496-2101) · cite via [`CITATION.cff`](CITATION.cff)
