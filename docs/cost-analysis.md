# Azure cost analysis — cv-pipeline production deployment

**Author:** Danil Sysenko
**Date:** 2026-05-26
**Status:** Estimates only — see section 1 for why a Cost Management export was
not available.

> This document was written by a collaborator on the original group project and
> is reproduced here with its authorship intact. See the repository
> [attribution](../README.md#attribution) section.

---

## 1. Why this is an estimation, not a Cost Management export

The project ran inside a university-managed Azure tenant. Team accounts had access to the ML workspace through internal sharing but held **no role assignments at subscription or resource group scope**:

```
> az role assignment list --assignee <me> --all --output table
> (empty)
```

Without a role assignment at that scope, the Azure Cost Management blade returns no data at all, so there was no billing export to work from.

**Mitigation:** an estimation methodology was agreed instead — estimate from
the job telemetry that *is* visible, plus public list prices, and state the
assumptions explicitly.

This document therefore uses **Azure public list prices** (West Europe region, pay-as-you-go) as the cost basis. The view it presents is *"what would this cost to run in a normal subscription, without educational subsidies"* — which is the figure that matters for handover.

---

## 2. Methodology

### 2.1 What we measure

Per-job and per-month cost across the four lines that show up in any ML deployment:

1. **Training compute** — Azure ML jobs that train new model versions
2. **Inference compute** — managed online endpoint serving `/infer` 24/7
3. **Storage** — image data, processed patches, model registry artifacts
4. **Network egress + ancillary services** — ACR, Application Insights, etc.

### 2.2 How we measure

For each line, the calculation is:

```
monthly_cost = unit_price (USD per X) × usage (X per month)
```

Where:

- **unit_price** comes from Azure published pricing pages (cited below; pulled 2026-05-26)
- **usage** comes from one of:
  - Real per-job duration from AML run metadata (e.g. Filipp's `green_nose_c68r6r1d4v` run JSON)
  - Reasonable extrapolation based on training/usage cadence assumptions stated next to the number

### 2.3 Sample measurement (validated)

Cloud training job `green_nose_c68r6r1d4v`, owned by Filipp Lotsmanov:

| Field | Value |
|---|---|
| Compute target | Arc-enabled Kubernetes, university-managed |
| Instance type | `gpu` (generic Arc node, no public SKU equivalent) |
| Wall-clock duration | 00:02:00 (2 minutes) |
| Container image | `mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04` |
| `jobCost.chargedGpuSeconds` | `null` (educational compute, not billed) |

Because the Arc compute is not on the Azure public price list, we map it to the closest **equivalent public SKU**: NVIDIA T4 single-GPU. Rationale: U-Net + ResNet34 segmentation (24.4 M parameters) at batch size 16 fits comfortably in 16 GB GPU memory, which is what T4 offers. V100 would be overprovisioned for this workload.

### 2.4 Equivalent SKU selection

| Workload | Mapped SKU | Reason |
|---|---|---|
| Training | `Standard_NC4as_T4_v3` (4 vCPU, 28 GB RAM, 1× T4 16 GB) | Smallest T4 SKU; fits U-Net + batch 16 with headroom |
| Inference endpoint | `Standard_NC4as_T4_v3` | Same model loaded once on warm endpoint; T4 is the standard inference SKU |
| Data preprocessing | `Standard_D4s_v3` (4 vCPU, 16 GB RAM, no GPU) | CPU-only Petri-dish extraction + patch generation |

**Note:** The earlier-generation `Standard_NC6s_v3` (V100) was retired on Sept 30, 2025 and is no longer the right reference price. We use the active replacement family.

---

## 3. Reference prices (West Europe, pay-as-you-go, USD, May 2026)

All numbers from Azure published pricing pages, captured 2026-05-26.

| Resource | SKU / tier | Unit | Price (USD) | Source |
|---|---|---|---:|---|
| GPU compute (training, inference) | NC4as_T4_v3 | per hour | **$0.526** | [Vantage](https://instances.vantage.sh/azure/vm/nc4ast4-v3) |
| GPU compute (training, spot) | NC4as_T4_v3 spot | per hour | **$0.205** | [Vantage](https://instances.vantage.sh/azure/vm/nc4ast4-v3) |
| CPU compute (data pipeline) | D4s_v3 | per hour | **$0.192** | Azure VM pricing |
| Blob Storage hot tier (LRS) | Hot, locally redundant | per GB-month | **$0.018** | [Azure Storage pricing 2026](https://www.nops.io/blog/azure-storage-pricing/) |
| Blob Storage cool tier (LRS) | Cool, locally redundant | per GB-month | **$0.010** | Same |
| Blob Storage archive (LRS) | Archive, locally redundant | per GB-month | **$0.00099** | Same |
| Container Registry | Standard | per day | **$0.667** (~$20/mo) | [ACR pricing guide](https://www.pump.co/blog/azure-container-registry-pricing/) |
| Egress (outbound to internet) | Standard | per GB | **~$0.087** | Azure bandwidth pricing |

---

## 4. Monthly cost estimate

### 4.1 Usage assumptions

Stated up front so the numbers can be re-run if assumptions change:

- **Training:** 1 full training run per week (~10 min on T4 — extrapolating from `green_nose` 2-min/5-epoch run to a 50-epoch baseline) + 1 hyperparameter sweep run per month (10 configs × 10 min each = ~1.7 h)
- **Inference endpoint:** 24/7 single-instance T4 deployment (the way `ManagedOnlineEndpoint` is provisioned by default)
- **Data pipeline:** 1 weekly preprocessing run, ~30 min CPU
- **Storage:**
  - Raw images: 100 GB (HADES dataset + future uploads)
  - Processed patches + intermediate data: 50 GB
  - Model registry (4 versions × ~100 MB each): 0.4 GB
- **Egress:** ~10 GB/month (Next.js frontend pulling masks, inference responses)
- **Ancillary:** ACR Standard tier (~5 GB image storage)

### 4.2 Numbers

| Line item | Calculation | Monthly cost |
|---|---|---:|
| **Training compute (regular)** | 4 runs × 0.17 h × $0.526 | $0.36 |
| **Training compute (sweep)** | 1.7 h × $0.526 | $0.89 |
| **Inference endpoint** | 720 h × $0.526 | **$378.72** |
| **Data preprocessing** | 4 runs × 0.5 h × $0.192 | $0.38 |
| **Blob storage (hot)** | 150 GB × $0.018 | $2.70 |
| **Model registry storage** | 0.4 GB × $0.018 | $0.01 |
| **Container Registry** | Standard tier | $20.00 |
| **Network egress** | 10 GB × $0.087 | $0.87 |
| **Subtotal** | | **$403.93** |
| Application Insights, misc. (~10%) | | $40.39 |
| **TOTAL** | | **~$444/month** |

### 4.3 Where the money actually goes

A pie chart drawn from the table above would have one giant slice and noise:

- Inference endpoint: **85.4%** of monthly cost
- Container Registry: 4.5%
- Storage: 0.6%
- Training + preprocessing combined: <0.4%

**Conclusion:** the always-on inference endpoint dominates everything else combined. Any cost optimisation strategy that does not address it is decorative.

---

## 5. Optimisation strategies

Five strategies, ranked by impact on the cost breakdown above.

### 5.1 Auto-shutdown for inference endpoint during off-hours

**What:** Configure the AML managed online endpoint to scale to zero instances outside business hours (e.g. 19:00–07:00 weekdays + full weekends). Use Azure Functions + AML SDK to scale up/down on a schedule.

**Savings:**
- Active hours: 12 h/day × 5 days + 0 on weekends = 60 h/week = ~260 h/month
- Inference cost: 260 × $0.526 = **$136.76**
- Saved: $378.72 − $136.76 = **$241.96/month (–55% on the line, –54% on total)**

**Trade-off:** Cold start latency the first request after a scale-up — typically 30–90 s for a managed endpoint. For NPEC researchers who hit the service during the workday this is acceptable. For 24/7 automated pipelines or external callers it is not.

**When this is wrong:** If the workload has any production traffic outside business hours (e.g. an external robot scanning plates at 2 AM), do not enable this.

### 5.2 Spot pricing for training jobs

**What:** Submit training and sweep jobs with `priority="low"` (AML's term for spot capacity). Spot VMs cost ~60% less than on-demand.

**Savings:**
- Training compute on-demand: $0.36 + $0.89 = $1.25/month
- Training compute on spot ($0.205/h): 0.17 × 4 × $0.205 + 1.7 × $0.205 = $0.49/month
- Saved: **$0.76/month (–61% on the line, but –0.2% on total)**

**Trade-off:** Spot jobs can be pre-empted with 30 s warning. AML retries pre-empted jobs automatically, but the wall-clock time becomes unpredictable. Acceptable for training (we don't care if it takes 12 min instead of 10), bad for time-critical pipelines.

**Note:** This is the right thing to do even though the absolute number is tiny — it is free, it is correct practice, and as training intensity scales up (real hyperparameter sweeps, retraining triggered by drift, etc.) the absolute savings grow with it.

### 5.3 Right-size the inference endpoint SKU

**What:** Audit whether the endpoint actually needs a T4 GPU at all. U-Net ResNet34 inference for a single 640×640 image takes ~0.8 s on CPU (per `README.md` performance notes for RTX 3060 → CPU is 20–40× slower → ~16–32 s/image). For the throughput pattern of researchers uploading plates occasionally, **CPU might be enough**.

If yes, switch to `Standard_D4s_v3` (4 vCPU, 16 GB RAM, no GPU): $0.192/h instead of $0.526/h.

**Savings:**
- Current: 720 × $0.526 = $378.72
- CPU-only: 720 × $0.192 = $138.24
- Saved: **$240.48/month (–63% on the line, –54% on total)**

**Trade-off:** Inference latency goes from ~1 s to ~20 s per image. AC #489 ("p95 < 5 s for 640×640") would be violated. So this strategy is conditional on **rewriting the acceptance criterion** based on real production access patterns. If researchers upload a plate and walk away to make coffee, 20 s is fine. If the frontend shows a spinner, it's not.

**Combination note:** strategies 5.1 and 5.3 are mutually exclusive on the same endpoint. Pick one. 5.1 keeps GPU latency for the hours it runs; 5.3 trades all GPU latency for permanent cost reduction.

### 5.4 Lifecycle storage policy: cool → archive tier for old uploads

**What:** Set a Blob lifecycle rule:
- Files newer than 90 days → Hot tier ($0.018/GB-month)
- Files 90 days – 1 year → Cool tier ($0.010/GB-month)
- Files older than 1 year → Archive tier ($0.00099/GB-month)

**Savings:**
- Storage cost is currently **$2.70/month** and would drop to roughly **$1.20/month** at steady state, assuming 1/3 hot, 1/3 cool, 1/3 archive across the year.
- Saved: **~$1.50/month**.

**Trade-off:** Archive tier has a 180-day minimum retention period and re-hydration cost ($5.50 per 10,000 reads). If a researcher needs to re-process old data, they pay both per-GB retrieval fees and a delay (rehydration takes hours).

**Why include it despite the small absolute number:** a cost analysis that *only* discusses the GPU bill misses the point. Cost-effective resource management includes storage tiering even when the line item is small, because that is the pattern that scales.

### 5.5 ACR cleanup automation (tag retention + purge)

**What:** Configure ACR purge tasks to keep only the last 10 versions of each image. Without this, every CI build adds 200–500 MB and ACR storage grows indefinitely. After ~6 months a Standard tier registry fills up (100 GB included) and pays $0.003/day per GB overage.

**Savings:**
- Direct: prevents ~$10–30/month tier overage at 12-month horizon (not visible in current numbers).
- Indirect: faster pulls during deployment because fewer untagged layers.

**Trade-off:** Lose the ability to roll back to arbitrary old versions. Mitigated by tagging "release" images explicitly and excluding them from the purge rule.

### 5.6 Quick-glance ranking

| Strategy | Saving | % of bill | Effort | Conditional on |
|---|---:|---:|---|---|
| 5.1 Endpoint auto-shutdown | $241.96 | –54% | Low | No off-hours traffic |
| 5.3 CPU-only inference | $240.48 | –54% | Medium | AC #489 relaxed |
| 5.2 Spot training | $0.76 | –0.2% | Low | None |
| 5.4 Storage lifecycle | $1.50 | –0.3% | Low | None |
| 5.5 ACR purge | ~$0–20 | varies | Low | None |

**Net recommendation:** combine 5.1 + 5.2 + 5.4 + 5.5 for a total of **~$244/month saved** without changing functional behaviour during business hours. New monthly total: **~$200**, a 55% reduction. 5.3 is a separate decision tied to a product trade-off, not a pure cost decision.

---

## 6. Confidence and caveats

- **Equivalent SKU mapping is approximate.** The real Arc Kubernetes cost is unknown to us; the T4 mapping is defensible but not precise. If the operator's subscription gives them reserved or Hybrid Benefit pricing, real cost is 20–40% lower.
- **One sample training job.** We have run metadata for one cloud job (`green_nose_c68r6r1d4v`). More samples would tighten the duration extrapolation in §4.1.
- **Inference traffic pattern is assumed, not measured.** No production traffic exists yet. The 24/7 single-instance assumption is the most conservative reading of the AML default; real production usage may differ.
- **Egress is a placeholder.** Without real usage we cannot estimate this well; we used 10 GB/month as a token figure. Real number depends on whether mask images are served from blob storage (cheap intra-region) or from the API (egress-billed).
- **No reserved-instance discount applied.** A 1-year reservation on the inference SKU drops the price ~27%. We did not factor this in because it requires a procurement decision NPEC has not made.

A second pass on this analysis after the first month of real production traffic will produce numbers we trust to ±10% instead of the current ±40%.

---

## 7. Acceptance criteria check

Against the two cost-optimisation items raised during the project:

### #1239 — "Document Azure resource usage and costs"

- [x] Real cost numbers from Azure public list prices (§3 with source citations)
- [x] Covers compute, storage, projected inference (§4)
- [x] Formatted as a table with monthly estimates (§4.2)
- [x] Methodology section explains estimation approach and instructor approval (§1–2)

### #1240 — "Identify cost optimisation strategies"

- [x] At least 2 strategies with trade-off explanation (5 provided in §5)
- [x] Specific to our Azure setup, not generic (each strategy ties to a concrete line in §4.2)

---

## 8. References

Pulled 2026-05-26:

- [Standard_NC4as_T4_v3 — Vantage](https://instances.vantage.sh/azure/vm/nc4ast4-v3)
- [Standard_NC6s_v3 retirement notice — Microsoft Learn](https://learn.microsoft.com/en-us/azure/virtual-machines/ncv3-nc6s-nc12s-nc24s-retirement)
- [Azure Blob Storage pricing 2026 — nOps](https://www.nops.io/blog/azure-storage-pricing/)
- [Cloud storage pricing comparison 2026 — Finout](https://www.finout.io/blog/cloud-storage-pricing-comparison)
- [Azure Container Registry pricing guide — Pump](https://www.pump.co/blog/azure-container-registry-pricing/)
- [ACR SKU features — Microsoft Learn](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-skus)
- Filipp's training run metadata: `green_nose_c68r6r1d4v` (job JSON shared 2026-05-20)
