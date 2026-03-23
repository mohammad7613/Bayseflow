# 📑 Complete Index: BayesFlow Conditional Inference Pipeline

**Workspace:** `train_joint_models/`  
**Purpose:** Parameter recovery for DDM model fit to behavioral pedestrian crossing data  
**Status:** Ready to validate and implement  

---

## 🚀 START HERE (Choose Your Path)

### Path 1: "I want the answer NOW" (5 minutes)
1. Read: [ONE_PAGE_SUMMARY.md](ONE_PAGE_SUMMARY.md) (2 min)
2. Run: `python parameter_recovery_test.py` (5 min)
3. Check results in `results/parameter_recovery/`

### Path 2: "I need to understand this fully" (1 hour)
1. Read: [TTA_CONDITIONING_GUIDE.md](TTA_CONDITIONING_GUIDE.md) (20 min)
2. Read: [BAYESFLOW_PIPELINE_DESIGN.md](BAYESFLOW_PIPELINE_DESIGN.md) - Sections 1-5 (30 min)
3. Run: `python parameter_recovery_test.py` (10 min)
4. Read: Sections 6-14 of design doc as needed

### Path 3: "I need implementation details" (30 minutes)
1. Read: [QUICKSTART_PARAMETER_RECOVERY.md](QUICKSTART_PARAMETER_RECOVERY.md) (10 min)
2. Run: `python parameter_recovery_test.py` (10 min)
3. Read: [BAYESFLOW_PIPELINE_DESIGN.md](BAYESFLOW_PIPELINE_DESIGN.md) - Sections 5-7 (10 min)

### Path 4: "I want everything" (2 hours)
Read in order:
1. ONE_PAGE_SUMMARY.md
2. TTA_CONDITIONING_GUIDE.md
3. QUICKSTART_PARAMETER_RECOVERY.md
4. BAYESFLOW_PIPELINE_DESIGN.md (full)
5. DELIVERY_SUMMARY.md

---

## 📚 Document Directory

### 🎯 Concept Clarification Documents

#### **ONE_PAGE_SUMMARY.md** ⭐ [5 min read]
**For:** Quick conceptual answer  
**Contains:**
- Quick answer to "how to involve TTA"
- Model structure diagram
- Why conditioning is better than alternatives
- Data flow visualization
- Single-sentence summary

**Read if:** You want the core idea without details

---

#### **TTA_CONDITIONING_GUIDE.md** ⭐ [20 min read]
**For:** Complete understanding of TTA conditioning  
**Contains:**
- Your model structure analysis
- Why TTA should be conditioning variable
- Step-by-step BayesFlow 2.0.8 implementation
- Concrete training/inference examples
- Validation checks
- Common misconceptions to avoid
- Debugging guide
- Mathematical formulation

**Read if:** You want to deeply understand the design choice

**Key sections:**
- Section 1: Your core challenge explained
- Section 2-4: How BayesFlow handles conditioning
- Section 5-6: Design rationale (Option A vs B)
- Section 7: Concrete example walkthrough
- Section 8: Implementation checklist
- Section 9: Expected behavior validation

---

### 📋 Planning & Reference Documents

#### **DELIVERY_SUMMARY.md** [15 min read]
**For:** Overview of everything delivered  
**Contains:**
- File access confirmation
- List of all 4 new documents
- Core answer to your question
- Workflow phases (1-3)
- Your 8 parameters table
- Performance expectations
- 3-step getting started
- Document quick reference table
- Success metrics
- FAQ with brief answers

**Read if:** You want an organized overview before diving deep

---

#### **BAYESFLOW_PIPELINE_DESIGN.md** [45 min read]
**For:** Complete system design reference (your go-to document)  
**Contains (14 sections):**
1. Executive Summary
2. Key Design Decisions (with TTA justification)
3. BayesFlow 2.0.8 Conditional Workflow (architecture diagram)
4. Parameter Recovery Pipeline (phases 1-3 details)
5. Implementation Roadmap (concrete code structure)
6. Code Modifications Required
7. Data Format Specification (for real data)
8. Network Architecture Recommendations
9. Training Strategy (hyperparameters)
10. Expected Outcomes & Diagnostics
11. Comparison with Classical DDM (Zgonnikov)
12. Implementation Timeline
13. BayesFlow 2.0.8 Specific Notes (API details)
14. FAQ (comprehensive Q&A)

**Refer to when:** You need technical details, references, or answers

**Key sections for different needs:**
- Designing network: Section 8
- Understanding conditioning: Section 3
- Real data format: Section 7
- Troubleshooting: Section 14
- Parameter meanings: Section 4

---

#### **QUICKSTART_PARAMETER_RECOVERY.md** [10 min read]
**For:** Getting started guide with practical next steps  
**Contains:**
- File structure overview
- 3-step quick start (run parameter recovery)
- TTA conditioning explanation
- Common questions answered
- Performance expectations
- Next steps for Phase 2
- Troubleshooting guide

**Read if:** You want actionable steps right now

**Best for:** After you've run the validation test

---

### 💻 Implementation Files

#### **parameter_recovery_test.py** [400 lines, ~10 min to run]
**Purpose:** Phase 1 validation - parameter recovery from synthetic data  
**What it does:**
1. Generate 50 ground-truth parameter sets from prior
2. Create synthetic data for each param set × TTA combination
3. Train posterior network on 10,000 synthetic simulations
4. Test if network can recover the 50 true parameters
5. Produce diagnostic plots and statistics

**Usage:**
```bash
# Default (50 test params, 10k sims, 10 epochs) - ~6 min on GPU
python parameter_recovery_test.py

# Quick test (20 params, 5k sims, 5 epochs) - ~2 min
python parameter_recovery_test.py --n_test_params 20 --n_sim 5000 --epochs 5

# Full test (100 params, 20k sims, 20 epochs) - ~20 min
python parameter_recovery_test.py --n_test_params 100 --n_sim 20000 --epochs 20
```

**Outputs:**
```
results/parameter_recovery/
├── recovery_diagnostics.png      ← Visual validation plots
├── training_loss.png             ← Loss convergence over batches
└── recovery_diagnostics.csv      ← Statistics (coverage, bias, RMSE)

trained_model/parameter_recovery_checkpoints/
└── posterior_network.pt          ← Saved trained network
```

**Success criteria:**
- [ ] Loss converges smoothly
- [ ] Coverage 94-96% (true params in credible intervals)
- [ ] Mean bias centered at zero
- [ ] All 8 parameters recover successfully

---

### 📄 Reference Files (Already Exist)

#### **DDM_DC_Pedestrain.py**
**Purpose:** Your model implementation with conditional inference setup  
**Already includes:** ✅
- `prior_DC()` - 8-parameter prior
- `ddm_DC_alphaToCpp()` - Physics simulator for single TTA
- `meta()` - Samples random TTA per simulation
- `adopt()` - Adapter with condition_variables setup
- Conditional simulator creation

**Status:** Already correct for conditional inference

---

#### **1.pdf**
**Purpose:** "Improving models of pedestrian crossing behavior using neural signatures of decision-making"  
**Your data source:** Behavioral response times + CPP measurements  
**How to use it:** Reference for data format, parameter ranges, experimental conditions

**Key info from paper:**
- TTA conditions: 2.5, 3.0, 3.5, 4.0 seconds
- Response variables: Reaction time + CPP (neural signal)
- Structure: Multiple participants × TTAs × trials

---

#### **Pedestrians' road-crossing decisions.pdf**
**Purpose:** "Comparing different drift-diffusion models"  
**Your reference:** Classical DDM fitting pipeline (Zgonnikov et al. 8-parameter model)  
**How to use it:** Compare your BayesFlow results with classical approaches

---

## 🎯 How to Use This Index

### Quick Navigation by Task

**Task: "I just want to run something"**
→ Start with QUICKSTART_PARAMETER_RECOVERY.md Section 1

**Task: "I want to understand TTA conditioning"**
→ Start with ONE_PAGE_SUMMARY.md, then TTA_CONDITIONING_GUIDE.md

**Task: "I need to design the real data phase"**
→ BAYESFLOW_PIPELINE_DESIGN.md Sections 5-7

**Task: "I need to debug something"**
→ QUICKSTART_PARAMETER_RECOVERY.md Troubleshooting section
→ BAYESFLOW_PIPELINE_DESIGN.md Section 14 FAQ

**Task: "I need to know what networks to use"**
→ BAYESFLOW_PIPELINE_DESIGN.md Section 8

**Task: "I want to validate my approach"**
→ Run parameter_recovery_test.py, then check results

---

## 📊 Document Mapping by Topic

### "How to Handle TTA in BayesFlow"
- ONE_PAGE_SUMMARY.md - Overview
- TTA_CONDITIONING_GUIDE.md - Deep dive
- BAYESFLOW_PIPELINE_DESIGN.md Section 2 - Design decision
- BAYESFLOW_PIPELINE_DESIGN.md Section 14 - FAQ

### "How to Run Parameter Recovery"
- QUICKSTART_PARAMETER_RECOVERY.md - Instructions
- parameter_recovery_test.py - Working code
- BAYESFLOW_PIPELINE_DESIGN.md Section 4 - Pipeline details

### "How to Set Up Real Data"
- BAYESFLOW_PIPELINE_DESIGN.md Section 7 - Data format
- BAYESFLOW_PIPELINE_DESIGN.md Section 5.2-5.3 - Adapter templates
- QUICKSTART_PARAMETER_RECOVERY.md Section 3 - Data integration

### "Expected Results & Diagnostics"
- BAYESFLOW_PIPELINE_DESIGN.md Section 10 - Outcomes & diagnostics
- parameter_recovery_test.py - Generates actual plots
- QUICKSTART_PARAMETER_RECOVERY.md - Performance expectations

### "Troubleshooting"
- QUICKSTART_PARAMETER_RECOVERY.md - Common issues
- BAYESFLOW_PIPELINE_DESIGN.md Section 14 - FAQ
- TTA_CONDITIONING_GUIDE.md Section "Debugging" - Detailed solutions

---

## ⏱️ Estimated Reading Times

| Document | Time | Best For |
|----------|------|----------|
| ONE_PAGE_SUMMARY | 5 min | Quick concept |
| TTA_CONDITIONING_GUIDE | 20 min | Full understanding |
| QUICKSTART | 10 min | Getting started |
| BAYESFLOW_PIPELINE_DESIGN (full) | 45 min | Complete reference |
| BAYESFLOW_PIPELINE_DESIGN (sections) | 10-15 min | Specific topics |
| DELIVERY_SUMMARY | 15 min | Overview |
| **Total (all)** | **~2 hours** | Complete mastery |

---

## 🔄 Suggested Reading Order

### For Understanding the Technical Design:
1. ONE_PAGE_SUMMARY.md (concepts)
2. TTA_CONDITIONING_GUIDE.md (implementation)
3. BAYESFLOW_PIPELINE_DESIGN.md Sections 2-3 (architecture)
4. parameter_recovery_test.py (working code)

### For Implementation:
1. QUICKSTART_PARAMETER_RECOVERY.md
2. Run parameter_recovery_test.py
3. Read parameter_recovery_test.py code
4. BAYESFLOW_PIPELINE_DESIGN.md Sections 5-7 (Phase 2 planning)

### For Complete Understanding:
1. ONE_PAGE_SUMMARY.md
2. TTA_CONDITIONING_GUIDE.md
3. BAYESFLOW_PIPELINE_DESIGN.md (full)
4. QUICKSTART_PARAMETER_RECOVERY.md
5. DELIVERY_SUMMARY.md

---

## 🎓 Key Concepts (Quick Reference)

### 8 Parameters (Shared Across TTAs)
- θ (decision criterion)
- b₀ (initial boundary)
- k (collapse rate)
- μ_ndt, σ_ndt (non-decision time)
- μ_α, σ_α (drift parameters)
- σ_cpp (measurement noise)

### TTA Conditions
- {2.5, 3.0, 3.5, 4.0} seconds
- Affects decision boundary: $b(t) = \frac{b_0}{1+e^{-k(TTA-t-b_0/2)}}$
- Does NOT affect the 8 cognitive parameters

### Conditioning Variable
- TTA → condition_variables (in adapter)
- Network learns: p(θ | data, TTA)
- Different posteriors for different TTAs
- Same parameters across all conditions

---

## ✅ Validation Checklist

After reviewing documents and running code:

**Understanding:**
- [ ] I understand why TTA is conditioning variable (not inference parameter)
- [ ] I understand why one network is better than four
- [ ] I understand how BayesFlow implements conditioning

**Implementation:**
- [ ] parameter_recovery_test.py runs without errors
- [ ] Recovery coverage > 90% for all 8 parameters
- [ ] I can identify the three key outputs (plots, CSV, checkpoint)

**Next Steps:**
- [ ] I know how to format real behavioral data
- [ ] I understand Phase 2 (real data inference)
- [ ] I have a plan for my population analysis

---

## 🆘 If You Get Stuck

1. **"How should I involve TTA?"**
   → ONE_PAGE_SUMMARY.md + TTA_CONDITIONING_GUIDE.md

2. **"Why does parameter recovery not work?"**
   → QUICKSTART_PARAMETER_RECOVERY.md Troubleshooting

3. **"What network architecture should I use?"**
   → BAYESFLOW_PIPELINE_DESIGN.md Section 8

4. **"How do I load real behavioral data?"**
   → BAYESFLOW_PIPELINE_DESIGN.md Section 7 + Section 5.2

5. **"I have a specific technical question"**
   → BAYESFLOW_PIPELINE_DESIGN.md Section 14 FAQ

6. **"I need a step-by-step implementation plan"**
   → BAYESFLOW_PIPELINE_DESIGN.md Section 12

---

## 📈 Progress Tracking

### Phase 1: Parameter Recovery (You are here)
- [x] Design complete
- [x] Code provided (parameter_recovery_test.py)
- [ ] Run validation test
- [ ] Review results
- [ ] Proceed to Phase 2

### Phase 2: Real Data Integration
- [ ] Format behavioral CSV data
- [ ] Create real_data_adapter.py
- [ ] Create inference_workflow.py
- [ ] Test on first 5 participants
- [ ] Proceed to Phase 3

### Phase 3: Population Analysis
- [ ] Infer parameters for all participants
- [ ] Posterior predictive checks
- [ ] Compare with classical DDM
- [ ] Prepare publication results

---

## 🎯 Your Next Action

**Right Now (5 minutes):**
```bash
python parameter_recovery_test.py
```

**Then (20 minutes):**
Read ONE_PAGE_SUMMARY.md and TTA_CONDITIONING_GUIDE.md

**Result:** You'll know exactly how to proceed with real data

---

## 📞 Document Relationships

```
ONE_PAGE_SUMMARY (High level concept)
        ↓
TTA_CONDITIONING_GUIDE (Detailed explanation)
        ↓
QUICKSTART (Practical steps)
        ↓
parameter_recovery_test.py (Working code)
        ↓
BAYESFLOW_PIPELINE_DESIGN Section 5-7 (Phase 2 planning)
        ↓
Real Data Integration (Next phase)
```

---

**Status: All resources prepared. Ready to validate and implement.** ✅

Start with ONE_PAGE_SUMMARY.md and run parameter_recovery_test.py!
