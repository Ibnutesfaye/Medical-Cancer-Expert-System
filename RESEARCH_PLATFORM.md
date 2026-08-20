# Privacy-Preserving Medical Cancer AI — Research Platform Design

> Extension of the Medical Cancer Expert System into a world-class
> research-grade Privacy-Preserving AI platform suitable for
> Master's thesis, PhD research, top AI conferences, and healthcare AI deployment.

---

## Table of Contents

1. [Platform Vision](#1-platform-vision)
2. [Three Inference Modes](#2-three-inference-modes)
3. [Concrete ML Integration](#3-concrete-ml-integration)
4. [OpenFHE Integration](#4-openfhe-integration)
5. [Encrypted Medical Workflow](#5-encrypted-medical-workflow)
6. [Research Benchmark Comparison](#6-research-benchmark-comparison)
7. [Deep Learning Model Upgrade](#7-deep-learning-model-upgrade)
8. [Explainable AI](#8-explainable-ai)
9. [Federated Learning](#9-federated-learning)
10. [Differential Privacy](#10-differential-privacy)
11. [Improved RAG Pipeline](#11-improved-rag-pipeline)
12. [Doctor Dashboard](#12-doctor-dashboard)
13. [Hospital Integration](#13-hospital-integration)
14. [Security Architecture](#14-security-architecture)
15. [Enterprise Software Architecture](#15-enterprise-software-architecture)
16. [Production Folder Structure](#16-production-folder-structure)
17. [Complete API Design](#17-complete-api-design)
18. [Extended Database Schema](#18-extended-database-schema)
19. [UI Pages Design](#19-ui-pages-design)
20. [Deployment Guide](#20-deployment-guide)
21. [Research Contributions](#21-research-contributions)
22. [Future Work](#22-future-work)

---

## 1. Platform Vision

The Medical Cancer Expert System is extended into a **Privacy-Preserving Medical AI Platform** — a system where patient data never leaves the patient's device in plaintext, yet the cloud server can still run AI inference on it.

```
Current System                     Extended Platform
──────────────                     ─────────────────
ResNet18 CNN          →            ResNet18 + EfficientNet + ViT Ensemble
Plain inference       →            3 Inference Modes (Standard / Concrete ML / OpenFHE)
Basic RAG             →            Hybrid RAG + Medical Knowledge Graph
Admin Dashboard       →            Admin + Doctor + Research + Benchmark Dashboards
No privacy            →            FHE + Federated Learning + Differential Privacy
No explainability     →            Grad-CAM + Attention Maps + Saliency Maps
Single hospital       →            Multi-hospital Federated Network
```

### Research Pillars

| Pillar | Technology | Value |
|---|---|---|
| Privacy-Preserving Inference | Concrete ML, OpenFHE | Patient data encrypted end-to-end |
| Explainable AI | Grad-CAM, Integrated Gradients | Doctors understand predictions |
| Federated Learning | Flower, PySyft | Multi-hospital training without data sharing |
| Differential Privacy | Opacus | Mathematical privacy guarantees |
| Advanced RAG | BM25 + FAISS + Reranker | More accurate medical answers |
| Clinical Integration | FHIR, HL7 | Real hospital connectivity |

---

## 2. Three Inference Modes

The platform supports three modes selectable per request. All three share the same input/output API contract.

```
POST /images/analyze
{
  "mode": "standard" | "concrete_ml" | "openfhe",
  "image_data": "<base64 or encrypted bytes>",
  "encryption_key_id": "<optional, for encrypted modes>"
}
```

### Mode 1 — Standard AI (Existing)

```
Image (plaintext)
      ↓
ResNet18 CNN (PyTorch)
      ↓
Softmax → 14-class probabilities
      ↓
Cancer/Non-cancer label + confidence
      ↓
LLM explanation (Groq)
```

- Speed: ~50ms inference
- Accuracy: 88.75% (brain), 86.36% (lung), 89.52% (skin)
- Privacy: None — image processed in plaintext on server
- Use case: Internal hospital systems, research environments

### Mode 2 — Concrete ML Private AI

```
Image (plaintext, patient device)
      ↓
ResNet18 feature extraction → 512-dim vector (plaintext, patient device)
      ↓
Quantization (8-bit integers)
      ↓
FHE Encryption of feature vector
      ↓
Encrypted feature vector → Cloud Server
      ↓
Encrypted inference (small FHE classifier)
      ↓
Encrypted prediction → Patient device
      ↓
Decryption → Cancer/Non-cancer result
```

- Speed: ~2–10 seconds (classifier only, not full ResNet)
- Privacy: Feature vector encrypted — server never sees raw image or plaintext features
- Security: 128-bit security level

### Mode 3 — OpenFHE Private AI

```
Image pixels (plaintext, patient device)
      ↓
CKKS encoding of pixel values as ciphertexts
      ↓
Homomorphic convolution operations on cloud
      ↓
SIMD-packed encrypted activations
      ↓
Encrypted logistic regression head
      ↓
Encrypted prediction → Patient device
      ↓
CKKS decryption → probability scores
```

- Speed: ~30–300 seconds depending on circuit depth
- Privacy: Full pixel-level privacy — server operates only on ciphertexts
- Security: 128-bit CKKS with bootstrapping

---

## 3. Concrete ML Integration

### Why ResNet18 Cannot Run Fully Under Concrete ML

Concrete ML compiles PyTorch models into FHE circuits. Full ResNet18 has 18 layers with ReLU activations. Each ReLU requires a **programmable bootstrapping** operation in FHE, which takes seconds per layer. 18 layers × ~3s = ~54 seconds minimum — impractical.

**Solution: Hybrid Architecture**

```
Patient Device (plaintext)          Cloud Server (encrypted)
──────────────────────────          ─────────────────────────
Image (224×224)                     Encrypted feature vector
      ↓                                       ↓
ResNet18 feature extractor          Concrete ML classifier
(frozen, pretrained)                (LogisticRegression / small NN)
      ↓                                       ↓
512-dim feature vector              Encrypted prediction
      ↓                             (cancer probability)
Quantize to 8-bit integers
      ↓
FHE Encrypt
      ↓ ─────────────────────────────────────►
```

### Step-by-Step Concrete ML Implementation

#### Step 1 — Feature Extraction (Patient Device)
```python
import torch
import torchvision.models as models
import numpy as np

# Load pretrained ResNet18, remove final FC layer
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.eval()

def extract_features(image_tensor):
    with torch.no_grad():
        features = feature_extractor(image_tensor)
    return features.squeeze().numpy()  # shape: (512,)
```

#### Step 2 — Train Concrete ML Classifier
```python
from concrete.ml.sklearn import LogisticRegression
from sklearn.model_selection import train_test_split

# features shape: (N, 512), labels: 0=non-cancer, 1=cancer
X_train, X_test, y_train, y_test = train_test_split(features, labels)

# n_bits controls quantization precision (tradeoff: accuracy vs FHE speed)
model = LogisticRegression(n_bits=8)
model.fit(X_train, y_train)

# Compile to FHE circuit
model.compile(X_train)
print("FHE circuit compiled successfully")
```

#### Step 3 — Key Generation
```python
# Generate FHE keys on patient device
from concrete.ml.deployment import FHEModelClient

client = FHEModelClient(path_dir="./fhe_model", key_dir="./patient_keys")
client.generate_private_and_evaluation_keys()
serialized_eval_keys = client.get_serialized_evaluation_keys()
# Send serialized_eval_keys to server (public — does not reveal private key)
```

#### Step 4 — Encrypt and Send
```python
# Patient encrypts their feature vector
features = extract_features(patient_image)
encrypted_input = client.quantize_encrypt_serialize(features)
# Send encrypted_input to server
```

#### Step 5 — Encrypted Inference on Server
```python
from concrete.ml.deployment import FHEModelServer

server = FHEModelServer(path_dir="./fhe_model")
server.load()

# Server receives: encrypted_input + serialized_eval_keys
encrypted_result = server.run(encrypted_input, serialized_eval_keys)
# Send encrypted_result back to patient
```

#### Step 6 — Decryption on Patient Device
```python
# Patient decrypts the result
result = client.deserialize_decrypt_dequantize(encrypted_result)
cancer_probability = float(result[0])
cancer_detected = cancer_probability > 0.5
```

### Supported Concrete ML Models

| Model | FHE Compatible | Accuracy | Speed | Recommended |
|---|---|---|---|---|
| LogisticRegression | ✅ Yes | ~82% | ~2s | ✅ Best for FHE |
| DecisionTree | ✅ Yes | ~79% | ~1s | Good |
| XGBoost | ✅ Yes | ~84% | ~5s | Good |
| Small NN (2 layers) | ✅ Yes | ~86% | ~8s | Best accuracy |
| Full ResNet18 | ❌ No | 88% | ~54s+ | Not practical |

### Performance Limitations

- Each multiplication gate in FHE adds noise — bootstrapping resets noise but costs time
- 8-bit quantization reduces accuracy by ~2-4% vs float32
- RAM requirement: ~4-8 GB for medium models
- GPU does NOT accelerate Concrete ML (CPU only currently)

---

## 4. OpenFHE Integration

### FHE Scheme Selection

| Scheme | Best For | Supports | Noise |
|---|---|---|---|
| CKKS | Approximate arithmetic, real numbers, neural networks | +, ×, approx | Approximate |
| BFV | Exact integer arithmetic | +, × | Exact |
| BGV | Exact integer arithmetic (batched) | +, × | Exact |

**For medical image inference: CKKS** — neural network activations are real numbers, approximate results acceptable for classification.

### Key OpenFHE Concepts

| Concept | Meaning |
|---|---|
| Ciphertext | Encrypted data — server operates on this |
| Plaintext | Unencrypted data — patient holds this |
| Key Switching | Converts ciphertext from one key to another |
| Modulus Switching | Reduces ciphertext size after multiplication |
| Relinearization | Reduces ciphertext size after mult (keeps 2 polynomials) |
| SIMD Packing | Pack N values into one ciphertext (batch processing) |
| Noise Budget | Operations consume noise budget — bootstrapping resets it |
| Multiplicative Depth | Number of sequential multiplications supported |
| Bootstrapping | Refreshes noise budget to allow deeper circuits |

### OpenFHE Workflow for Medical Image Inference

#### Step 1 — Parameter Setup (Server, one-time)
```python
from openfhe import *

parameters = CCParamsCKKSRNS()
parameters.SetMultiplicativeDepth(12)   # supports 12 sequential multiplications
parameters.SetScalingModSize(50)        # precision bits
parameters.SetBatchSize(4096)           # SIMD slots
parameters.SetSecurityLevel(HEStd_128_classic)

cc = GenCryptoContext(parameters)
cc.Enable(PKESchemeFeature.PKE)
cc.Enable(PKESchemeFeature.KEYSWITCH)
cc.Enable(PKESchemeFeature.LEVELEDSHE)
cc.Enable(PKESchemeFeature.ADVANCEDSHE)  # for bootstrapping
```

#### Step 2 — Key Generation (Patient Device)
```python
# Patient generates key pair
keypair = cc.KeyGen()
public_key = keypair.publicKey     # sent to server
private_key = keypair.secretKey    # never leaves patient device

# Generate evaluation keys (for server-side operations)
cc.EvalMultKeyGen(keypair.secretKey)
cc.EvalRotateKeyGen(keypair.secretKey, [1, 2, 4, 8, 16])
```

#### Step 3 — Encrypt Image Features (Patient Device)
```python
# Extract 512-dim feature vector from ResNet18 on patient device
features = extract_features(patient_image)   # shape: (512,)

# Encode and encrypt using CKKS
plaintext = cc.MakeCKKSPackedPlaintext(features.tolist())
ciphertext = cc.Encrypt(keypair.publicKey, plaintext)

# Send ciphertext + public eval keys to server
```

#### Step 4 — Homomorphic Inference (Server)
```python
# Server performs encrypted matrix-vector multiplication
# Weight matrix W is public (model weights), input is encrypted

# Encrypted dot product: W @ encrypted_features + b
def encrypted_linear(cc, ciphertext, weights, bias):
    result = cc.EvalMult(ciphertext, weights[0])
    for i in range(1, len(weights)):
        term = cc.EvalMult(ciphertext, weights[i])
        result = cc.EvalAdd(result, term)
    bias_pt = cc.MakeCKKSPackedPlaintext([bias])
    result = cc.EvalAdd(result, bias_pt)
    return result

# Apply sigmoid approximation (polynomial: 0.5 + 0.197x - 0.004x^3)
def encrypted_sigmoid_approx(cc, ciphertext):
    x3 = cc.EvalMult(cc.EvalMult(ciphertext, ciphertext), ciphertext)
    term1 = cc.EvalMult(ciphertext, 0.197)
    term2 = cc.EvalMult(x3, -0.004)
    result = cc.EvalAdd(cc.EvalAdd(term1, term2), 0.5)
    return result

encrypted_output = encrypted_linear(cc, ciphertext, W, b)
encrypted_prob = encrypted_sigmoid_approx(cc, encrypted_output)
# Return encrypted_prob to patient
```

#### Step 5 — Decryption (Patient Device)
```python
plaintext_result = cc.Decrypt(private_key, encrypted_prob)
cancer_probability = plaintext_result.GetRealPackedValue()[0]
cancer_detected = cancer_probability > 0.5
```

### Bootstrapping for Deep Networks

When multiplicative depth is exhausted (noise too high):
```python
# Refresh ciphertext noise budget
cc.Enable(PKESchemeFeature.FHE)  # enables bootstrapping
bootstrapped = cc.EvalBootstrap(ciphertext)
# Continue with more operations
```

---

## 5. Encrypted Medical Workflow

### Complete End-to-End Privacy Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        PATIENT DEVICE                           │
│                                                                 │
│  1. Upload MRI/CT/Skin image                                    │
│  2. ResNet18 extracts 512-dim feature vector (local)            │
│  3. Quantize features to 8-bit integers                         │
│  4. Generate FHE keys (private key NEVER leaves device)         │
│  5. Encrypt feature vector with public key                      │
│  6. Send: encrypted_features + eval_keys → Server              │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTPS (encrypted transport)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        CLOUD SERVER                             │
│                                                                 │
│  7. Receive encrypted features (server CANNOT decrypt)          │
│  8. Load FHE model (weights are public/model parameters)        │
│  9. Run encrypted inference: W_enc @ features_enc + b_enc       │
│ 10. Apply polynomial activation approximation                   │
│ 11. Return encrypted_prediction → Patient                       │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTPS
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        PATIENT DEVICE                           │
│                                                                 │
│ 12. Decrypt encrypted_prediction with private key               │
│ 13. Get cancer probability (0.0 – 1.0)                          │
│ 14. Display result with confidence score                        │
│ 15. Optional: LLM explanation (run locally or via secure API)   │
└─────────────────────────────────────────────────────────────────┘
```

### Privacy Guarantees

| Property | Guarantee |
|---|---|
| Image privacy | Raw pixels never leave patient device |
| Feature privacy | 512-dim features encrypted before upload |
| Prediction privacy | Server returns encrypted result only |
| Key privacy | Private key never transmitted |
| Model privacy | Server weights remain proprietary |
| Transport | TLS 1.3 for all communications |

### GDPR / HIPAA Compliance

- No PHI (Protected Health Information) stored in plaintext on server
- Encrypted predictions stored with patient-controlled decryption
- Audit log records access without exposing content
- Right to erasure: delete ciphertext = data destroyed

---

## 6. Research Benchmark Comparison

### Complete Comparison Table

| Metric | Standard AI | Concrete ML | OpenFHE CKKS |
|---|---|---|---|
| **Accuracy** | 88.75% | ~84–86% | ~82–85% |
| **Precision** | 96.70% | ~91% | ~89% |
| **Recall** | 88.00% | ~84% | ~82% |
| **F1 Score** | 92.15% | ~87% | ~85% |
| **Inference Time** | ~50ms | ~2–10s | ~30–300s |
| **Encryption Time** | 0 | ~100–500ms | ~1–5s |
| **Decryption Time** | 0 | ~10–50ms | ~100–500ms |
| **Memory Usage** | ~500MB | ~2–4GB | ~8–32GB |
| **CPU Usage** | Low | Medium | Very High |
| **Latency (total)** | ~100ms | ~5–15s | ~60–600s |
| **Ciphertext Size** | N/A | ~10–50KB | ~1–10MB |
| **Security Level** | None | 128-bit | 128-bit |
| **Scalability** | High | Medium | Low |
| **Model Size** | ~45MB (ResNet18) | ~1–5MB | ~1MB weights |
| **GPU Support** | ✅ Yes | ❌ No | ❌ No |
| **Patient Privacy** | ❌ None | ✅ Feature-level | ✅ Pixel-level |
| **HIPAA Compatible** | ⚠️ Requires controls | ✅ Yes | ✅ Yes |

### Supported Neural Networks per Framework

| Architecture | Standard AI | Concrete ML | OpenFHE |
|---|---|---|---|
| ResNet18/50 | ✅ Full | ✅ Feature extractor only | ✅ Small head only |
| LogisticRegression | ✅ | ✅ Full | ✅ Full |
| XGBoost | ✅ | ✅ Full | ❌ Not practical |
| MLP (2-layer) | ✅ | ✅ Full | ✅ With poly activation |
| EfficientNet | ✅ | ✅ Extractor only | ✅ Head only |
| ViT | ✅ | ❌ Too deep | ❌ Too deep |

### Accuracy vs Privacy Tradeoff

```
Privacy Level  HIGH ──────────────────────────────►
               │
Accuracy HIGH  │  Standard AI ●
               │
               │               Concrete ML ●
               │
               │                              OpenFHE ●
Accuracy LOW   │
               └────────────────────────────────────►
                     No Privacy    FHE-Feature   FHE-Pixel
```

### Advantages and Disadvantages

**Standard AI**
- ✅ Fastest, highest accuracy, full model capacity
- ❌ No privacy — server sees raw patient data

**Concrete ML**
- ✅ Strong privacy with acceptable accuracy loss
- ✅ Practical inference time (2–10s)
- ✅ Drop-in sklearn-compatible API
- ❌ Accuracy ~2–4% lower than standard
- ❌ Limited model architectures

**OpenFHE**
- ✅ Maximum privacy — server sees only ciphertexts
- ✅ NIST-approved cryptographic primitives
- ✅ Suitable for regulatory compliance
- ❌ Very slow (minutes per inference)
- ❌ Requires significant engineering
- ❌ High memory requirements

---

## 7. Deep Learning Model Upgrade

### Model Comparison for Medical Imaging

| Model | Params | Accuracy | FHE-Compatible | Speed | Recommended |
|---|---|---|---|---|---|
| ResNet18 (current) | 11M | 88.75% | ✅ Extractor | Fast | Current baseline |
| ResNet50 | 25M | ~91% | ✅ Extractor | Medium | ✅ Best upgrade |
| EfficientNet-B3 | 12M | ~93% | ✅ Extractor | Medium | ✅ Best accuracy/size |
| DenseNet121 | 8M | ~91% | ✅ Extractor | Medium | Good for lung |
| ConvNeXt-Tiny | 28M | ~93% | ✅ Extractor | Medium | ✅ Modern CNN |
| ViT-Base | 86M | ~94% | ❌ Too deep for FHE | Slow | Research only |
| MobileNetV3 | 5M | ~87% | ✅ Full possible | Very fast | Edge deployment |
| Ensemble (3 models) | 3×11M | ~95% | ✅ Each extractor | Slow | ✅ Best overall |

### Recommended Architecture: Ensemble + EfficientNet

```python
import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

class CancerEnsemble(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        # Model 1: ResNet50 (general features)
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()

        # Model 2: EfficientNet-B3 (efficient features)
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b3')
        self.efficientnet._fc = nn.Identity()

        # Model 3: DenseNet121 (dense features)
        self.densenet = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.densenet.classifier = nn.Identity()

        # Fusion head
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 1536 + 1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        f1 = self.resnet(x)        # (B, 2048)
        f2 = self.efficientnet(x)  # (B, 1536)
        f3 = self.densenet(x)      # (B, 1024)
        combined = torch.cat([f1, f2, f3], dim=1)
        return self.classifier(combined)
```

### Best Model for FHE

**EfficientNet-B3 as feature extractor + Concrete ML LogisticRegression head**

Reasons:
1. EfficientNet uses compound scaling — best accuracy per parameter
2. 12M parameters — smaller than ResNet50, faster feature extraction on patient device
3. 1536-dim features — richer than ResNet18's 512-dim
4. LogisticRegression FHE head — minimal multiplicative depth (1 multiplication)
5. Total FHE inference time: ~2–3 seconds

### Vision Transformer for Research

```python
from transformers import ViTForImageClassification

vit = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=14,
    ignore_mismatched_sizes=True
)
```

ViT achieves ~94% accuracy but is NOT suitable for FHE due to attention mechanisms requiring many multiplications. Use it for standard mode only as the highest-accuracy option.

---

## 8. Explainable AI

### Why XAI Matters for Doctors

A doctor cannot trust a black-box prediction. XAI generates visual explanations showing *which regions* of the image drove the prediction.

### Grad-CAM Implementation

```python
import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, class_idx=None):
        self.model.eval()
        output = self.model(image_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        # Pool gradients over spatial dimensions
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear')
        cam = cam.squeeze().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# Usage: attach to last conv layer of ResNet
target_layer = model.layer4[-1].conv2
gradcam = GradCAM(model, target_layer)
heatmap = gradcam.generate(image_tensor)
```

### Integrated Gradients

```python
def integrated_gradients(model, image, baseline=None, steps=50):
    if baseline is None:
        baseline = torch.zeros_like(image)

    # Interpolate between baseline and image
    alphas = torch.linspace(0, 1, steps)
    interpolated = [baseline + alpha * (image - baseline) for alpha in alphas]

    gradients = []
    for interp in interpolated:
        interp.requires_grad_(True)
        output = model(interp.unsqueeze(0))
        output.max().backward()
        gradients.append(interp.grad.squeeze())

    avg_gradients = torch.stack(gradients).mean(0)
    integrated_grads = (image - baseline) * avg_gradients
    return integrated_grads.abs().sum(0).numpy()
```

### What Doctors See

```
┌────────────────────────────────────────────────────┐
│  Original MRI      │  Grad-CAM Heatmap              │
│  ┌──────────┐      │  ┌──────────┐                  │
│  │  Brain   │      │  │ 🔴🔴🟡  │ ← Hot = tumor    │
│  │  image   │  →   │  │ 🟡🟢🟢  │                  │
│  └──────────┘      │  └──────────┘                  │
│                                                      │
│  Prediction: Glioma Tumor (94.2% confidence)         │
│  Key regions: Frontal lobe (red area, 87% weight)    │
│  Attending: Temporal sulcus (yellow area, 11%)       │
└────────────────────────────────────────────────────┘
```

### Attention Maps for ViT

Vision Transformers produce attention weights directly:
```python
def get_attention_map(vit_model, image_tensor):
    with torch.no_grad():
        outputs = vit_model(image_tensor, output_attentions=True)
    # Average attention across heads, last layer
    attn = outputs.attentions[-1].mean(1)  # (B, seq_len, seq_len)
    cls_attn = attn[0, 0, 1:]  # CLS token attention to all patches
    attn_map = cls_attn.reshape(14, 14).numpy()
    return cv2.resize(attn_map, (224, 224))
```

---

## 9. Federated Learning

### Architecture

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Hospital A  │   │  Hospital B  │   │  Hospital C  │
│  500 MRIs    │   │  300 CTs     │   │  800 Skins   │
│  Local model │   │  Local model │   │  Local model │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                   │                   │
       │   gradients only  │   gradients only  │
       └───────────────────┴───────────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Central Server │
                  │  FedAvg         │
                  │  Aggregation    │
                  │  Global Model   │
                  └─────────────────┘
                           │
                  Distribute updated weights
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
Hospital A            Hospital B           Hospital C
```

### Implementation with Flower Framework

```python
import flwr as fl
import torch

class CancerFederatedClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    def get_parameters(self, config):
        # Return model weights as NumPy arrays
        return [p.numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        # Set global model weights
        for p, w in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(w)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # Train locally for config["local_epochs"] epochs
        train_local(self.model, self.train_loader,
                    epochs=config.get("local_epochs", 3))
        return self.get_parameters({}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = evaluate_local(self.model, self.val_loader)
        return loss, len(self.val_loader.dataset), {"accuracy": accuracy}

# Server-side FedAvg aggregation
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,           # use all clients each round
    min_fit_clients=2,          # minimum 2 hospitals per round
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=weighted_average,
)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=20),
    strategy=strategy,
)
```

### Security in Federated Learning

| Attack | Defence |
|---|---|
| Gradient inversion | Gradient compression + DP noise |
| Model poisoning | Median/Trimmed-mean aggregation |
| Membership inference | Differential privacy |
| Communication interception | TLS + encrypted gradients |
| Byzantine clients | FedProx / robust aggregation |

### Communication Protocol

```
Round 1:
  Server → Hospitals: initial model weights (W_0)
  Hospitals → Server: gradients ΔW_1, ΔW_2, ΔW_3

Round 2:
  Server: W_1 = W_0 + FedAvg(ΔW_1, ΔW_2, ΔW_3)
  Server → Hospitals: W_1
  ...repeat for N rounds
```

---

## 10. Differential Privacy

### Concept

Differential Privacy adds mathematically calibrated noise to gradients during training, so an adversary cannot determine whether any individual patient's data was in the training set.

```
Real gradient: [0.23, -0.45, 0.12, ...]
DP gradient:   [0.25, -0.43, 0.15, ...]  ← noise added
```

### Implementation with Opacus

```python
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import torch.optim as optim

# Make model compatible with DP
model = ModuleValidator.fix(model)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

privacy_engine = PrivacyEngine()

model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    epochs=25,
    target_epsilon=8.0,    # privacy budget (lower = more private)
    target_delta=1e-5,     # probability of privacy failure
    max_grad_norm=1.0,     # gradient clipping
)

# Training loop (unchanged)
for epoch in range(25):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()  # automatically clips + adds noise

    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    print(f"Epoch {epoch}: ε = {epsilon:.2f}")
```

### Privacy Budget Interpretation

| ε (epsilon) | Privacy Level | Accuracy Impact |
|---|---|---|
| ε = 1 | Very strong privacy | ~5–10% accuracy loss |
| ε = 3 | Strong privacy | ~3–5% accuracy loss |
| ε = 8 | Moderate privacy | ~1–3% accuracy loss |
| ε = 50 | Weak privacy | ~0.5% accuracy loss |
| ε = ∞ | No privacy | 0% accuracy loss |

### Advantages and Limitations

**Advantages**
- Mathematical privacy guarantee — provable bound on information leakage
- Protects against membership inference attacks
- Compatible with existing PyTorch training loops
- Works with federated learning

**Limitations**
- Accuracy degrades with stronger privacy (lower ε)
- Requires large batch sizes for good utility
- Hyperparameter tuning is more complex
- Does not protect against all attacks (e.g., model inversion with very low ε)

---

## 11. Improved RAG Pipeline

### Current vs Improved

| Component | Current | Improved |
|---|---|---|
| Retrieval | FAISS only | BM25 + FAISS Hybrid |
| Reranking | None | Cross-Encoder Reranker |
| Knowledge | PDFs only | PDFs + PubMed + WHO + NCCN |
| Filtering | None | Metadata + date + source filters |
| Confidence | None | Confidence scoring per chunk |
| Hallucination | Basic prompt | Medical safety guardrails |

### Hybrid Search Architecture

```python
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import faiss

class HybridMedicalRAG:
    def __init__(self):
        self.bm25 = None          # keyword search
        self.faiss_index = None   # semantic search
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.docs = []

    def retrieve(self, query: str, top_k: int = 20) -> list:
        # Step 1: BM25 keyword retrieval
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_top = bm25_scores.argsort()[-top_k:][::-1]

        # Step 2: FAISS semantic retrieval
        query_vec = self.embed(query)
        _, faiss_top = self.faiss_index.search(query_vec, top_k)
        faiss_top = faiss_top[0]

        # Step 3: Merge candidates (Reciprocal Rank Fusion)
        candidates = list(set(bm25_top.tolist() + faiss_top.tolist()))
        candidate_docs = [self.docs[i] for i in candidates]

        # Step 4: Cross-encoder reranking
        pairs = [(query, doc['text']) for doc in candidate_docs]
        scores = self.reranker.predict(pairs)

        # Sort by reranker score
        ranked = sorted(
            zip(scores, candidate_docs),
            key=lambda x: x[0], reverse=True
        )
        return [doc for _, doc in ranked[:5]]
```

### Medical Knowledge Graph

```
                    Cancer
                   /  |  \
           Glioma  Melanoma  Lung Cancer
              |        |          |
          Symptoms  Treatment  Risk Factors
              |        |          |
          Headache  Surgery  Smoking
              |        |          |
           ...     Chemo      ...
```

### Medical Safety Guardrails

```python
MEDICAL_SAFETY_RULES = [
    "NEVER provide specific drug dosages",
    "ALWAYS recommend consulting a doctor",
    "NEVER diagnose definitively — use 'may indicate'",
    "ALWAYS include disclaimer on AI limitations",
    "NEVER override doctor's clinical judgement",
]

HALLUCINATION_DETECTION = {
    "min_source_confidence": 0.7,
    "require_citation": True,
    "max_unsupported_claims": 0,
    "verify_drug_names_against_db": True,
}
```

---

## 12. Doctor Dashboard

### Dashboard Tabs

| Tab | Contents |
|---|---|
| Patient Overview | Patient list, risk scores, recent scans |
| Scan History | Timeline of uploaded images with predictions |
| Disease Progression | Charts showing confidence scores over time |
| Grad-CAM Viewer | Side-by-side original + heatmap overlay |
| Encrypted Predictions | Audit log of FHE inference results |
| Research Statistics | Model performance, dataset stats |
| Benchmark Results | Standard vs Concrete ML vs OpenFHE comparison |

### Patient Timeline View

```
Patient: John Doe  |  Cancer Type: Glioma (suspected)
─────────────────────────────────────────────────────
Date        Scan Type   Prediction    Confidence   Mode
────────────────────────────────────────────────────────
2026-01-15  Brain MRI   Cancer        94.2%        Standard
2026-03-20  Brain MRI   Cancer        91.8%        Concrete ML
2026-05-10  Brain MRI   Cancer        89.5%        Standard
2026-07-01  Brain MRI   Cancer        96.1%        OpenFHE

Trend: ↑ Confidence increasing → recommend biopsy
```

### Grad-CAM Visualization Component (React)

```tsx
interface GradCAMViewerProps {
  originalImage: string     // base64
  heatmapImage: string      // base64 Grad-CAM overlay
  cancerType: string
  confidence: number
  keyRegions: Array<{ name: string; weight: number }>
}

export function GradCAMViewer({ originalImage, heatmapImage,
  cancerType, confidence, keyRegions }: GradCAMViewerProps) {
  return (
    <div className="grid grid-cols-2 gap-4">
      <div>
        <p className="text-sm font-medium mb-2">Original Scan</p>
        <img src={originalImage} className="rounded-xl w-full" />
      </div>
      <div>
        <p className="text-sm font-medium mb-2">AI Attention Map</p>
        <img src={heatmapImage} className="rounded-xl w-full" />
        <div className="mt-3 space-y-1">
          {keyRegions.map(r => (
            <div key={r.name} className="flex justify-between text-xs">
              <span>{r.name}</span>
              <span className="font-bold text-red-500">{r.weight}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
```

---

## 13. Hospital Integration

### FHIR R4 Integration

```python
from fhirclient import client
from fhirclient.models import patient, observation, imagingstudy

# Connect to hospital FHIR server
settings = {
    'app_id': 'medical_cancer_expert',
    'api_base': 'https://hospital-fhir.example.com/fhir/R4'
}
smart = client.FHIRClient(settings=settings)

# Fetch patient data
pt = patient.Patient.read('patient-123', smart.server)

# Store AI prediction as FHIR Observation
obs = observation.Observation({
    'status': 'final',
    'code': {'coding': [{'system': 'http://loinc.org', 'code': '24627-2',
                         'display': 'Chest X-ray AP'}]},
    'subject': {'reference': f'Patient/{pt.id}'},
    'valueCodeableConcept': {
        'coding': [{'code': 'malignant', 'display': 'Malignant finding'}]
    },
    'component': [{
        'code': {'text': 'AI Confidence'},
        'valueQuantity': {'value': 94.2, 'unit': '%'}
    }]
})
obs.create(smart.server)
```

### Integration Points

| System | Protocol | Data Exchanged |
|---|---|---|
| HIS (Hospital Info System) | HL7 v2 / FHIR | Patient demographics |
| PACS (Radiology) | DICOM / FHIR | MRI/CT images |
| EMR (Electronic Medical Records) | FHIR R4 | Patient history |
| LIS (Laboratory) | HL7 v2 | Lab results |
| Pathology System | FHIR | Biopsy results |

---

## 14. Security Architecture

### Zero Trust Architecture

```
Every request verified regardless of source location:
  1. Identity verification (JWT + MFA)
  2. Device verification (certificate pinning)
  3. Network verification (mTLS)
  4. Request authorization (RBAC per endpoint)
  5. Audit logging (every access recorded)
```

### Role-Based Access Control

| Role | Permissions |
|---|---|
| Patient | Upload images, view own results, chat |
| Doctor | View patient history, Grad-CAM, doctor notes |
| Researcher | Benchmark dashboard, anonymized data, metrics |
| Admin | All above + user management, system config |
| System | Internal service-to-service calls only |

### Threat Model

| Threat | Attack | Defence |
|---|---|---|
| Data breach | SQL injection | Parameterized queries, ORM |
| Auth bypass | JWT forgery | RS256 signing, short expiry |
| Image theft | MITM | TLS 1.3, HSTS |
| Model theft | Model extraction | Rate limiting, output truncation |
| Patient re-identification | De-anonymization | k-anonymity, DP noise |
| FHE key theft | Key exfiltration | Keys never leave patient device |
| Insider threat | Admin data access | Audit logs, encryption at rest |

---

## 19. UI Pages Design

### Page List and Purpose

| Page | Route | Who Can Access | Purpose |
|---|---|---|---|
| Home | `/` | Public | Landing page, project overview |
| Login | `/login` | Public | JWT authentication |
| Register | `/register` | Public | New user registration |
| Dashboard | `/dashboard` | All users | Personal overview, quick actions |
| Medical Chat | `/chat` | All users | RAG chatbot interface |
| Image Analysis | `/analyze` | All users | Standard ResNet18 inference |
| Encrypted AI | `/encrypted` | All users | FHE inference (Concrete ML / OpenFHE) |
| Benchmark | `/benchmark` | Researcher, Admin | Mode comparison dashboard |
| Research | `/research` | Researcher, Admin | Experiment management |
| Doctor Dashboard | `/doctor` | Doctor, Admin | Patient management + Grad-CAM |
| Admin Dashboard | `/admin` | Admin | Full system management |
| System Monitor | `/monitor` | Admin | Prometheus + health metrics |
| Settings | `/settings` | All users | Profile, API keys, preferences |

### Encrypted AI Page Layout

```
┌────────────────────────────────────────────────────────────────┐
│  🔐 Encrypted AI Inference                                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Inference Mode:  ○ Standard  ● Concrete ML  ○ OpenFHE        │
│                                                                │
│  ┌─────────────────────────────────┐                          │
│  │  Drop image here or click       │                          │
│  │  (Feature extraction runs       │                          │
│  │   locally — image stays private)│                          │
│  └─────────────────────────────────┘                          │
│                                                                │
│  Privacy Status:                                               │
│  ✅ Feature extraction: LOCAL (image never uploaded)           │
│  ✅ Encryption: AES-128 + FHE 128-bit                          │
│  ✅ Private key: NEVER leaves your device                      │
│  ✅ Server sees: encrypted ciphertext only                     │
│                                                                │
│  [Generate FHE Keys]  [Encrypt & Analyze]                      │
│                                                                │
│  ──────────────────────────────────────────────────            │
│  Result (after decryption on your device):                     │
│  🔒 Cancer Detected: Yes                                       │
│  🔒 Type: Glioma Tumor                                         │
│  🔒 Confidence: 91.4%                                          │
│  ⏱  Total time: 6.2s (encrypt: 0.4s, infer: 5.5s, decrypt: 0.3s)│
└────────────────────────────────────────────────────────────────┘
```

### Benchmark Dashboard Layout

```
┌────────────────────────────────────────────────────────────────┐
│  📊 Inference Mode Benchmark                                   │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  [Run New Benchmark]  [Export CSV]  [Export PDF]               │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Accuracy Comparison                                    │  │
│  │  Standard AI ████████████████████ 88.75%               │  │
│  │  Concrete ML ██████████████████   84.20%               │  │
│  │  OpenFHE     █████████████████    82.10%               │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  Inference Latency (ms, log scale)                      │  │
│  │  Standard AI    ▌  50ms                                 │  │
│  │  Concrete ML    ████████ 5,000ms                        │  │
│  │  OpenFHE        ████████████████████████ 120,000ms      │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
│  Full Comparison Table:                                        │
│  Metric          Standard   Concrete ML   OpenFHE             │
│  ──────────────────────────────────────────────────────       │
│  Accuracy        88.75%     84.20%        82.10%              │
│  Precision       96.70%     91.00%        89.00%              │
│  Recall          88.00%     84.00%        82.00%              │
│  F1 Score        92.15%     87.40%        85.40%              │
│  Inference (ms)  50         5,000         120,000             │
│  Privacy         None       Feature-FHE   Pixel-FHE           │
│  Security bits   0          128           128                  │
└────────────────────────────────────────────────────────────────┘
```

### Doctor Dashboard Layout

```
┌────────────────────────────────────────────────────────────────┐
│  👨‍⚕️ Doctor Dashboard                                          │
├──────────────┬─────────────────────────────────────────────────┤
│  Patients    │  Patient: Ahmed Ali   ID: #1042                  │
│  ─────────── │  ─────────────────────────────────────────────  │
│  Ahmed Ali ● │  Disease Progression                            │
│  Sara Ben    │  Confidence %                                   │
│  John Doe    │  100 ─                        ●  96%            │
│  ...         │   90 ─         ●  94%   ●  92%                  │
│              │   80 ─  ●  88%                                   │
│  [+ Add]     │       Jan    Mar    May    Jul                   │
│              │                                                  │
│              │  Latest Scan: 2026-07-01  Glioma (96%)          │
│              │                                                  │
│              │  ┌─────────────┐ ┌─────────────┐               │
│              │  │ Original    │ │ Grad-CAM    │               │
│              │  │ [MRI scan]  │ │ [Heatmap]   │               │
│              │  └─────────────┘ └─────────────┘               │
│              │                                                  │
│              │  Key Region: Frontal lobe (87%)                 │
│              │  Secondary: Temporal sulcus (11%)               │
│              │                                                  │
│              │  Doctor Notes:                                   │
│              │  [Recommend biopsy — confidence increasing]      │
│              │  [Add Note ▼]                                    │
└──────────────┴─────────────────────────────────────────────────┘
```

---

## 20. Deployment Guide

### Docker Compose (Development + Production)

```yaml
# docker-compose.yml
version: '3.9'

services:

  nginx:
    image: nginx:1.25-alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - backend
      - frontend

  backend:
    build:
      context: ./backend
      dockerfile: ../docker/Dockerfile.backend
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - DATABASE_URL=${DATABASE_URL}
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
    volumes:
      - ./backend:/app
      - model_data:/app/models
    ports:
      - "8000:8000"
    depends_on:
      - mysql
      - redis

  frontend:
    build:
      context: ./frontend
      dockerfile: ../docker/Dockerfile.frontend
    ports:
      - "3000:3000"

  mysql:
    image: mysql:8.0
    environment:
      MYSQL_ROOT_PASSWORD: ${MYSQL_ROOT_PASSWORD}
      MYSQL_DATABASE: medical_cancer_expert
    volumes:
      - mysql_data:/var/lib/mysql
    ports:
      - "3306:3306"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}

volumes:
  mysql_data:
  redis_data:
  model_data:
  grafana_data:
```

### Dockerfile — Backend

```dockerfile
# docker/Dockerfile.backend
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ cmake libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

EXPOSE 8000
CMD ["uvicorn", "main_v2:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### NGINX Configuration

```nginx
# nginx/nginx.conf
events { worker_connections 1024; }

http {
    upstream backend  { server backend:8000; }
    upstream frontend { server frontend:3000; }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=30r/m;

    server {
        listen 80;
        return 301 https://$host$request_uri;
    }

    server {
        listen 443 ssl http2;
        ssl_certificate     /etc/ssl/certs/cert.pem;
        ssl_certificate_key /etc/ssl/certs/key.pem;
        ssl_protocols       TLSv1.3;

        # API
        location /api/ {
            limit_req zone=api burst=10 nodelay;
            proxy_pass http://backend/;
            proxy_set_header Authorization $http_authorization;
            proxy_set_header X-Real-IP $remote_addr;
        }

        # Frontend
        location / {
            proxy_pass http://frontend;
        }
    }
}
```

### CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r backend/requirements.txt
      - run: pytest backend/ -v --timeout=30

  test-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '20' }
      - run: cd frontend && npm ci && npm run build

  deploy:
    needs: [test-backend, test-frontend]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Deploy via Docker Compose
        run: |
          docker-compose -f docker-compose.yml build
          docker-compose -f docker-compose.yml up -d
```

### Prometheus Metrics Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'medical-cancer-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

### Grafana Dashboard Panels

| Panel | Metric | Description |
|---|---|---|
| Inference Rate | requests/sec per mode | Standard vs FHE usage |
| Accuracy Over Time | val_acc per epoch | Model performance trend |
| FHE Latency | p50/p95/p99 ms | Encrypted inference latency |
| Active Users | active_sessions | Concurrent users |
| Cancer Detections | cancer_detected_total | Positive detection count |
| Memory Usage | process_resident_memory | RAM per service |
| Error Rate | http_errors/total | 4xx + 5xx rate |
| Federated Rounds | fl_rounds_completed | Training progress |

---

## 21. Research Contributions

This platform contributes to the following research areas:

### Novel Contributions

| # | Contribution | Field |
|---|---|---|
| 1 | Hybrid FHE inference: ResNet extractor + Concrete ML classifier | Privacy-Preserving ML |
| 2 | CKKS-based medical image classification with polynomial activations | Homomorphic Encryption |
| 3 | Three-mode benchmark: Standard vs Concrete ML vs OpenFHE | Privacy-ML Benchmarking |
| 4 | Federated learning across heterogeneous hospital datasets | Healthcare AI |
| 5 | DP-SGD training with medical class imbalance correction | Differential Privacy |
| 6 | Hybrid BM25+FAISS+CrossEncoder RAG for oncology Q&A | Medical NLP |
| 7 | Grad-CAM + attention maps for encrypted model explanations | XAI in Healthcare |
| 8 | FHIR R4 integration with encrypted AI predictions | Clinical Informatics |

### Publication Targets

| Venue | Type | Topic |
|---|---|---|
| IEEE S&P / CCS | Conference | FHE medical inference + privacy guarantees |
| NeurIPS / ICML | Conference | Federated learning + differential privacy |
| MICCAI | Conference | Medical image analysis + XAI |
| Nature Digital Medicine | Journal | Clinical AI + privacy |
| JAMIA | Journal | FHIR integration + EHR AI |

### Thesis Chapter Structure

```
Chapter 1: Introduction
  - Problem: patient privacy in cloud AI
  - Motivation: GDPR, HIPAA compliance
  - Contributions

Chapter 2: Background
  - Homomorphic Encryption (CKKS, BFV, BGV)
  - Federated Learning
  - Differential Privacy
  - Medical Image Analysis

Chapter 3: System Architecture
  - Three inference modes
  - Hybrid FHE architecture
  - Security model

Chapter 4: Privacy-Preserving Inference
  - Concrete ML implementation
  - OpenFHE implementation
  - Performance analysis

Chapter 5: Federated Medical AI
  - Multi-hospital training
  - Secure aggregation
  - Convergence analysis

Chapter 6: Evaluation
  - Benchmark: accuracy vs privacy vs latency
  - Ablation studies
  - Clinical usability study

Chapter 7: Conclusion and Future Work
```

---

## 22. Future Work

### Short-term (3–6 months)

| Task | Priority | Effort |
|---|---|---|
| Train EfficientNet-B3 + ResNet50 ensemble | High | 2 weeks |
| Implement Concrete ML LogisticRegression pipeline | High | 1 week |
| Add Grad-CAM to all inference modes | High | 1 week |
| Build benchmark dashboard frontend | Medium | 2 weeks |
| Integrate Flower federated learning (2 nodes) | Medium | 3 weeks |

### Medium-term (6–12 months)

| Task | Priority | Effort |
|---|---|---|
| Full OpenFHE CKKS pipeline for linear head | High | 1 month |
| DP-SGD training with Opacus | High | 2 weeks |
| FHIR R4 integration prototype | Medium | 1 month |
| Doctor dashboard with Grad-CAM viewer | Medium | 3 weeks |
| Multi-hospital federated experiment (3+ nodes) | High | 2 months |

### Long-term (12+ months)

| Task | Priority | Effort |
|---|---|---|
| Full pixel-level FHE inference (OpenFHE CNN) | Research | 6 months |
| Vision Transformer under CKKS | Research | 4 months |
| ViT attention under FHE (approximated) | Research | 6 months |
| Clinical trial integration | High | 1 year |
| Regulatory CE/FDA submission preparation | High | 1 year |
| Multi-party computation (MPC) inference | Research | 8 months |

### Open Research Questions

1. **Can ViT attention be approximated polynomially** for use under CKKS with bounded noise?
2. **What is the minimum FHE multiplicative depth** required for clinically acceptable accuracy?
3. **How does federated learning convergence** change with heterogeneous hospital data distributions (non-IID)?
4. **What epsilon value** provides sufficient DP protection for medical images without exceeding 3% accuracy loss?
5. **Can Grad-CAM heatmaps be computed** on encrypted activations without decryption?

---

## Implementation Roadmap

### Phase 1 — Foundation (Weeks 1–4)
```
Week 1: Set up new folder structure, Docker, NGINX
Week 2: Implement Concrete ML feature extractor + classifier
Week 3: Build encrypted inference API endpoints
Week 4: Frontend: Encrypted AI page + key management
```

### Phase 2 — XAI + Benchmarking (Weeks 5–8)
```
Week 5: Grad-CAM for all CNN models
Week 6: Benchmark service + database tables
Week 7: Benchmark dashboard frontend (charts)
Week 8: Research experiment management
```

### Phase 3 — Federated + Privacy (Weeks 9–12)
```
Week 9:  Flower FL server + client setup
Week 10: DP-SGD training with Opacus
Week 11: Doctor dashboard + Grad-CAM viewer
Week 12: Privacy metrics dashboard
```

### Phase 4 — OpenFHE + Clinical (Weeks 13–20)
```
Week 13–15: OpenFHE CKKS pipeline
Week 16–17: FHIR R4 integration prototype
Week 18–19: Audit logging + RBAC
Week 20: Full system testing + documentation
```

---

## Technology Stack Summary

### New Dependencies to Add

```
# backend/requirements.txt additions
concrete-ml==1.6.0          # FHE inference
openfhe==0.8.8               # OpenFHE Python bindings
flower==1.8.0                # Federated learning
opacus==1.4.0                # Differential privacy
rank-bm25==0.2.2             # BM25 retrieval
sentence-transformers==3.0   # Cross-encoder reranker
fhirclient==4.2.0            # FHIR R4 integration
onnx==1.16.0                 # ONNX model export
onnxruntime==1.18.0          # Client-side inference
prometheus-client==0.20.0    # Metrics export
redis==5.0.0                 # Session cache
efficientnet-pytorch==0.7.1  # EfficientNet model

# frontend/package.json additions
"@zama-ai/concrete-ml-wasm": "^1.0.0"  # Client-side FHE (WASM)
"onnxruntime-web": "^1.18.0"            # Client-side feature extraction
"recharts": "^2.12.0"                   # Benchmark charts
"@tanstack/react-query": "^5.0.0"       # Data fetching
"zustand": "^4.5.0"                     # State management
```

---

> **Important Disclaimer**
> This platform is designed for research and educational purposes.
> All clinical applications must undergo regulatory review (CE marking, FDA clearance).
> FHE implementations provided are proof-of-concept — production deployment
> requires security audit and performance optimization.
> Always consult qualified medical professionals for clinical decisions.
