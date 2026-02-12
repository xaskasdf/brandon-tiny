# Deep Dive: Investigación Avanzada para brandon-tiny

> Expansión detallada de las ideas de RESEARCH_NEXT_STEPS.md
> Basado en ~100 papers analizados por 5 agentes de investigación en paralelo.
> Proyecto: brandon-tiny | GPU: RTX 3090 24GB | Modelos: 10M-110M params

---

## Tabla de Contenidos

1. [Looped/Recursive Transformers](#1-loopedrecursive-transformers)
2. [Resonance Buffers y Feedback Transformer](#2-resonance-buffers-y-feedback-transformer)
3. [Computación Adaptativa (Attention Budget, MoD, ACT)](#3-computación-adaptativa)
4. [Dual Brain: Test-Time Compute Scaling](#4-dual-brain-test-time-compute)
5. [BitNet y Arquitecturas Ternarias](#5-bitnet-y-arquitecturas-ternarias)
6. [Síntesis: Arquitectura Óptima Propuesta](#6-síntesis-arquitectura-óptima)

---

## 1. Looped/Recursive Transformers

### 1.1 Fundamento Teórico

Los transformers de profundidad fija son **NO Turing-completos** — solo pueden computar funciones en la clase TC^0 (circuitos de profundidad constante). Esto significa que no pueden realizar operaciones iteradas (multiplicación repetida, razonamiento multi-paso) en un solo forward pass.

**Looped Transformers son Turing-completos** (Giannou et al., ICML 2023). Un transformer looped con menos de 13 capas puede emular:
- Un computador de propósito general
- Álgebra lineal numérica (inversión de matrices, power iteration)
- Algoritmos de in-context learning (SGD sobre redes neuronales)

La profundidad no escala con la complejidad del programa — solo el número de iteraciones.

### 1.2 Samsung TRM: El Caso de Éxito

**Tiny Recursive Model (TRM):** 7M params, 2 capas únicas, dim=512

| Benchmark | TRM (7M) | DeepSeek R1 | o3-mini | Gemini 2.5 Pro |
|-----------|----------|-------------|---------|----------------|
| ARC-AGI-1 | **45%** | 34.3% | 42.1% | 38.7% |
| ARC-AGI-2 | **8%** | 2% | 4.2% | 3.5% |

Arquitectura detallada:
- **SwiGLU + RMSNorm + RoPE** (idéntico a nuestro modelo)
- **Think-Act cycle:** Genera "pensamientos latentes" antes de cada acción
- **16 iteraciones** del bloque de 2 capas (32 capas efectivas)
- **adaLN-Zero** para diferenciar iteraciones (en vez de step embeddings aditivos)

### 1.3 adaLN-Zero: Cómo Diferenciar Iteraciones

La técnica recomendada para informar a cada capa "en qué iteración estamos":

```python
# adaLN-Zero: Adaptive Layer Norm with Zero initialization
class AdaLNZero(nn.Module):
    def __init__(self, dim, n_loops):
        self.step_embed = nn.Embedding(n_loops, dim * 6)  # 6 params: scale, shift, gate x 2
        nn.init.zeros_(self.step_embed.weight)  # Zero init = identity at start

    def forward(self, x, step_idx):
        params = self.step_embed(step_idx)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = params.chunk(6, dim=-1)
        # Attn: h = alpha1 * attn(gamma1 * norm(x) + beta1)
        # FFN:  h = alpha2 * ffn(gamma2 * norm(x) + beta2)
```

**Ventajas sobre step embedding aditivo:**
- **Zero-init** = modelo se comporta como no-looped al inicio del training
- **Modulación multiplicativa** = más expresivo que suma
- Probado extensivamente en DiT (Diffusion Transformers) con excelentes resultados

### 1.4 Entrenamiento de Modelos Looped

**Full BPTT es esencial** — no usar implicit differentiation ni detach:
- Gradientes deben fluir a través de TODAS las iteraciones
- Deep supervision: computar loss en iteraciones intermedias (no solo la final)
- Esto previene vanishing gradients y mejora convergencia

**Curriculum de iteraciones:**
1. Iniciar con pocas iteraciones (2-4)
2. Gradualmente incrementar hasta el máximo (8-16)
3. Permite que el modelo aprenda representaciones estables primero

**Estabilidad:**
- Gradient clipping agresivo (0.5-1.0) es crucial
- RMSNorm entre iteraciones previene explosión de activaciones
- Learning rate más bajo que modelos estándar (~50-70% del normal)

### 1.5 Halting Adaptativo: PonderNet > ACT

**ACT (Graves 2016):** Notoriamente inestable. El hiperparámetro τ es extremadamente sensible.

**PonderNet (Banino 2021):** Reformulación estable con prior geométrico:
- Halting como variable Bernoulli con regularización KL
- λ_p robusto: sesga el número esperado de pasos hacia 1/λ_p
- **Near-perfect accuracy en tareas de extrapolación donde ACT falla completamente**
- Gradientes no-sesgados y baja varianza

**Recomendación:** Usar PonderNet para el halting, no ACT.

### 1.6 LoopViT y COCONUT

**LoopViT (18M params):** 65.8% en ARC-AGI-1 (solo visión, sin LM)
- Confirma que looping funciona a escala tiny para razonamiento

**COCONUT (Chain of Continuous Thought):**
- Razonamiento latente en espacio continuo (no tokens explícitos)
- Más poderoso que CoT para modelos pequeños
- Los "pensamientos" son hidden states, no texto — modelo no pierde contexto

### Papers Clave
- [Samsung TRM](https://arxiv.org/abs/2502.17416) - 7M, ARC-AGI
- [Looped Transformers as Programmable Computers](https://arxiv.org/abs/2301.13196) - Turing completeness
- [Universal Transformer](https://arxiv.org/abs/1807.03819) - ACT + sharing
- [Relaxed Recursive Transformers (DeepMind)](https://arxiv.org/abs/2410.20672) - LoRA per iteration
- [PonderNet](https://arxiv.org/abs/2107.05407) - Stable halting
- [COCONUT](https://arxiv.org/abs/2412.06769) - Continuous thought

---

## 2. Resonance Buffers y Feedback Transformer

### 2.1 Resonance Buffers: Nuestra Idea Original (Validada)

**Concepto:** Buffers persistentes de tamaño fijo que cada capa lee y escribe, con decaimiento exponencial. Proporcionan un canal de comunicación inter-capa **separado** del residual stream.

**Relación con literatura existente:**

| Mecanismo | Tipo | Overhead | Cross-layer? |
|-----------|------|----------|-------------|
| Attention sinks (emergente) | Working memory implícita | 0 | No |
| Register tokens (Darcet, ICLR 2024) | Working memory explícita | ~1K params | No (per-layer) |
| DenseFormer DWA (NeurIPS 2024) | Weighted skip connections | ~90 scalars | Sí |
| Persistent Memory (Sukhbaatar 2019) | Vectores aprendidos fijos | dim * n | No |
| **Resonance Buffers (nuestra idea)** | Memoria dinámica cross-layer | ~100-666K | **Sí** |

**Validación teórica:**
- **Gradient highway:** Con γ=0.9 y 8 capas: 0.9^8 = 0.43 de fuerza de gradiente — mucho mejor que vanilla deep networks
- **Comparación con Highway Networks:** Similar a carry gate, pero canal separado del residual
- **Comparación con DenseNet:** Similar efecto pero sin concatenaciones crecientes
- **NTMs/DNCs:** Mismo espíritu (memoria externa diferenciable) pero sin overhead de content-based addressing

**Qué pueden resolver que standard transformers NO:**
1. **Aritmética multi-paso:** Acumular resultados intermedios entre capas
2. **Generalización composicional:** Capas bajas escriben features, capas altas leen composiciones
3. **Contexto global:** A diferencia de attention (per-layer), resonance mantiene resumen running

### 2.2 Arquitectura Concreta de Resonance

```python
class ResonanceBuffer(nn.Module):
    """Memoria cross-layer con decaimiento exponencial."""

    def __init__(self, n_buffers=4, dim=288, rank=32):
        super().__init__()
        self.n_buffers = n_buffers
        self.dim = dim

        # Estado inicial aprendible
        self.init_state = nn.Parameter(torch.zeros(n_buffers, dim))

        # Decay rates (aprendibles, inicializados a 0.9)
        self.gamma_logit = nn.Parameter(torch.full((n_buffers,), 2.2))  # sigmoid(2.2) ≈ 0.9

        # Low-rank projections para mantener overhead bajo
        self.write_down = nn.Linear(dim, rank, bias=False)
        self.write_up = nn.Linear(rank, n_buffers * dim, bias=False)
        self.write_gate = nn.Linear(dim, n_buffers)

        self.read_down = nn.Linear(n_buffers * dim, rank, bias=False)
        self.read_up = nn.Linear(rank, dim, bias=False)

    def reset(self, batch_size, device):
        """Inicializar estado para nuevo batch."""
        self.state = self.init_state.unsqueeze(0).expand(batch_size, -1, -1).clone()

    def step(self, h_layer):
        """Llamar después de cada capa. h_layer: [B, T, dim]"""
        B = h_layer.shape[0]
        gamma = torch.sigmoid(self.gamma_logit)  # [n_buffers]

        # Decay
        self.state = gamma.unsqueeze(0).unsqueeze(-1) * self.state

        # Gated write (usa mean pooling sobre secuencia)
        h_mean = h_layer.mean(dim=1)  # [B, dim]
        gate = torch.sigmoid(self.write_gate(h_mean))  # [B, n_buffers]
        write_val = self.write_up(self.write_down(h_mean))  # [B, n_buffers * dim]
        write_val = write_val.view(B, self.n_buffers, self.dim)
        self.state = self.state + gate.unsqueeze(-1) * write_val

        # Read
        read_val = self.read_up(self.read_down(self.state.view(B, -1)))  # [B, dim]
        return read_val.unsqueeze(1)  # [B, 1, dim] — broadcast over sequence

    # Param count: rank=32, dim=288, n_buffers=4
    # write: 288*32 + 32*1152 + 288*4 = 9216 + 36864 + 1152 = ~47K
    # read: 1152*32 + 32*288 = 36864 + 9216 = ~46K
    # Total: ~94K params (0.9% overhead en 10M model)
```

### 2.3 DenseFormer: Free Lunch

**DenseFormer DWA (Depth Weighted Averaging)** — el mecanismo más simple con mayor impacto:
- Solo ~90 scalars extra
- Cada capa recibe una combinación ponderada de TODAS las capas anteriores
- Implementación trivial: `h = sum(w_i * h_i for i in range(layer_idx))`

**Recommendation:** Implementar DenseFormer DWA primero (trivial), luego resonance buffers.

### 2.4 Register Tokens para LLMs

De "Vision Transformers Need Registers" (Darcet, ICLR 2024):
- 4 tokens learnable al inicio de la secuencia
- Participan en attention en todas las capas
- Se descartan del output
- Overhead: 4 × dim = 4 × 288 = **1,152 params**

Para LLM causal: colocar registers **antes de BOS** para que todos los tokens puedan atender a ellos. Reemplaza el fenómeno de attention sinks (modelo ya no necesita "desperdiciar" attention en tokens iniciales).

### 2.5 Feedback Transformer Modernizado

**TransformerFAM (2024):** La versión moderna del Feedback Transformer:
- FAM tokens como tokens adicionales en la secuencia
- No requiere pesos adicionales — usa attention estándar
- FAM del output de la última capa se retroalimenta como input
- Complejidad lineal
- Probado a 1B, 8B, 24B

**Gated Feedback (nuestra extensión):**
```python
# Multi-level: last layer → layer 1, middle layer → layer 1
feedback_last = None
feedback_mid = None

for i, layer in enumerate(layers):
    if i == 0 and feedback_last is not None:
        gate_last = sigmoid(self.feedback_gate_last(h))
        gate_mid = sigmoid(self.feedback_gate_mid(h))
        h = h + gate_last * self.feedback_proj_last(feedback_last)
        h = h + gate_mid * self.feedback_proj_mid(feedback_mid)
    h, cache = layer(h, ...)
    if i == len(layers) // 2:
        feedback_mid = h.detach()  # or no detach for full gradient flow
    if i == len(layers) - 1:
        feedback_last = h
```

Overhead: 2 × (dim² + dim) = 2 × (288² + 288) = **~166K params** (1.7%)

### 2.6 Value Residual Learning

De ResFormer/SVFormer (2024) — **0 params extra:**
- Problema: over-smoothing en transformers profundos (tokens pierden información)
- Solución: residual connections para **value states** (no solo hidden states)
- SVFormer: todas las capas comparten value state de la primera capa
- Previene pérdida de diversidad representacional

### 2.7 Ranking de Mecanismos (para 10M model)

| Rank | Mecanismo | Overhead | Impacto | Complejidad |
|------|-----------|----------|---------|-------------|
| 1 | DenseFormer DWA | ~90 scalars | Alto (free lunch) | Muy Baja |
| 2 | Register Tokens (4) | ~1K params | Medio-Alto | Baja |
| 3 | Value Residual Learning | 0 params | Medio | Muy Baja |
| 4 | Resonance Buffers (rank 32) | ~94K params | Medio-Alto | Media |
| 5 | Feedback + Looping | ~166K params | Alto (con looping) | Media |
| 6 | Memorizing Transformer kNN | ~few scalars | Alto (memoria externa) | Media-Alta |

### Papers Clave
- [DenseFormer (NeurIPS 2024)](https://arxiv.org/abs/2402.02622)
- [Vision Transformers Need Registers (ICLR 2024)](https://arxiv.org/abs/2309.16588)
- [TransformerFAM (2024)](https://arxiv.org/abs/2404.09173)
- [Feedback Transformer (2020)](https://arxiv.org/abs/2002.09402)
- [Memorizing Transformers (ICLR 2022)](https://arxiv.org/abs/2203.08913)
- [Recurrent Memory Transformer (NeurIPS 2022)](https://arxiv.org/abs/2207.06881)
- [Value Residual Learning (2024)](https://arxiv.org/html/2410.17897v5)
- [Compressive Transformer (ICLR 2020)](https://arxiv.org/abs/1911.05507)
- [Talking Heads (NeurIPS 2024)](https://arxiv.org/abs/2406.09519)

---

## 3. Computación Adaptativa

### 3.1 Mixture of Depths (MoD): Detalles Profundos

**Mecanismo exacto:** Router `nn.Linear(dim, 1)` produce peso escalar por token. Los top-k tokens (k = C × S, donde C es capacity ratio) pasan por el bloque completo. Resto: residual skip.

**Resultados cuantitativos:**
- C=0.125 (12.5%): solo 1 de 8 tokens procesados por capa MoD
- **50% menos FLOPs** con **<1% degradación** en perplexity
- En isoFLOP: modelos MoD óptimos son más grandes (más params) pero más rápidos

**Variante MoDification (Oct 2024):** Reemplaza top-k con threshold-p — cada token decide independientemente (causal-friendly). No necesita ver toda la secuencia.

**A-MoD (Dec 2024):** Elimina el router lineal — usa attention map de la capa anterior como routing signal. 0 params extra, convergencia 2x más rápida en transfer learning.

**A 60M (la escala más pequeña probada):** Beneficios visibles pero menores que a gran escala. Esperar ~30-40% reducción de FLOPs con <1% degradación.

### 3.2 ACT/PonderNet para Profundidad Variable

**ACT (Graves 2016):**
```
h_n = sigmoid(W_h · s_n + b_h)         # halting probability
N(x) = min{n' : Σh_n ≥ 1 - ε}         # número de pasos
R = 1 - Σh_n                            # remainder
y = Σ h_n · y_n + R · y_N               # output pesado
Ponder cost = N(x) + R(x)               # regularización
```

**Problema:** τ extremadamente sensible, alta varianza entre seeds.

**PonderNet (mejor alternativa):**
- Halting como Bernoulli con prior geométrico
- KL divergence contra prior reemplaza ponder cost
- λ_p robusto: número esperado de pasos ≈ 1/λ_p
- **Gradientes no-sesgados** (ACT tiene gradientes sesgados)

### 3.3 Early Exit: CALM y LayerSkip

**CALM (NeurIPS 2022):** Early exit para modelos autoregresivos
- **3x speedup** manteniendo calidad
- 3 medidas de confianza: softmax response, state propagation, learned classifier
- **KV Cache propagation:** Cuando token sale en capa 5, sus KV se copian a capas 6-8
- Garantías estadísticas de calidad secuencia-completa

**LayerSkip (Meta, ACL 2024):**
- Layer dropout durante training: rates bajos para capas early, altos para capas late
- Loss en TODAS las capas (ensemble of models at different depths)
- Self-speculative decoding: capas early = draft model, capas late = verifier
- **1.82-2.16x speedup**

**Dynamic Layer Selection (NeurIPS 2024):**
- Layer skipping >> early exit para decoder-only
- Oracle: solo **23.3% de capas** necesarias en promedio
- Per-sequence allocation > per-token allocation (con métodos actuales)

### 3.4 Nuestra Idea: Attention Budget

**Lo que la hace novel:**
1. **Asignación upfront** — ACT decide paso-a-paso; MoD usa competencia per-capa. Budget predice el compute total a embedding time.
2. **Metáfora económica** — presupuesto que se "gasta" capa por capa
3. **Autonomía por token** — no compite contra otros tokens (a diferencia de MoD top-k)

**Arquitectura propuesta:**
```python
class BudgetPredictor(nn.Module):
    def __init__(self, dim, max_layers):
        self.proj = nn.Linear(dim, 1, bias=True)
        self.max_layers = max_layers

    def forward(self, x):
        return torch.sigmoid(self.proj(x)) * self.max_layers

# En forward:
budget = self.budget_predictor(h)  # [B, T, 1]
for i, layer in enumerate(self.layers):
    remaining = torch.clamp(budget - i, 0, 1)  # soft mask [0,1]
    layer_out, cache = layer(h, freqs_cis, mask, cache)
    h = remaining * layer_out + (1 - remaining) * h  # soft skip
```

**Diseño clave:**
- **Soft masking** durante training (diferenciable), hard threshold en inferencia
- **PonderNet-style KL** contra prior geométrico como regularización
- **Alternar routed/unrouted layers** (MoD finding: every-other es más estable)
- **KV propagation** (CALM-style) para tokens que salen temprano

### 3.5 WARNING: Capacity Bottleneck a Escala Tiny

**Mixture-of-Recursions (NeurIPS 2025):** A 135M params, MoR **underperforms vanilla** por capacity bottleneck. A 360M+ funciona bien.

**Implicación para 10-30M:**
- NO combinar weight sharing + adaptive depth a esta escala
- Usar capas independientes con budget mechanism
- Validar primero en 30M (12 capas dan más granularidad)
- Con 8 capas, cada capa = 12.5% del modelo — granularidad de routing es gruesa

### Papers Clave
- [Mixture-of-Depths (2024)](https://arxiv.org/abs/2404.02258)
- [MoDification (2024)](https://arxiv.org/abs/2410.14268)
- [ACT (Graves 2016)](https://arxiv.org/abs/1603.08983)
- [PonderNet (2021)](https://arxiv.org/abs/2107.05407)
- [CALM (NeurIPS 2022)](https://arxiv.org/abs/2207.07061)
- [LayerSkip (Meta, ACL 2024)](https://arxiv.org/abs/2404.16710)
- [Mixture-of-Recursions (NeurIPS 2025)](https://arxiv.org/abs/2507.10524)
- [Dynamic Layer Selection (NeurIPS 2024)](https://arxiv.org/abs/2410.20022)

---

## 4. Dual Brain: Test-Time Compute Scaling

### 4.1 El Principio Fundamental

Si un solo sample tiene probabilidad p de ser correcto:

**P(al menos uno correcto en N samples) = 1 - (1-p)^N**

Para nuestro 10M model con p=0.15:

| N (candidatos) | P(≥1 correcto) | Mejora |
|----------------|-----------------|--------|
| 1 | 15.0% | 1.0x |
| 4 | 47.8% | 3.2x |
| 8 | 72.7% | 4.8x |
| 16 | 89.9% | 6.0x |
| 64 | 99.1% | 6.6x |

**Resultado clave:** Un LLM de 0.5B con test-time compute supera a GPT-4o en MATH-500. Un 3B supera a un 405B (135x ratio). El principio aplica en general.

### 4.2 TinyGSM: 81.5% en GSM8K con Modelos Pequeños

**Insight principal:** El bottleneck es calidad de datos, NO tamaño del modelo.

- 12.3M problemas de matemáticas generados por GPT-3.5-turbo
- Soluciones en **Python** (no lenguaje natural) — permite verificación automática
- Generator (1.3B) + Verifier (1.3B) = 81.5% en GSM8K
- **125M + verifier = 63.1%** (¡enorme boost de verificación!)

### 4.3 PRMs vs ORMs

- **ORM:** Evalúa solo la respuesta final (correcto/incorrecto)
- **PRM:** Evalúa cada paso intermedio de razonamiento

**PRMs superan ORMs por ~4% accuracy** con datos limitados. Pero PRMs requieren modelos de 1-7B mínimo. Para 10M: usar ORM o code execution.

**Math-Shepherd** (automatizar labels PRM sin humanos):
1. Tomar problema + solución parcial hasta paso k
2. Samplear muchas completions desde paso k
3. Ejecutar para verificar correctness
4. Fracción que llega a respuesta correcta = quality score del paso k
5. 86% accuracy en labels automáticos

### 4.4 Majority Voting / Self-Consistency

Self-Consistency (Wang 2022): Samplear K reasoning paths, extraer respuesta final, majority vote.

| Benchmark | CoT (greedy) | Self-Consistency | Mejora |
|-----------|-------------|------------------|--------|
| GSM8K | ~58% | ~76% | **+17.9%** |
| SVAMP | ~79% | ~90% | **+11.0%** |

**Weighted Majority Voting** siempre domina plain voting si el reward model es "better than random."

**Para 10M:** Solo funciona en tareas donde base accuracy > 20-30%. Abajo de eso, necesitas verificador.

### 4.5 Ejecución de Código: El Verificador Perfecto (GRATIS)

**La recomendación más importante para nuestro proyecto:**

1. Entrenar generator para producir soluciones en Python
2. Generar N=16-64 candidatos
3. **Ejecutar** todos los candidatos en sandbox
4. Filtrar por los que ejecutan sin errores
5. Majority vote entre ejecuciones válidas

**100% accurate verification, zero params, zero training.** Esto es lo que TinyGSM, MathCoder y rStar-Math aprovechan.

### 4.6 DPO/SPIN Self-Improvement Loop

**Workflow para 10M model:**

1. Generar 16 soluciones por problema a T=0.8
2. Evaluar correctness (code execution o ground truth)
3. Formar pares (correct, incorrect) como preference pairs
4. Entrenar con DPO loss: `L = -log(σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x))))`
5. Repetir 2-3 iteraciones (SPIN style)

**SPIN (Chen 2024):** Self-play donde el modelo juega contra versiones previas de sí mismo. Iteración 0 = DPO performance. Iteración 1 ya lo supera.

### 4.7 Estrategia por Tiers

| Tier | Estrategia | Params Extra | Ganancia Esperada |
|------|-----------|-------------|-------------------|
| 1 | Code Execution Verifier | 0 | 2-4x single sample |
| 2 | Majority Voting (N=16-32) | 0 | 1.5-3x |
| 3 | DPO/SPIN self-improvement | 0 (mismo modelo) | +5-15% base accuracy |
| 4 | ORM separado (10M) | 10M | 1.5-2x adicional |
| 5 | GenRM (self-verification) | 0 | Incierto a 10M |

### 4.8 Temperatura y Sampling Óptimo

**Multi-temperature strategy (recomendada):**
- N/3 candidatos a T=0.6 (confident)
- N/3 a T=0.8 (balanced)
- N/3 a T=1.0 (exploratory)

Esto captura tanto soluciones seguras como creativas.

**Latencia para 10M model (256 tokens):**
- N=1: ~50-100ms GPU
- N=16: ~200-400ms GPU
- N=64: ~500ms-1s GPU

Un modelo de 10M es tan pequeño que test-time compute es **casi gratis**.

### 4.9 Compute-Optimal Allocation

De Snell et al. (2024) — la dificultad determina la estrategia óptima:

| Dificultad | Estrategia Óptima |
|------------|-------------------|
| Fácil (>60% accuracy) | Poco test-time compute. 1-2 samples bastan. |
| Media (10-60%) | **Sweet spot.** N=16-64 da mayores gains. |
| Difícil (<5%) | Modelo más grande es mejor. Test-time compute no ayuda. |

**Implicación:** Identificar tareas donde 10M tiene 10-40% base accuracy y concentrar test-time compute ahí.

### Papers Clave
- [TinyGSM (2023)](https://arxiv.org/abs/2312.09241)
- [Let's Verify Step by Step (OpenAI, 2023)](https://arxiv.org/abs/2305.20050)
- [Math-Shepherd (2023)](https://arxiv.org/abs/2312.08935)
- [Self-Consistency (2022)](https://arxiv.org/abs/2203.11171)
- [Test-Time Compute Scaling (2024)](https://arxiv.org/abs/2408.03314)
- [Inference Scaling Laws (2024)](https://arxiv.org/abs/2408.00724)
- [GenRM (2024)](https://arxiv.org/abs/2408.15240)
- [Weaver (Stanford, 2025)](https://hazyresearch.stanford.edu/blog/2025-06-18-weaver)
- [DPO (2023)](https://arxiv.org/abs/2305.18290)
- [SPIN (2024)](https://arxiv.org/abs/2401.01335)
- [rStar-Math (2025)](https://arxiv.org/abs/2501.04519)

---

## 5. BitNet y Arquitecturas Ternarias

### 5.1 BitNet b1.58 a Escala Tiny

**BitNet b1.58 Reloaded** (Schneider-Kamp, 2024) — el estudio definitivo para escala tiny:

- Testado desde **100K hasta 48M params** en vision y language
- **100K-2.2M (vision):** 1.58-bit iguala o supera state-of-the-art 16-bit
- **6M-48M (language):** Competitivo cuando hidden dimension se duplica
- **Regla crucial: 2x el hidden dimension** para compensar weights ternarios

### 5.2 Training Recipe

De BitNet b1.58 2B4T Technical Report:

| Paso | Detalle |
|------|---------|
| Warmup | ~2K steps en FP16, luego switch abrupto a 1.58-bit |
| Optimizer | Adam, betas=(0.9, 0.95) |
| LR | Two-stage: cosine decay alto, luego cooldown bajo |
| Weight decay | Aplicar a latent weights; **remover en segunda mitad** |
| Gradual quantization | No beneficia — switch abrupto funciona igual o mejor |
| Fresh optimizer al switch | Catches up rápidamente |

**Quantización absmean:**
```python
alpha = mean(|W|)
W_q = round(clip(W / alpha, -1, 1))  # produce {-1, 0, +1}
# Dequantized: W_q * alpha
```

**Variante median:** Más robusta a outliers en tiny scale.

### 5.3 Cambios Arquitecturales

**ReLU-squared reemplaza SwiGLU:**
```python
# SwiGLU: 3 matrices (gate, up, down)
# ReLU²: 2 matrices (up, down) + ReLU(x)²
# Beneficios: ~50% zero activations (alinea con sparsity ternaria), menos params
```

**SubLN (RMSNorm extra):** Insertar RMSNorm **antes** de cada capa linear cuantizada. Estabiliza entrenamiento previniendo explosión de activaciones.

**STE (Straight-Through Estimator):** Forward = quantized weights. Backward = gradientes fluyen como si no hubiera quantización.

### 5.4 Análisis de Memoria

Para nuestro 10M model (dim=288):

| Config | Params Únicos | Depth | Memoria Pesos |
|--------|--------------|-------|---------------|
| Float16, 8 layers | 10M | 8 | ~20MB |
| Ternary, 8 layers | 10M | 8 | **~2MB** |
| Ternary, dim=576, 8 layers | 40M | 8 | **~8MB** |
| Ternary, dim=576, 4 layers × 4 loops | 40M logical | 16 | **~8MB** |

**El sweet spot: dim=576, 4 capas únicas looped 4 veces, ternario — 40M logical params, 16 capas efectivas, en 8MB.**

### 5.5 Inference Speed

**bitnet.cpp (ACL 2025):**

| Plataforma | Speedup vs FP16 | Reducción Energía |
|------------|-----------------|-------------------|
| ARM CPUs | 1.37-5.07x | 55-70% |
| x86 CPUs | 2.37-6.17x | 72-82% |
| GPU (TriRun) | hasta 8x | - |

### 5.6 Hybrid: BitNet + Looped Transformer

**Nadie ha probado esta combinación exacta.** Pero la convergencia de ternary weights + sparsity + weight sharing es la trayectoria inevitable según ["All LLMs Will Be Sparse BitNet Hybrids"](https://huggingface.co/blog/codys12/rl-2025).

Nuestro `block_sharing` ya es el primer paso. Extender a full looping con ternary weights sería novel.

### 5.7 Hybrid: BitNet + MoE (MoTE)

**MoTE (Mixture of Ternary Experts, 2025):**
- FFN pre-entrenado como shared expert (full precision)
- Ternary routed experts adicionales {-1, 0, +1}
- Mismo memory footprint: +4.3% accuracy vs MoE full precision

Para 10M model:
- Shared attention (ternary): ~3M params
- 8 ternary FFN experts, top-2 routing: ~7M params total
- Memoria: ~2MB
- Compute activo por token: ~5M params

### 5.8 ¿Es BitNet "Free Lunch" a 10M?

**NO.** Nuanced:

| A favor | En contra |
|---------|-----------|
| 10x menos memoria inference | A params iguales, FP16 es mejor calidad |
| Con 2x width, más logical params | Training sigue usando FP32 shadow weights |
| Funciona desde 100K (vision) | Gap en quality a 99-300M en language |
| bitnet.cpp da speedup real | GPU kernels ternarios aún inmaduros |

**Recomendación por deployment target:**
- **Edge/mobile:** BitNet compelling. 40M ternary en 8MB > 10M float en 20MB
- **Calidad máxima:** Float16. Quality-per-parameter es mayor
- **Hybrid approach:** Train float16, luego fine-tune a 1.58-bit ([HuggingFace guide](https://huggingface.co/blog/1_58_llm_extreme_quantization))

**Ternary models son más data-hungry:** Spectra 1.1 muestra que TriLMs benefician más de datos adicionales que de más params. Para 10M ternario, apuntar a 100x+ token-to-parameter ratio (vs 20-50x normal).

### Papers Clave
- [BitNet b1.58 Reloaded (2024)](https://arxiv.org/abs/2407.09527) - Tiny scale
- [The Era of 1-bit LLMs (2024)](https://arxiv.org/abs/2402.17764) - Original
- [BitNet 2B4T Technical Report (2025)](https://arxiv.org/abs/2504.12285)
- [Spectra: Ternary LMs (2024)](https://arxiv.org/abs/2407.12327)
- [MatMul-free Language Modeling (2024)](https://arxiv.org/abs/2406.02528)
- [Continual QAT (ACL 2025)](https://arxiv.org/abs/2502.11895)
- [MoTE (2025)](https://arxiv.org/abs/2506.14435)
- [bitnet.cpp (ACL 2025)](https://arxiv.org/abs/2502.11880)

---

## 6. Síntesis: Arquitectura Óptima Propuesta

### 6.1 Fase Inmediata (Zero-Cost Improvements)

Aplicar a los modelos actuales sin cambiar la arquitectura base:

1. **DenseFormer DWA** — ~90 scalars, implementación trivial
2. **Value Residual Learning** — 0 params, conexiones residuales para value states
3. **4 Register Tokens** — ~1K params, al inicio de cada secuencia

### 6.2 Fase Experimental (Modificaciones al 10M/30M)

Probar una-por-una para medir impacto aislado:

4. **Resonance Buffers** (4 buffers, rank 32) — ~94K params
5. **Feedback: last layer → layer 1** con gate — ~83K params
6. **Attention Budget** con soft masking — ~300 params + regularización

### 6.3 Fase Avanzada (Nuevas Arquitecturas)

Implementar como modelos separados:

7. **Looped Transformer** (4 layers × 8 loops, adaLN-Zero, PonderNet halting)
8. **BitNet 1.58** (2x width, ReLU², SubLN, 2K FP16 warmup)
9. **Looped + Ternary Hybrid** (4 layers × 4 loops, dim=576, ternario)

### 6.4 Fase Alignment/Inference

Post-training para mejorar calidad:

10. **Code Execution Verifier** para math tasks
11. **DPO/SPIN self-improvement** (2-3 iteraciones)
12. **Best-of-N** con majority voting (N=16-64)

### 6.5 La Arquitectura "Dream"

Si todo funciona según lo predicho, el modelo ideal:

```yaml
# "Resonant Looped Ternary Transformer" — El Sueño
model:
  dim: 576              # 2x para compensar ternario
  n_layers: 4           # Solo 4 capas únicas
  n_loops: 4            # 4 iteraciones = 16 capas efectivas
  n_heads: 8
  n_kv_heads: 2
  vocab_size: 8192
  hidden_dim: 1536
  max_seq_len: 512

  # Innovaciones
  quantization: "ternary_1.58bit"  # {-1, 0, +1}
  activation: "relu_squared"        # Reemplaza SwiGLU
  step_embedding: "adaln_zero"      # Diferencia iteraciones
  halting: "pondernet"              # Profundidad variable per-token
  resonance_buffers: 4              # Memoria cross-layer
  register_tokens: 4                # Working memory in-sequence
  dense_former_dwa: true            # Weighted averaging
  feedback: "last_to_first"         # Top-down context

  # Estimates
  logical_params: ~40M    # Por weight sharing
  unique_params: ~10M     # Realmente almacenados
  memory_weights: ~8MB    # Ternario
  effective_depth: 16     # 4 × 4 loops

inference:
  best_of_n: 16
  code_verification: true
  multi_temperature: [0.6, 0.8, 1.0]
```

**Memoria total:** ~8MB (vs 20MB float16 de 10M actual)
**Profundidad efectiva:** 16 capas (vs 8 actuales)
**Params lógicos:** ~40M (vs 10M actuales)
**Con test-time compute:** Hasta 6x mejor accuracy en sweet-spot tasks

---

## Apéndice: Papers Citados (por Categoría)

### Looped/Recursive Transformers
1. [Samsung TRM (2025)](https://arxiv.org/abs/2502.17416)
2. [Looped Transformers as Programmable Computers (ICML 2023)](https://arxiv.org/abs/2301.13196)
3. [Universal Transformer (ICLR 2019)](https://arxiv.org/abs/1807.03819)
4. [Relaxed Recursive Transformers (DeepMind, 2024)](https://arxiv.org/abs/2410.20672)
5. [PonderNet (2021)](https://arxiv.org/abs/2107.05407)
6. [COCONUT (2024)](https://arxiv.org/abs/2412.06769)
7. [On Expressive Power of Looped Transformers (2024)](https://arxiv.org/abs/2410.01405)

### Resonance, Feedback, Memory
8. [DenseFormer (NeurIPS 2024)](https://arxiv.org/abs/2402.02622)
9. [Vision Transformers Need Registers (ICLR 2024)](https://arxiv.org/abs/2309.16588)
10. [TransformerFAM (2024)](https://arxiv.org/abs/2404.09173)
11. [Feedback Transformer (2020)](https://arxiv.org/abs/2002.09402)
12. [Memorizing Transformers (ICLR 2022)](https://arxiv.org/abs/2203.08913)
13. [Recurrent Memory Transformer (NeurIPS 2022)](https://arxiv.org/abs/2207.06881)
14. [Value Residual Learning (2024)](https://arxiv.org/html/2410.17897v5)
15. [Compressive Transformer (ICLR 2020)](https://arxiv.org/abs/1911.05507)
16. [Persistent Memory (2019)](https://arxiv.org/abs/1907.01470)
17. [Talking Heads (NeurIPS 2024)](https://arxiv.org/abs/2406.09519)
18. [Evolved Universal Transformer Memory (NeurIPS 2024)](https://arxiv.org/abs/2410.13166)
19. [Block-Recurrent Transformers (NeurIPS 2022)](https://arxiv.org/abs/2203.07852)
20. [Gated Feedback RNNs (ICML 2015)](https://arxiv.org/abs/1502.02367)

### Computación Adaptativa
21. [Mixture-of-Depths (2024)](https://arxiv.org/abs/2404.02258)
22. [MoDification (2024)](https://arxiv.org/abs/2410.14268)
23. [A-MoD (2024)](https://arxiv.org/abs/2412.20875)
24. [ACT (Graves, 2016)](https://arxiv.org/abs/1603.08983)
25. [CALM (NeurIPS 2022)](https://arxiv.org/abs/2207.07061)
26. [LayerSkip (Meta, ACL 2024)](https://arxiv.org/abs/2404.16710)
27. [Mixture-of-Recursions (NeurIPS 2025)](https://arxiv.org/abs/2507.10524)
28. [Dynamic Layer Selection (NeurIPS 2024)](https://arxiv.org/abs/2410.20022)
29. [Sparse Universal Transformer (EMNLP 2023)](https://arxiv.org/abs/2310.07096)
30. [SkipNet (ECCV 2018)](https://arxiv.org/abs/1711.09485)

### Test-Time Compute & Verification
31. [TinyGSM (2023)](https://arxiv.org/abs/2312.09241)
32. [Let's Verify Step by Step (OpenAI, 2023)](https://arxiv.org/abs/2305.20050)
33. [Math-Shepherd (2023)](https://arxiv.org/abs/2312.08935)
34. [Self-Consistency (2022)](https://arxiv.org/abs/2203.11171)
35. [Test-Time Compute Scaling (2024)](https://arxiv.org/abs/2408.03314)
36. [Inference Scaling Laws (2024)](https://arxiv.org/abs/2408.00724)
37. [GenRM (2024)](https://arxiv.org/abs/2408.15240)
38. [Weaver (Stanford, 2025)](https://hazyresearch.stanford.edu/blog/2025-06-18-weaver)
39. [DPO (2023)](https://arxiv.org/abs/2305.18290)
40. [SPIN (2024)](https://arxiv.org/abs/2401.01335)
41. [rStar-Math (2025)](https://arxiv.org/abs/2501.04519)
42. [Training Verifiers to Solve Math (2021)](https://arxiv.org/abs/2110.14168)

### BitNet & Quantization
43. [BitNet b1.58 Reloaded (2024)](https://arxiv.org/abs/2407.09527)
44. [The Era of 1-bit LLMs (2024)](https://arxiv.org/abs/2402.17764)
45. [BitNet 2B4T (2025)](https://arxiv.org/abs/2504.12285)
46. [Spectra (2024)](https://arxiv.org/abs/2407.12327)
47. [MatMul-free LM (2024)](https://arxiv.org/abs/2406.02528)
48. [Continual QAT (ACL 2025)](https://arxiv.org/abs/2502.11895)
49. [MoTE (2025)](https://arxiv.org/abs/2506.14435)
50. [bitnet.cpp (ACL 2025)](https://arxiv.org/abs/2502.11880)
51. [Extra RMSNorm for 1.58-bit (2025)](https://arxiv.org/abs/2505.08823)
52. [StableQAT (2025)](https://arxiv.org/abs/2601.19320)
53. [LLM.int8() (2022)](https://arxiv.org/abs/2208.07339)

### Small Model Design
54. [MobileLLM (ICML 2024)](https://arxiv.org/abs/2402.14905)
55. [Highway Networks (2015)](https://arxiv.org/abs/1505.00387)
56. [StreamingLLM (ICLR 2024)](https://arxiv.org/abs/2309.17453)
57. [Neural Turing Machines (2014)](https://arxiv.org/abs/1410.5401)
