# Arquitectura y Compresión para Modelos Tiny: Guía de Mejoras

> Investigación para lograr coherencia y razonamiento en modelos de 10M-110M parámetros.
> Proyecto: brandon-tiny | GPU: RTX 3090 24GB | Modelos: Llama 2 style (RoPE, RMSNorm, SwiGLU, GQA)

---

## Estado Actual del Proyecto

| Modelo | Params | Pretrain val_loss | Finetune val_loss | Datos |
|--------|--------|-------------------|-------------------|-------|
| 226K v2 | 234K | 4.56 (distill) | 6.06 | TinyStories (teacher: 10M) |
| 10M v2 | 10.5M | 1.87 | 3.76 | TinyStories 537M tok |
| 10M synthetic | 10.5M | 2.60 (continued) | 3.64 | + Synthetic 3.7M tok |
| 10M MTP | 10.9M | 2.94 | (running) | TinyStories, n_predict=4 |
| 30M v2 | 30M | ~3.33 (running) | - | SmolLM 966M tok |
| 110M | 110M | - | - | (planned) |

**Limitaciones observadas:**
- 10M genera texto semi-coherente pero repetitivo, matemáticas rotas
- 226K produce gibberish, demasiado pequeño para instruction following
- Los datos sintéticos mejoran conocimiento factual (+0.12 val_loss)
- MTP mejora representaciones internas (val_loss 2.94 vs 1.87 baseline)

---

## Tabla de Contenidos

1. [Arquitecturas Revolucionarias para Modelos Tiny](#1-arquitecturas-revolucionarias)
2. [Compresión de Tokens y Vocabulario](#2-compresión-de-tokens)
3. [Técnicas de Entrenamiento para Razonamiento](#3-entrenamiento-para-razonamiento)
4. [Compresión y Eficiencia de Pesos](#4-compresión-y-eficiencia)
5. [Ideas Experimentales / Locas](#5-ideas-experimentales)
6. [Roadmap de Implementación](#6-roadmap)

---

## 1. Arquitecturas Revolucionarias

### 1.1 Looped Transformers (Razonamiento Recursivo)

**Qué es:** En vez de N capas diferentes, usar 1-4 capas únicas repetidas L veces. Nuestro `block_sharing` ya hace algo similar (pares adyacentes comparten pesos), pero esto lo lleva al extremo.

**Por qué funciona para modelos tiny:**
- Muchos problemas de razonamiento requieren **profundidad** (muchos pasos), no muchos parámetros
- Un transformer de k capas looped L veces casi iguala uno de kL capas no-looped
- Genera "pensamientos latentes" internos en cada iteración (más poderoso que CoT)
- Compatible con **early exit adaptativo**: inputs simples terminan en pocas iteraciones, inputs complejos usan más

**El caso Samsung TRM (Tiny Recursive Model):**
- **7M parámetros, 2 capas** — podría correr en un smartwatch
- Logra **45% en ARC-AGI-1** y **8% en ARC-AGI-2**
- Supera a DeepSeek R1, o3-mini y Gemini 2.5 Pro en razonamiento abstracto
- Funciona por refinamiento iterativo: pregunta → guess inicial → refina latent → actualiza predicción → repite

**Implementación para nuestro proyecto:**

```yaml
# Config propuesto: Looped 10M
model:
  dim: 384          # Más ancho (era 256)
  n_layers: 4       # Solo 4 capas únicas
  n_loops: 8        # Repetir 8 veces = 32 capas efectivas
  n_heads: 8
  n_kv_heads: 2
  vocab_size: 8192
  hidden_dim: 1024
  adaptive_halt: true  # Early exit por token
```

**Cambios en `model.py`:**
- Agregar `n_loops` a ModelConfig
- Modificar forward para iterar sobre las mismas capas
- Agregar step embedding para diferenciar iteraciones
- Opcional: halting probability para exit adaptativo

**Dificultad:** Media | **Impacto estimado:** Alto | **Papers:** [arxiv 2502.17416](https://arxiv.org/abs/2502.17416), [arxiv 2510.04871](https://arxiv.org/abs/2510.04871)

---

### 1.2 Feedback Transformer (Top-Down Context)

**Qué es:** La salida de la última capa en el timestep anterior se retroalimenta a TODAS las capas en el timestep actual. La capa más baja puede "ver" la representación más abstracta.

**Por qué funciona para modelos tiny:**
- Un Feedback Transformer de **la mitad del tamaño** iguala a Transformer-XL en language modeling
- Cada capa tiene acceso a la representación más profunda, compensando tener pocas capas
- Especialmente beneficioso para modelos **pequeños y shallow**

**Implementación:**
```python
# En TinyLlama.forward():
feedback_state = None  # top-layer hidden state from previous step

for layer in self.layers:
    if feedback_state is not None:
        h = h + self.feedback_proj(feedback_state)  # inyectar contexto top-down
    h, cache = layer(h, freqs_cis, mask, cache)

feedback_state = h[:, -1:]  # guardar última posición de la capa final
```

**Dificultad:** Baja | **Impacto estimado:** Alto (2x eficiencia de tamaño) | **Paper:** [arxiv 2002.09402](https://arxiv.org/abs/2002.09402)

---

### 1.3 Mixture of Depths (Compute Variable por Token)

**Qué es:** Cada capa tiene un router que decide qué tokens procesan esa capa y cuáles la saltan (residual directo). Los tokens "fáciles" (artículos, preposiciones) se saltan; los "difíciles" (números, nombres propios, razonamiento) usan todas las capas.

**Resultados:**
- Probado desde **60M parámetros**
- Un modelo MoD de 220M iguala al baseline siendo **60% más rápido** por step
- Reduce FLOPs 50% sin perder calidad

**Implementación:**
```python
class MoDBlock(nn.Module):
    def __init__(self, config, capacity=0.5):
        self.router = nn.Linear(config.dim, 1)  # scalar per token
        self.block = TransformerBlock(config)
        self.capacity = capacity  # fracción de tokens que procesan

    def forward(self, x, freqs_cis, mask, cache):
        B, T, D = x.shape
        k = int(T * self.capacity)

        # Router decide qué tokens procesan
        weights = self.router(x).squeeze(-1)  # [B, T]
        topk_indices = weights.topk(k, dim=-1).indices
        topk_mask = torch.zeros_like(weights, dtype=torch.bool)
        topk_mask.scatter_(1, topk_indices, True)

        # Solo tokens seleccionados pasan por attention + MLP
        x_selected = x[topk_mask].view(B, k, D)
        x_processed, cache = self.block(x_selected, freqs_cis, mask, cache)

        # Merge: tokens no seleccionados mantienen su valor (residual skip)
        out = x.clone()
        out[topk_mask] = x_processed.view(-1, D) * torch.sigmoid(weights[topk_mask]).unsqueeze(-1)
        return out, cache
```

**Dificultad:** Media | **Impacto:** Medio-Alto | **Paper:** [arxiv 2404.02258](https://arxiv.org/abs/2404.02258)

---

### 1.4 Hourglass Transformer (Compresión Jerárquica)

**Qué es:** Estructura U-Net dentro del transformer: las primeras capas operan a resolución completa, luego se comprime la secuencia (pooling), las capas medias operan sobre la secuencia corta (barato), y al final se expande de vuelta.

**Para nuestro 30M (28 capas, 14 únicas):**
```
Capas 1-3:  resolución completa (512 tokens)
Pooling:    downsample 2x
Capas 4-11: media resolución (256 tokens) ← 4x menos FLOPs en attention
Upsample:   expand + skip connection
Capas 12-14: resolución completa (512 tokens)
```

**Beneficio:** Las 8 capas medias procesan secuencias de 256 tokens en vez de 512, reduciendo FLOPs de attention ~4x para esas capas. El overhead de pooling/upsampling es ~2-3% de parámetros.

**Dificultad:** Media-Alta | **Impacto:** Medio | **Paper:** [arxiv 2110.13711](https://arxiv.org/abs/2110.13711)

---

### 1.5 Universal Transformer + Adaptive Computation Time (ACT)

**Qué es:** Un solo bloque transformer aplicado T veces con pesos compartidos. Cada token tiene una "halting probability" que decide cuándo parar de computar. Tokens fáciles paran en 2-3 iteraciones; tokens difíciles usan todas las T iteraciones.

**Por qué es perfecto para nuestro proyecto:**
- Ya tenemos `block_sharing = True` — esto es la extensión natural
- Nuestro 10M con 24 capas (12 únicas) podría ser 4 capas únicas × 8 loops
- Los params ahorrados se reinvierten en dim/hidden_dim más grandes
- ACT da profundidad variable **por token** — ideal para razonamiento

**Comparación de eficiencia:**
```
Actual 10M v2:     12 bloques únicos × dim=256 = 10.5M params, depth=24
Universal 10M:      4 bloques únicos × dim=384 = ~10M params, depth=32 (variable)
```
Mismos params, más profundidad efectiva, dim más ancho para mejores representaciones.

**Dificultad:** Media | **Impacto:** Alto | **Paper:** [arxiv 1807.03819](https://arxiv.org/abs/1807.03819)

---

## 2. Compresión de Tokens

### 2.1 Vocabulario Óptimo para Modelos Tiny

**Hallazgo clave:** Nuestro vocab de 8K es bueno. SmolLM2-135M usa 49K vocab pero con 576 dim y 30 capas — el embedding table consume la mayoría de los params. Para nuestro caso:

| Vocab Size | Embedding Params | % de 10M modelo |
|-----------|-----------------|-----------------|
| 8,192 | 2.1M (con dim=256) | 20% |
| 16,384 | 4.2M | 40% |
| 32,000 | 8.2M | 78% |
| 49,152 | 12.6M | >100% |

**Conclusión:** 8K es el sweet spot para modelos <30M. Weight tying (que ya usamos) es crucial — comparte la embedding table con la output projection, ahorrando ~20% de params.

### 2.2 Byte Latent Transformer (BLT) — Meta

**Qué es:** Elimina el tokenizer por completo. Opera sobre bytes UTF-8 crudos, agrupa bytes en "patches" variables basados en entropía, y procesa los patches con un transformer latente.

**Componentes:**
1. **Entropy Model** (~50-100M params): Predice entropía del siguiente byte para decidir dónde poner límites de patch
2. **Local Encoder**: Convierte bytes → patches via cross-attention
3. **Latent Transformer**: Procesa patches (la parte pesada)
4. **Local Decoder**: Patches → bytes via cross-attention

**Veredicto para nuestro proyecto:** **No práctico.** El entropy model solo ya es más grande que nuestro modelo completo. No se ha probado bajo 400M params. Pero el concepto de "compute variable por unidad de texto" es valioso — mejor logrado con Mixture of Depths a nuestra escala.

### 2.3 Dynamic Token Pooling (Idea Propia)

**Concepto:** En vez de BLT completo, una versión simplificada:

1. Tokenizar normalmente con nuestro BPE 8K
2. Después del embedding, un **pooling layer** merge tokens predecibles en representaciones más compactas
3. El transformer procesa la secuencia comprimida
4. Un decoder expande de vuelta para predicción token-by-token

```python
class DynamicTokenPooler(nn.Module):
    """Merge consecutive tokens when they're highly predictable."""
    def __init__(self, dim, max_merge=4):
        self.merge_predictor = nn.Linear(dim, 1)  # merge score per token
        self.max_merge = max_merge

    def forward(self, x):  # x: [B, T, D]
        scores = torch.sigmoid(self.merge_predictor(x))  # [B, T, 1]
        # Tokens con score alto se fusionan con el siguiente
        # Resultado: secuencia más corta, menos attention cost
        ...
```

**Beneficio:** Reduce la longitud de secuencia para el transformer, similar a cómo BLT agrupa bytes en patches pero sin el overhead masivo.

**Dificultad:** Media-Alta | **Impacto:** Medio

---

## 3. Entrenamiento para Razonamiento

### 3.1 Datos "Textbook Quality" (Enfoque Phi)

Microsoft demostró con Phi-2 (2.7B) que puede superar Llama-2-70B (25x más grande) en razonamiento, usando datos de altísima calidad.

**Qué significa "textbook quality" concretamente:**
1. **Filtrado agresivo** de datos web por nivel educativo (clasificador que distingue contenido de textbook vs web random)
2. **Datos sintéticos con razonamiento explícito** — derivaciones paso a paso, código con explicaciones
3. **Entrenamiento en dos fases:** (1) web diverso para fundamentos, (2) web ultra-filtrado + sintético para razonamiento
4. **Datos en la frontera de capacidad del modelo** — ni muy fáciles ni imposibles

**Aplicación a nuestro proyecto:**
```
Fase 1 (Pretrain): TinyStories + SmolLM (fundamentos de lenguaje)
Fase 2 (Continued Pretrain): Datos sintéticos "textbook" (nuestro synthetic_pretrain/)
Fase 3 (Finetune): Instrucciones con CoT obligatorio
```

Nuestros resultados ya validan esto: el continued pretrain con datos sintéticos mejoró val_loss de 3.76→3.64.

### 3.2 Mezcla Óptima de Datos de Entrenamiento

Investigación sobre ratios óptimos para pretraining:

| Componente | % Recomendado | Propósito |
|-----------|---------------|-----------|
| Texto general (stories, web filtrado) | 40-50% | Fundamentos de lenguaje |
| Código (Python, explicaciones) | 15-20% | Razonamiento estructurado |
| Matemáticas (problemas + soluciones) | 10-15% | Razonamiento numérico |
| Datos factuales (enciclopedia, ciencia) | 10-15% | Conocimiento base |
| Diálogos educativos | 10% | Formato instrucción |

**Insight crítico:** Código temprano en el entrenamiento activa capacidad de razonamiento más rápido. La estrategia óptima es **más código al principio, menos al final** del pretraining.

**Paper:** [MathCoder2](https://arxiv.org/html/2410.08196v1), [CMR Scaling Law](https://aclanthology.org/2024.emnlp-main.903.pdf)

### 3.3 Scratchpad / Working Memory

**Hallazgo fundamental:** Los transformers tienen profundidad de computación limitada por forward pass. Los scratchpads permiten computación secuencial ilimitada al externalizar el estado intermedio.

**Inductive Scratchpad** (Apple, NeurIPS 2024):
- En vez de escribir TODOS los pasos intermedios, el modelo aprende una **función de inducción** que se aplica N veces
- Logra **6x generalización de longitud** en aritmética
- Independiente del número de pasos de razonamiento

**Self-Notes** (Meta, NeurIPS 2023):
- El modelo puede "desviarse" del input en cualquier momento para escribir sus pensamientos
- Notas intercaladas con el texto de entrada
- Supera CoT estándar y scratchpad convencional

**Ya tenemos la infraestructura:** Nuestros tags `<think>` en los datos sintéticos habilitan esto. Necesitamos:
1. Aumentar proporción de datos con `<think>` tags al 40-50%
2. Asegurar que los traces de razonamiento sean **correctos** (verificar)
3. Entrenar con traces progresivamente más largos

### 3.4 Destilación de Razonamiento (Process-Level)

Nuestra destilación actual (`distill.py`) usa outcome-level KL divergence. Para razonamiento necesitamos **process-level**:

**Outcome-level** (actual): Matchea distribución final del teacher → no enseña cómo razonar
**Process-level** (nuevo): Evalúa cada paso intermedio de razonamiento → el student aprende qué pasos son útiles

**Pipeline DeepSeek R1 (adaptado a tiny):**
1. Usar un LLM grande (API o local) para generar reasoning traces
2. Filtrar traces verificando la respuesta final (ejecutar código, verificar matemáticas)
3. Fine-tune nuestro modelo en triples (problema, trace, respuesta) verificados
4. Usar **reverse KLD** (ya soportado en `distill.py`, flag `reverse_kld`) — mejor para students pequeños

**Dificultad:** Media | **Impacto:** Alto

### 3.5 Curriculum de Razonamiento por Complejidad

```
Etapa 1 (steps 0-5K):     1-step reasoning ("If A then B")
Etapa 2 (steps 5K-10K):   2-step reasoning ("A→B→C")
Etapa 3 (steps 10K-15K):  3-step con aritmética básica
Etapa 4 (steps 15K+):     Multi-step con código como razonamiento
```

Similar a nuestro MTP curriculum (k=1→4), pero para complejidad de razonamiento.

---

## 4. Compresión y Eficiencia de Pesos

### 4.1 BitNet b1.58 (Pesos Ternarios)

**Qué es:** Cada peso se restringe a {-1, 0, +1} (1.58 bits). Las activaciones se cuantizan a INT8 per-token.

**Hallazgo clave para modelos tiny:** Un modelo 1.58-bit necesita **~2x hidden dimension** para igualar uno de 16-bit. Pero cada peso ocupa 10x menos memoria.

**Implicación para nuestro 10M:**
```
Actual:  dim=256, hidden_dim=720  → 10.5M params × 16 bits = ~21 MB
BitNet:  dim=512, hidden_dim=1440 → ~40M "logical params" × 1.58 bits = ~8 MB
```
¡Un modelo con 4x más capacidad lógica que cabe en menos espacio!

**Implementación:**
```python
class BitLinear(nn.Module):
    """Drop-in replacement for nn.Linear with ternary weights."""
    def __init__(self, in_features, out_features):
        self.weight = nn.Parameter(torch.randn(out_features, in_features))  # latent FP

    def forward(self, x):
        # Quantize weights to {-1, 0, +1}
        alpha = self.weight.abs().mean()
        w_quant = (self.weight / alpha).round().clamp(-1, 1)

        # Quantize activations to INT8
        gamma = x.abs().max(dim=-1, keepdim=True).values
        x_quant = (x * 127 / (gamma + 1e-8)).round().clamp(-128, 127)

        # Forward with quantized values (STE for backward)
        out = F.linear(x_quant, w_quant) * (alpha * gamma / 127)
        return out
```

**Nota:** SwiGLU se reemplaza por ReLU² en BitNet. Weight decay se elimina en la segunda mitad del training.

**Probado a escala:** Desde **100K hasta 48M params** con éxito ([BitNet b1.58 Reloaded](https://arxiv.org/abs/2407.09527))

**Dificultad:** Media | **Impacto:** Muy Alto | **Papers:** [arxiv 2402.17764](https://arxiv.org/abs/2402.17764), [arxiv 2407.09527](https://arxiv.org/abs/2407.09527)

### 4.2 MatMul-Free Language Model

**Va más allá de BitNet:** Reemplaza TODA la multiplicación de matrices:
- Dense layers → ternary BitLinear (sumas y negaciones)
- Self-attention → **MatMul-free Linear GRU** (productos Hadamard element-wise)

**Resultados a 370M:** Accuracy promedio de 40.0 vs Transformer++ de 41.1 — notablemente cerca con dramáticamente menos computación. Hasta 61% reducción de memoria en training, 10x en inference.

**Paper:** [arxiv 2406.02528](https://arxiv.org/abs/2406.02528)

### 4.3 Weight Sharing Extremo

Nuestro `block_sharing` comparte pares de capas adyacentes. Podemos ir más lejos:

| Estrategia | Capas Únicas | Depth Efectivo | Params Ahorrados |
|-----------|-------------|----------------|-----------------|
| Sin sharing | 24 | 24 | 0% |
| Block sharing (actual) | 12 | 24 | ~50% layers |
| Full loop (4 capas × 6) | 4 | 24 | ~83% layers |
| Universal (1 capa × 24) | 1 | 24 | ~96% layers |

Los params ahorrados se reinvierten en dim más ancho → mejores representaciones por capa.

---

## 5. Ideas Experimentales / Locas

### 5.1 "Dual Brain" — Generador + Verificador (ALTA PRIORIDAD)

**Concepto:** Entrenar dos modelos de 10M:
- **Modelo G (Generador):** Genera respuestas con razonamiento
- **Modelo V (Verificador):** Predice si una respuesta es correcta

**En inference:**
```python
def smart_answer(question, G, V, n_candidates=16):
    candidates = [G.generate(question) for _ in range(n_candidates)]
    scores = [V.score(question, c) for c in candidates]
    return candidates[argmax(scores)]
```

**Por qué funciona:** Un modelo de 10M generando 16 candidatos con un verificador de 10M seleccionando el mejor supera a un modelo de 160M generando 1 respuesta. El costo de inference es similar, pero la calidad es dramáticamente mejor.

**Aplicación inmediata:**
- Para matemáticas: verificar ejecutando el código
- Para lógica: majority voting entre candidatos
- General: entrenar V en pares (pregunta, respuesta, correcto/incorrecto)

**Dificultad:** Baja | **Impacto:** Muy Alto | **Paper:** [TinyGSM](https://arxiv.org/abs/2312.09241)

### 5.2 Tool-Augmented Tiny Model (TALM)

**Hallazgo:** Un modelo de 220M con herramientas supera a uno de 3B sin ellas.

**Tokens especiales para herramientas:**
```
Q: ¿Cuánto es 347 × 892?
A: Necesito multiplicar estos números.
<calc>347 * 892</calc><result>309524</result>
La respuesta es 309,524.
```

**Herramientas a implementar:**
- `<calc>...</calc>` — Calculadora Python
- `<code>...</code>` — Ejecutor de código
- `<search>...</search>` — Retrieval de contexto
- `<result>...</result>` — Resultado inyectado

**Esta es la forma MÁS EFECTIVA de darle aritmética a un modelo de 10M** — no puede internalizar carry operations, pero puede aprender a invocar una calculadora.

**Dificultad:** Baja-Media | **Impacto:** Muy Alto | **Paper:** [TALM](https://arxiv.org/abs/2205.12255)

### 5.3 Iterative Refinement (Pensar Más, No Más Grande)

**Concepto:** El modelo genera una respuesta, luego la toma como contexto adicional y la refina:

```python
def iterative_generate(model, question, n_refinements=3):
    answer = model.generate(f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n")

    for i in range(n_refinements):
        prompt = f"""<|im_start|>user
{question}

Previous attempt: {answer}

Improve this answer:<|im_end|>
<|im_start|>assistant
"""
        answer = model.generate(prompt)

    return answer
```

**Necesita datos de entrenamiento específicos:** Pares (pregunta + intento_previo → respuesta_mejorada)

**Dificultad:** Baja | **Impacto:** Medio

### 5.4 Working Memory Slots (Memoria de Trabajo)

**Concepto:** Agregar 16-32 vectores aprendibles como "slots de memoria" a los que cada token puede leer/escribir durante el forward pass.

```python
class WorkingMemory(nn.Module):
    def __init__(self, dim, n_slots=16):
        self.memory = nn.Parameter(torch.randn(n_slots, dim) * 0.02)
        self.read_proj = nn.Linear(dim, dim)
        self.write_gate = nn.Linear(dim, n_slots)

    def forward(self, x):
        # Read: tokens atienden a la memoria
        read_query = self.read_proj(x)
        attn_weights = torch.softmax(read_query @ self.memory.T / math.sqrt(x.size(-1)), dim=-1)
        memory_read = attn_weights @ self.memory

        # Write: actualizar memoria con info de los tokens
        write_gates = torch.sigmoid(self.write_gate(x)).mean(dim=1)  # [B, n_slots]
        memory_update = write_gates.unsqueeze(-1) * x.mean(dim=1, keepdim=True)
        # (actualización in-place durante training)

        return x + memory_read  # residual
```

**Overhead:** <3% de parámetros adicionales. Da al modelo un "scratchpad interno" para razonamiento multi-paso.

**Dificultad:** Media | **Impacto:** Medio

### 5.5 Mixture of Cognitive Experts (MiCRo)

**Concepto loco:** En vez de expertos genéricos, particionar cada capa en expertos que imitan redes cognitivas:
- **Experto 1:** Lenguaje (fluencia, gramática)
- **Experto 2:** Lógica/Razonamiento (matemáticas, código)
- **Experto 3:** Conocimiento del mundo (hechos, ciencia)
- **Experto 4:** Social (diálogo, contexto)

Un router simple (linear layer) decide cuál experto activar por token.

**Para 10M params:** Simplificar a 2 expertos (Lenguaje + Razonamiento):
```python
class DualExpertFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        self.router = nn.Linear(dim, 2)  # 2 expertos
        self.expert_lang = FeedForward(dim, hidden_dim)
        self.expert_reason = FeedForward(dim, hidden_dim)

    def forward(self, x):
        weights = torch.softmax(self.router(x), dim=-1)
        out = weights[..., 0:1] * self.expert_lang(x) + \
              weights[..., 1:2] * self.expert_reason(x)
        return out
```

**Dificultad:** Media | **Impacto:** Medio-Alto | **Paper:** [arxiv 2506.13331](https://arxiv.org/abs/2506.13331)

### 5.6 Code-as-Reasoning (Razonamiento via Código)

**Insight:** El código es más estructurado que el lenguaje natural → más fácil de aprender para modelos pequeños. Además, el código se puede **ejecutar** para verificar correctness (verificador gratuito).

**En vez de:**
```
Q: Si tengo 15 manzanas y regalo 7, ¿cuántas me quedan?
A: <think>Tengo 15. Regalo 7. 15 - 7 = 8.</think>
Me quedan 8 manzanas.
```

**Entrenar a generar:**
```
Q: Si tengo 15 manzanas y regalo 7, ¿cuántas me quedan?
A: <code>
total = 15
regaladas = 7
quedan = total - regaladas
print(quedan)
</code><result>8</result>
Me quedan 8 manzanas.
```

**TinyGSM demostró:** Un modelo de 1.3B + verificador logra **81.5% en GSM8K** usando Python como formato de razonamiento.

**Dificultad:** Baja | **Impacto:** Alto

### 5.7 🔥 IDEA ORIGINAL: "Resonance Layers" — Capas que Reverberan

**Concepto totalmente nuevo:** Inspirado en cómo las ondas resuenan en una cavidad, crear capas que "reverberen" información entre ellas:

1. El modelo tiene N capas normales + M "resonance buffers" (vectores persistentes)
2. Después de cada capa, la hidden state se "difunde" parcialmente a los buffers
3. Los buffers acumulan información de múltiples capas y pasos
4. Antes de cada capa, los buffers inyectan su contenido acumulado
5. Los buffers decaen exponencialmente (como reverberación acústica)

```python
class ResonanceTransformer(nn.Module):
    def __init__(self, config):
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.resonance = nn.Parameter(torch.zeros(config.n_resonance, config.dim))
        self.decay = 0.9  # factor de decaimiento exponencial
        self.inject_proj = nn.Linear(config.dim, config.dim)
        self.diffuse_proj = nn.Linear(config.dim, config.n_resonance)

    def forward(self, x, freqs_cis):
        resonance = self.resonance.unsqueeze(0).expand(x.size(0), -1, -1)

        for layer in self.layers:
            # Inject resonance into hidden state
            r_weights = torch.softmax(x @ resonance.transpose(-1, -2), dim=-1)
            x = x + 0.1 * self.inject_proj(r_weights @ resonance)

            # Process through transformer layer
            x, _ = layer(x, freqs_cis, mask=None, cache=None)

            # Diffuse hidden state into resonance buffers
            diffuse_weights = torch.sigmoid(self.diffuse_proj(x.mean(dim=1)))  # [B, n_resonance]
            resonance = self.decay * resonance + (1 - self.decay) * \
                       diffuse_weights.unsqueeze(-1) * x.mean(dim=1, keepdim=True)

        return x
```

**Intuición:** Los resonance buffers actúan como una "memoria ecoica" que permite a las capas tempranas comunicarse indirectamente con las tardías a través de un canal persistente. Diferente de Feedback Transformer (que conecta capa final → capa 1), esto crea un canal de comunicación **inter-capa bidireccional continuo**.

**Overhead:** N_resonance × dim params (~0.1% del modelo)

**Dificultad:** Media | **Impacto:** Desconocido (experimental puro)

### 5.8 🔥 IDEA ORIGINAL: "Attention Budget" — Presupuesto de Atención Adaptativo

**Concepto:** Cada token nace con un "presupuesto de atención" que se gasta conforme pasa por las capas. Tokens con presupuesto agotado se saltan capas (como MoD pero con un mecanismo económico más elegante).

```python
class BudgetAttention(nn.Module):
    def __init__(self, config):
        self.initial_budget = nn.Parameter(torch.ones(1) * 5.0)  # presupuesto inicial
        self.cost_estimator = nn.Linear(config.dim, 1)  # costo por capa

    def forward(self, x, layer_fn, freqs_cis):
        budget = self.initial_budget.expand(x.size(0), x.size(1))  # [B, T]

        for layer in self.layers:
            # Estimar costo de procesar cada token en esta capa
            cost = torch.sigmoid(self.cost_estimator(x)).squeeze(-1)  # [B, T]

            # Solo tokens con presupuesto suficiente procesan
            mask = budget > cost
            if mask.any():
                x[mask] = layer(x[mask], freqs_cis)
                budget[mask] = budget[mask] - cost[mask]

        return x
```

**Ventaja sobre MoD:** No requiere top-k fijo, el presupuesto se adapta naturalmente. Tokens de razonamiento "cuestan más" pero el modelo aprende a asignarles más presupuesto.

---

## 6. Roadmap de Implementación

### Prioridad por Impacto/Esfuerzo

```
IMPACTO
  ↑
  │  ★ Dual Brain (5.1)     ★ BitNet (4.1)
  │  ★ Tool-Aug (5.2)       ★ Looped Trans (1.1)
  │  ★ Code-Reasoning (5.6) ★ Universal+ACT (1.5)
  │
  │  ● Feedback Trans (1.2)  ● Process Distill (3.4)
  │  ● Data Curriculum (3.2) ● MoD (1.3)
  │
  │  ○ Working Memory (5.4)  ○ Hourglass (1.4)
  │  ○ Iterative Refine (5.3)○ Dual Expert (5.5)
  │
  │  △ Resonance (5.7)       △ MatMul-free (4.2)
  │  △ Budget Attn (5.8)     △ Dynamic Pool (2.3)
  └──────────────────────────────────→ ESFUERZO
       Bajo                        Alto
```

### Fase 1: Quick Wins (1-2 días cada uno)

1. **Tool-Augmented Generation** — Agregar tokens `<calc>`, `<code>`, `<result>` al tokenizer. Generar datos de entrenamiento con tool-use. Interceptar en inference.

2. **Best-of-N con Verificación** — Implementar en `scripts/test_generation.py`. Generar N=16 candidatos, usar majority voting para matemáticas.

3. **Mejorar datos sintéticos** — Aumentar proporción de CoT (40-50%), agregar code-as-reasoning, verificar traces.

### Fase 2: Arquitectura (3-5 días cada uno)

4. **Feedback Transformer** — Modificar `model.py` para retroalimentar hidden state de la última capa. Minimal, alto impacto.

5. **Looped Transformer** — Nuevo config: 4 capas × 8 loops, dim=384. Requiere step embeddings.

6. **BitNet variant** — Implementar `BitLinear`, crear modelo con dim=512. Comparar con float baseline.

### Fase 3: Training Pipeline (1-2 semanas)

7. **Process-Level Distillation** — Generar reasoning traces verificados con LLM grande. Fine-tune con reverse KLD.

8. **Dual Brain (Gen+Verify)** — Entrenar verificador, implementar rejection sampling.

9. **Curriculum de Razonamiento** — Tres fases de pretraining con complejidad creciente.

### Fase 4: Experimental (variable)

10. **Mixture of Depths** — Router por capa, capacity=0.5.
11. **Working Memory Slots** — 16 slots learnable por bloque.
12. **Resonance Layers** — Buffers de reverberación inter-capa.

---

## Capacidades Realistas por Escala

| Capacidad | 10M | 30M | 110M |
|-----------|-----|-----|------|
| Texto coherente (dominio controlado) | ✅ | ✅ | ✅ |
| Gramática/sintaxis | ✅ | ✅ | ✅ |
| Recall factual básico | ❌ | Limitado | Moderado |
| Razonamiento 1-paso | Marginal | ✅ (con training) | ✅ |
| Razonamiento 2-3 pasos | ❌ standard / ✅ con scratchpad | Marginal | ✅ (con CoT) |
| Aritmética básica | ❌ standard / ✅ con tools | ✅ (con training) | ✅ |
| Seguir instrucciones | Minimal | Básico | Bueno |
| Razonamiento abstracto | ❌ standard / ✅ TRM-style | Limitado | Limitado |
| Generación de código | ❌ | Minimal | Básico |

**La conclusión honesta:** A 10M-100M params, la capacidad de razonamiento viene principalmente de:
1. **Los datos** (calidad > cantidad)
2. **La estrategia de inference** (test-time compute, tools)
3. **La arquitectura** (profundidad via looping, no más params)

NO de los parámetros solos.

---

## Referencias Clave

### Arquitectura
- [Looped Transformers](https://arxiv.org/abs/2502.17416) — Razonamiento con pensamientos latentes
- [Samsung TRM](https://arxiv.org/abs/2510.04871) — 7M params superando 671B en ARC-AGI
- [Feedback Transformer](https://arxiv.org/abs/2002.09402) — 2x eficiencia de tamaño
- [Mixture of Depths](https://arxiv.org/abs/2404.02258) — Compute variable por token
- [Universal Transformer](https://arxiv.org/abs/1807.03819) — Weight sharing + ACT
- [Hourglass Transformer](https://arxiv.org/abs/2110.13711) — Compresión jerárquica

### Datos y Training
- [TinyStories](https://arxiv.org/abs/2305.07759) — Modelos <10M con datos controlados
- [TinyGSM](https://arxiv.org/abs/2312.09241) — 81.5% en GSM8K con modelos pequeños + código
- [Phi-2](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/) — Textbook quality data
- [MathCoder2](https://arxiv.org/html/2410.08196v1) — Código matemático en pretraining
- [Inductive Scratchpad](https://arxiv.org/abs/2406.06467) — 6x generalización de longitud
- [DeepSeek R1](https://arxiv.org/abs/2501.12948) — Destilación de razonamiento

### Compresión
- [BitNet b1.58](https://arxiv.org/abs/2402.17764) — Pesos ternarios
- [BitNet Reloaded](https://arxiv.org/abs/2407.09527) — Probado desde 100K params
- [MatMul-free LM](https://arxiv.org/abs/2406.02528) — Eliminar multiplicaciones
- [BLT](https://arxiv.org/abs/2412.09871) — Byte Latent Transformer (Meta)

### Inference
- [Test-Time Compute Scaling](https://arxiv.org/abs/2408.03314) — Más compute en inference > más params
- [TALM](https://arxiv.org/abs/2205.12255) — 220M con tools > 3B sin tools
- [Self-Notes](https://arxiv.org/abs/2305.00833) — Razonamiento intercalado

### Surveys
- [Small Language Models Survey](https://arxiv.org/abs/2409.15790)
- [SLMs Can Pack a Punch](https://arxiv.org/html/2501.05465v1)
- [MoE Survey](https://arxiv.org/abs/2407.06204)
