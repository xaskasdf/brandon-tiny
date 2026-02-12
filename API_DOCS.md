# LLM API

API local que expone dos endpoints para interactuar con un modelo de lenguaje.

**Base URL:** `http://localhost:5282`
**Swagger UI:** `http://localhost:5282/docs`

---

## POST /completion

Endpoint single-turn. Envías un prompt y recibes una respuesta.

### Parámetros

| Campo | Tipo | Default | Descripción |
|-------|------|---------|-------------|
| `prompt` | string | *requerido* | Texto de entrada |
| `system_prompt` | string | `null` | Instrucción de sistema (opcional) |
| `json` | bool | `false` | Si `true`, fuerza respuesta en JSON |
| `json_schema` | object | `null` | Schema para structured output (ver abajo) |
| `temperature` | float | `0.7` | Creatividad (0.0 - 2.0) |
| `max_tokens` | int | `1024` | Largo máximo de respuesta |

### Ejemplo: Texto plano

```bash
curl -X POST http://localhost:5282/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explica qué es Docker en una frase",
    "temperature": 0.3
  }'
```

**Respuesta:**

```json
{
  "usage": {
    "prompt_tokens": 14,
    "completion_tokens": 25,
    "total_tokens": 39
  },
  "response": "Docker es una plataforma que permite empaquetar aplicaciones en contenedores..."
}
```

### Ejemplo: JSON libre

```bash
curl -X POST http://localhost:5282/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Dame los 3 frameworks web más populares de Python con una descripción corta",
    "json": true
  }'
```

**Respuesta:**

```json
{
  "usage": { "prompt_tokens": 20, "completion_tokens": 80, "total_tokens": 100 },
  "response": {
    "frameworks": [
      { "nombre": "Django", "descripcion": "Framework full-stack con ORM y admin incluido" },
      { "nombre": "Flask", "descripcion": "Microframework ligero y flexible" },
      { "nombre": "FastAPI", "descripcion": "Framework moderno con tipado y async nativo" }
    ]
  }
}
```

### Ejemplo: Structured output (JSON Schema)

Fuerza al modelo a devolver exactamente la estructura definida.

```bash
curl -X POST http://localhost:5282/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Dame 3 lenguajes de programación populares con su año de creación",
    "json": true,
    "json_schema": {
      "name": "lenguajes",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "lenguajes": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "nombre": { "type": "string" },
                "anio": { "type": "integer" }
              },
              "required": ["nombre", "anio"],
              "additionalProperties": false
            }
          }
        },
        "required": ["lenguajes"],
        "additionalProperties": false
      }
    },
    "temperature": 0.3
  }'
```

**Respuesta:**

```json
{
  "usage": { "prompt_tokens": 78, "completion_tokens": 36, "total_tokens": 114 },
  "response": {
    "lenguajes": [
      { "nombre": "Python", "anio": 1991 },
      { "nombre": "Java", "anio": 1995 },
      { "nombre": "JavaScript", "anio": 1995 }
    ]
  }
}
```

### Ejemplo: Con system prompt

```bash
curl -X POST http://localhost:5282/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Cómo hago un SELECT con JOIN?",
    "system_prompt": "Eres un experto en PostgreSQL. Responde con ejemplos de código.",
    "temperature": 0.2
  }'
```

---

## POST /chat

Endpoint multi-turno. Envías el historial completo de mensajes.

### Parámetros

| Campo | Tipo | Default | Descripción |
|-------|------|---------|-------------|
| `messages` | array | *requerido* | Lista de `{ role, content }` |
| `json` | bool | `false` | Si `true`, fuerza respuesta en JSON |
| `json_schema` | object | `null` | Schema para structured output |
| `temperature` | float | `0.7` | Creatividad (0.0 - 2.0) |
| `max_tokens` | int | `1024` | Largo máximo de respuesta |

**Roles válidos:** `system`, `user`, `assistant`

### Ejemplo: Conversación

```bash
curl -X POST http://localhost:5282/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      { "role": "system", "content": "Eres un asistente experto en Python." },
      { "role": "user", "content": "Qué es un decorador?" },
      { "role": "assistant", "content": "Un decorador es una función que modifica el comportamiento de otra función." },
      { "role": "user", "content": "Dame un ejemplo simple" }
    ],
    "temperature": 0.3
  }'
```

**Respuesta:**

```json
{
  "usage": { "prompt_tokens": 55, "completion_tokens": 80, "total_tokens": 135 },
  "response": "Aquí tienes un ejemplo simple..."
}
```

### Ejemplo: Chat con structured output

```bash
curl -X POST http://localhost:5282/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      { "role": "system", "content": "Analiza el sentimiento del texto del usuario." },
      { "role": "user", "content": "Me encanta este producto, es increíble!" }
    ],
    "json": true,
    "json_schema": {
      "name": "sentimiento",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "sentimiento": { "type": "string", "enum": ["positivo", "negativo", "neutro"] },
          "confianza": { "type": "number" },
          "palabras_clave": { "type": "array", "items": { "type": "string" } }
        },
        "required": ["sentimiento", "confianza", "palabras_clave"],
        "additionalProperties": false
      }
    }
  }'
```

---

## Modos de respuesta

| `json` | `json_schema` | Comportamiento |
|--------|---------------|----------------|
| `false` | — | Texto plano |
| `true` | `null` | JSON libre (el modelo decide la estructura) |
| `true` | `{ name, strict, schema }` | Structured output (respeta el schema exacto) |

---

## Formato de json_schema

```json
{
  "name": "nombre_del_schema",
  "strict": true,
  "schema": {
    "type": "object",
    "properties": {
      "campo1": { "type": "string" },
      "campo2": { "type": "integer" },
      "campo3": {
        "type": "array",
        "items": { "type": "string" }
      }
    },
    "required": ["campo1", "campo2", "campo3"],
    "additionalProperties": false
  }
}
```

**Reglas cuando `strict: true`:**
- Todos los campos deben estar en `required`
- Debe incluir `"additionalProperties": false` en cada objeto
- Tipos soportados: `string`, `number`, `integer`, `boolean`, `array`, `object`, `null`
- Se puede usar `enum` para valores fijos

---
