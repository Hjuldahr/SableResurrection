# Sable Memory Schema

## Overview

Sable's memory is divided into three conceptual layers:

1. **Persistent Memory** — information retained across sessions and retrieved semantically.
2. **Runtime State** — Sable's current internal state, loaded into the program rather than treated as semantic memory.
3. **External Knowledge** — information obtained from outside sources when existing knowledge is insufficient, stale, or otherwise requires verification.

The vector database is used primarily as a **semantic retrieval system**, not as the sole source of truth. Text and structured metadata remain authoritative.

---

# Persistent Memory

## Raw Conversation

Timestamped conversational source material used for contextual retrieval, reflection, and memory formation.

### Fields

```text
timestamp: datetime
user_id: int
channel_id: int
message_id: int
text: text
tokens: int
embedding: vector
```

### Purpose

Raw conversation provides the evidence from which higher-level memories are derived.

It should generally remain immutable after insertion.

The embedding represents the semantic content of the message and allows retrieval of relevant past conversation without requiring exact keyword matches.

### Example

```text
timestamp: 2026-08-27T22:54:00-04:00
user_id: 123
channel_id: 456
message_id: 789
text: "I've been experimenting with Qdrant for Sable's memory system."
embedding: [...]
```

---

# Learned Topic Facts

Information Sable has learned about topics through conversation, observation, explicit user requests, or external research.

### Fields

```text
source_type: enum(
    user_requested,
    observation,
    web_lookup
)
source_message_id: int | null

topic_id: int | null

created: datetime
last_referenced: datetime

fact: text
embedding: vector
```

### Source Types

#### `user_requested`

Information the user explicitly asks Sable to retain or identifies as important.

Examples:

```text
"Sable, remind me about X later."

"This is very important to me."
```

This represents deliberate memory formation rather than an inference made from ordinary conversation.

#### `observation`

Information inferred or learned from normal interaction.

Examples:

```text
Python uses significant indentation.
Digital painting often involves iterative brushwork.
The local job market is currently difficult for this field.
```

The reflection system determines whether an observation is sufficiently durable to store.

#### `web_lookup`

Information acquired or verified through external lookup.

Used when information is:

* volatile
* rapidly changing
* recent
* regional
* otherwise dependent on current external knowledge

Examples include:

```text
current news
recent art movements
current scientific developments
regional recommendations
current events
```

### Topic Association

`topic_id` identifies the conceptual topic to which the fact belongs.

Examples:

```text
Programming
Digital Painting
Qdrant
Aboriginal Culture
Job Market
```

The embedding remains responsible for semantic retrieval; `topic_id` provides deterministic association and filtering.

### Provenance

`source_message_id` should identify the conversation that caused the fact to be learned when applicable.

For web-derived facts, an additional external-source record or URL may be associated with the fact.

---

# Learned User Facts

Information Sable has learned about an individual user through interaction.

### Fields

```text
user_id: int

created: datetime
last_referenced: datetime

confidence: float

fact: text

source_message_id: int

embedding: vector
```

### Examples

```text
user_id: 123
fact: "The user enjoys digital painting."
confidence: 0.9
```

```text
user_id: 123
fact: "The user is ambidextrous."
confidence: 0.8
```

Potential memory categories include:

```text
hobbies
preferences
skills
projects
languages
communication habits
etc.
```

### Confidence

Confidence represents Sable's confidence that the fact is actually true.

It may increase when subsequent interactions reinforce an existing memory and decrease when later interactions contradict it.

The reflection system should preferably update an existing memory when new evidence reinforces or contradicts it rather than creating redundant memories.

---

# Sable's Opinions

Sable's subjective judgments about users or topics.

### Fields

```text
subject_type: enum(
    user,
    topic
)

subject_id: int

created: datetime
last_referenced: datetime

strength: float

opinion: text
embedding: vector
```

### Examples

```text
subject_type: user
subject_id: 123
opinion: "Nioreux is technically ambitious."
```

```text
subject_type: topic
subject_id: 42
opinion: "Sable finds functional programming particularly elegant."
```

Opinions represent Sable's own perspective and should not be treated as objective facts.

---

# Sable's Preferences

Sable's persistent likes, dislikes, and preferences.

### Fields

```text
created: datetime
last_referenced: datetime

preference: text
strength: float

embedding: vector
```

### Examples

```text
"Sable prefers cats over dogs."
"Sable enjoys discussing programming languages."
"Sable prefers instrumental music."
```

Preferences differ from opinions:

```text
Preference:
"Sable prefers Python."

Opinion:
"Sable thinks Python has an elegant syntax."
```

A preference describes what Sable tends to choose or enjoy; an opinion describes a judgment about something.

---

# Runtime State

These values represent Sable's current state rather than semantic memories.

They should not initially be stored as vectorized memories.

---

## Sable's Persona

Current personality configuration used to construct an instruction modifier and potentially influence inference parameters.

### Fields

```text
playful: float
curious: float
formal: float
sarcastic: float
...
```

Values should normally occupy a defined range such as:

```text
0.0 → absent
1.0 → strongly expressed
```

### Purpose

Persona controls **how Sable behaves**, rather than what she knows.

It may be synthesized or modified based on accumulated memories and interactions.

The initial implementation should primarily translate persona values into prompt instructions.

Inference parameters such as temperature, repetition penalty, or maximum output length can later be influenced by persona if experimentation demonstrates that this produces useful and stable behavior.

---

## Sable's Emotion

Current emotional state represented using a Valence-Arousal-Dominance model.

### Fields

```text
valence: float
arousal: float
dominance: float

updated_at: datetime
```

### Purpose

Emotion represents Sable's current state and should influence the current interaction rather than being treated as ordinary long-term memory.

For example:

```text
valence
    negative ←──── 0 ────→ positive

arousal
    calm     ←──── 0 ────→ excited

dominance
    submissive ←── 0 ──→ dominant
```

A historical emotional event may eventually become a normal memory, but the current VAD state remains runtime state.

---

# External Knowledge

## Web Lookup

Web lookup is an external information source rather than a persistent memory collection.

It may be triggered when:

* existing model knowledge is insufficient
* learned memories are insufficient
* information is likely stale
* the user explicitly requests current information
* information is rapidly changing
* information is recent
* information is regional or location-dependent

Examples:

```text
current news
current software versions
recent scientific developments
local recommendations
regional cultural information
current events
```

The result may subsequently become a `Learned Topic Fact` if it is sufficiently durable to warrant retention.

---

# Retrieval Architecture

Semantic retrieval should be available to both **conversation generation** and **reflection**.

```text
                    Raw Conversation
                           │
                           ▼
                  Semantic Retrieval
                           │
                           ▼
                       Reflection
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
     User Facts       Topic Facts      Sable Memories
          │                │             ┌────┴────┐
          │                │             ▼         ▼
          │                │         Opinions  Preferences
          │                │
          └────────────────┼────────────────┘
                           │
                           ▼
                  Context Retrieval
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
          Persona                    Emotion
          Runtime                    Runtime
              │                         │
              └────────────┬────────────┘
                           ▼
                          LLM
```

Reflection should be able to retrieve existing memories before creating new ones.

For example:

```text
Existing memory:
"User enjoys digital painting."

New conversation:
"I've been doing a lot of digital painting lately."
```

Rather than creating another independent memory, reflection can recognize that the existing memory is being reinforced.

---

# Memory Lifecycle

The intended lifecycle is:

```text
Conversation
     │
     ▼
Raw Conversation
     │
     ▼
Relevant historical retrieval
     │
     ▼
Reflection
     │
     ├── ignore
     │
     ├── reinforce existing memory
     │
     ├── update existing memory
     │
     └── create new memory
              │
              ▼
        Persistent Memory
```

Retrieval updates:

```text
last_referenced
```

allowing frequently useful memories to become distinguishable from memories that have not been relevant for a long time.

Future memory scoring can combine:

```text
semantic similarity
+
confidence
+
recency
+
reference frequency
+
scope / association
```

rather than relying solely on vector similarity.

---

# Design Principles

### 1. Raw conversation is evidence

Higher-level memories should be derived from raw conversation rather than replacing it.

### 2. Embeddings are retrieval aids

The vector represents semantic similarity. Structured metadata remains authoritative for identity, scope, provenance, and filtering.

### 3. Memory should be selective

Not every message should become a durable memory.

### 4. Memories should be revisable

New evidence should be able to reinforce, weaken, update, or invalidate existing memories.

### 5. Facts and opinions remain distinct

Sable's belief about something should not be indistinguishable from information that is intended to be objective.

### 6. Persistent knowledge and runtime state remain separate

Persona and current emotion affect generation but are not ordinary semantic memories.

### 7. Provenance matters

Sable should be able to distinguish something she:

* inferred from conversation
* was explicitly asked to remember
* learned from external research

### 8. Retrieval should support both generation and reflection

Memory retrieval is useful not only for answering the user but also for deciding what Sable already knows.

---

# Initial Vector Database Scope

For the initial prototype, only the following need to exist:

```text
Raw Conversation
Learned Topic Facts
Learned User Facts
Sable's Opinions
Sable's Preferences
```

Persona and Emotion can remain ordinary program state.

Web lookup can remain an external tool.

The prototype can therefore be tested independently of Discord or the conversational runtime:

```text
synthetic memories
       ↓
embedding
       ↓
Qdrant
       ↓
semantic query
       ↓
retrieved memories
       ↓
evaluation
```

Once retrieval quality is established, the memory layer can be attached to Sable's conversational runtime and eventually to the Discord implementation.