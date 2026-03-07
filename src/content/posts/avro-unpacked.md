---
title: Avro Unpacked
pubDate: 2025-02-01
tags:
  - avro
category: Writing
draft: false
---

Data serialization is the backbone of distributed systems, enabling applications to communicate efficiently. Let’s explore why Apache Avro has emerged as a powerful alternative to traditional tools like Java serialization, JSON, and Protobuf—and how it solves their limitations.

## Language-Specific Formats

### Java’s built-in serialization

```java
public class User implements Serializable {
  private String name;
  private int age;
}

// Serialize
ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream("user.ser"));
out.writeObject(new User("Alice", 30));
```

### Python pkl

```python
import pickle

user = {"name": "Alice", "age": 30}
pickle.dump(user, open("user.pkl", "wb"))
```

### Problems with Language-Specific serialization

- Language Lock-in: Java serialization only works in Java; Pickle only in Python.
- Security Risks: Pickle can execute arbitrary code during deserialization.
- Versioning Issues: Adding/removing fields breaks backward compatibility.
- Performance Limitations: Not always optimized for performance, which can be a bottleneck in high-throughput applications.

### How Avro Fixes This

Avro uses language-agnostic schemas (defined in JSON). Data is serialized in a compact binary format readable by any language. No code execution risks.

## Problem with XML and JSON

### JSON example

```json
{
  "name": "Alice",
  "age": 30
}
```

### Problems

- Verbosity: Although all files are ultimately bits on disk, text-based formats like JSON or XML encode data in a human-readable ASCII/UTF-8 representation, while binary formats (like Avro data) use compact machine-readable encodings that are typically smaller and faster to parse.
- Lack of strict typing: No enforcement of data types or structure.
- Schema evolution nightmares: Adding a new field (for example, `email`) can break older clients.

#### Security vulnerabilities

Some built-in serialization mechanisms can be vulnerable to security exploits, particularly when handling data from untrusted sources. Deserializing malicious data could lead to arbitrary code execution.

### How Avro Fixes This

Avro schemas enforce structure and enable safe evolution:

```json
{
  "type": "record",
  "name": "User",
  "fields": [
    {"name": "name", "type": "string"},
    {"name": "age", "type": "int"}
  ]
}
```

Data is stored in binary format (compact), and schemas can evolve without breaking compatibility.

## Thrift and Protobuf

### Protobuf example

```proto
message User {
  required string name = 1;
  required int32 age = 2;
}
```

### Problems

- Schema evolution complexity: Can’t remove fields without reserving tags.

```proto
message User {
  required string name = 1;
  reserved 2;           // Can't use tag 2 again if field removed
  optional string phone = 3;
  optional int32 birth_year = 4; // New field
}
```

- Code generation overhead: Requires compiling `.proto`/`.thrift` files into classes.
- No dynamic typing: Schemas are rigid and tied to generated code.

### How Avro Fixes This

- Schema resolution: Avro readers can use a different schema than writers.
- No code generation: Avro supports dynamic typing (optional).
- Schema stored with data: The schema is embedded in the serialized payload.

## Why Avro

### Intro

Writer schema (old):

```json
{
  "type": "record",
  "name": "User",
  "fields": [
    {"name": "name", "type": "string"},
    {"name": "age", "type": "int"}
  ]
}
```

Reader schema (new):

```json
{
  "type": "record",
  "name": "User",
  "fields": [
    {"name": "name", "type": "string"}
  ]
}
```

When deserializing, Avro ignores the missing `age` field automatically.

### Advantages

- Cross-language support: Works seamlessly in Java, Python, C++, etc.
- Schema evolution: Add/remove fields without breaking compatibility.
- Efficiency: Compact binary format with embedded schemas.

While JSON, Protobuf, and others have their uses, Apache Avro stands out for systems where schemas evolve dynamically and cross-language compatibility is critical.

## Schemas

Apache Avro supports two schema formats.

### Avro IDL (AVDL)

A human-friendly format for defining Avro schemas and RPC protocols. Resembles programming language syntax for readability.

```text
protocol UserService {
  /** A user record */
  record User {
    string name;
    int age;
  }

  // RPC method definition
  User getUser(string id);
}
```

#### Advantages

- **Concise syntax**: Easier to write and read.
- **Supports RPC**: Can define data schemas and RPC interfaces in one file.
- **Namespaces and documentation**: Allows namespacing and inline comments.
- **Code generation**: Compiles to AVSC (JSON) and generates client/server code for RPC.

### Avro Schema (AVSC)

```json
{
  "type": "record",
  "name": "User",
  "namespace": "UserService",
  "fields": [
    {"name": "name", "type": "string"},
    {"name": "age", "type": "int"}
  ]
}
```

#### Advantages

- Machine-readable: JSON is widely supported.
- No compilation needed: Can be used directly.
- Dynamic typing: Schemas can be loaded at runtime.
- Self-describing: Schemas are often stored with data.
- Flexibility: Supports complex types (unions, enums, maps).

### Why AVSC Is Suited for Serialization While AVDL Is Not

- AVDL is an interface definition language and not the runtime format Avro libraries use.
- AVSC is the canonical JSON schema that Avro serialization/deserialization expects.
