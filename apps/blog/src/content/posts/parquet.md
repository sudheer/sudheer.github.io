---
title: Parquet Unpacked
pubDate: 2026-02-01
tags:
  - parquet
category: Writing
draft: false
---

# Understanding Apache Parquet: From Row Groups to Page Indexes

Parquet becomes much easier to understand once you stop thinking of it as simply **“a columnar file format”** and instead learn its physical structure one layer at a time.

The core hierarchy is:

```text
PARQUET FILE
    │
    └── ROW GROUP
          │
          └── COLUMN CHUNK
                  │
                  └── PAGE
```

Everything else—dictionary encoding, statistics, Bloom filters, page indexes, schema annotations—fits around this structure.

---

## 1. Why Parquet Is Columnar

Consider this table:

```text
┌────┬────────┬─────┬────────┐
│ id │ name   │ age │ salary │
├────┼────────┼─────┼────────┤
│ 1  │ Alice  │ 28  │ 100k   │
│ 2  │ Bob    │ 35  │ 120k   │
│ 3  │ Carol  │ 42  │ 150k   │
└────┴────────┴─────┴────────┘
```

A row-oriented format conceptually stores:

```text
1, Alice, 28, 100k
2, Bob,   35, 120k
3, Carol, 42, 150k
```

Parquet organizes values by column:

```text
id      → 1, 2, 3
name    → Alice, Bob, Carol
age     → 28, 35, 42
salary  → 100k, 120k, 150k
```

Why?

Consider:

```sql
SELECT AVG(salary)
FROM employees;
```

A columnar reader can conceptually do:

```text
id       → SKIP
name     → SKIP
age      → SKIP

salary   → READ
             │
             ▼
           AVG()
```

This is **column pruning**.

Parquet is therefore especially useful when queries examine:

```text
MANY ROWS
   +
FEW COLUMNS
```

---

## 2. Row Groups

Parquet does not necessarily store an entire file's column as one giant contiguous unit.

First, the logical table is divided horizontally into **row groups**.

```text
┌────┬────────┬─────┐
│ id │ name   │ age │
├────┼────────┼─────┤
│ 1  │ Alice  │ 28  │  ┐
│ 2  │ Bob    │ 35  │  ├── Row Group 1
│ 3  │ Carol  │ 42  │  ┘
├────┼────────┼─────┤  ← horizontal boundary
│ 4  │ Dan    │ 25  │  ┐
│ 5  │ Eve    │ 31  │  ├── Row Group 2
│ 6  │ Frank  │ 39  │  ┘
└────┴────────┴─────┘
```

“Horizontal” describes how we divide the **logical table**.

Inside each row group, the data is still organized by column:

```text
                ROW GROUP 1
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
         id         name       age

      [1,2,3]    [A,B,C]   [28,35,42]
```

> **Row Group:** A horizontal batch of rows whose contents are stored column-by-column.

---

## 3. Column Chunks

Inside a row group, each column becomes a **column chunk**.

```text
                    ROW GROUP 1
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      id chunk       name chunk      age chunk
       [1,2,3]        [A,B,C]       [28,35,42]
```

> **Column Chunk:** One column's data within one row group.

Therefore:

```text
3 Row Groups
×
4 Columns
────────────
12 Column Chunks
```

The same logical column has a different chunk in every row group:

```text
                  age
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
       RG1        RG2        RG3
        │          │          │
    age chunk  age chunk  age chunk
```

---

## 4. Pages

Column chunks can still be large, so Parquet divides them into **pages**.

```text
age COLUMN CHUNK
        │
        ├── Page 1
        ├── Page 2
        ├── Page 3
        └── Page 4
```

Our hierarchy is now:

```text
PARQUET FILE
    │
    └── ROW GROUP
          │
          └── COLUMN CHUNK
                  │
                  ├── PAGE
                  ├── PAGE
                  └── PAGE
```

Mental model:

```text
Row Group
    ↓
batch of rows

Column Chunk
    ↓
one column from that batch

Page
    ↓
smaller unit inside the column chunk
```

---

## 5. Data Pages and Dictionary Pages

Two important page concepts are:

```text
PAGE
 │
 ├── Data Page
 │
 └── Dictionary Page
```

A **data page** contains encoded column data.

Suppose:

```text
country

US
US
UK
US
IN
UK
US
```

Conceptually:

```text
country Column Chunk
        │
        ├── Data Page 1
        │     US
        │     US
        │     UK
        │     US
        │
        └── Data Page 2
              IN
              UK
              US
```

---

## 6. Dictionary Encoding

Repeated values can be expensive to store repeatedly.

Instead of:

```text
US, US, UK, US, IN, UK, US
```

Parquet can construct a dictionary:

```text
0 → US
1 → UK
2 → IN
```

The data becomes:

```text
US  US  UK  US  IN  UK  US
             ↓
0   0   1   0   2   1   0
```

The column chunk now conceptually looks like:

```text
country COLUMN CHUNK
        │
        ├── Dictionary Page
        │      0 → US
        │      1 → UK
        │      2 → IN
        │
        ├── Data Page
        │      0 0 1 0
        │
        └── Data Page
               2 1 0
```

Remember:

```text
Dictionary Page
       ↓
dictionary values


Data Page
       ↓
encoded references to those values
```

Dictionary encoding is particularly effective for low-cardinality or highly repetitive data.

---

## 7. Plain-Encoding Fallback

Dictionary encoding is not always efficient.

Consider unique identifiers:

```text
a81f...
b72c...
c93e...
d11a...
e57b...
```

A dictionary might become almost as large as the original data.

A writer can therefore stop dictionary encoding and switch later pages to another encoding, commonly `PLAIN`.

```text
COLUMN CHUNK
│
├── Dictionary Page
│
├── Data Page
│      dictionary encoded
│
├── Data Page
│      dictionary encoded
│
│   dictionary becomes inefficient
│
├── Data Page
│      PLAIN
│
└── Data Page
       PLAIN
```

> Do not assume one column chunk must use exactly one encoding for all its data pages.

---

## 8. File Footer and Metadata

At the end of a Parquet file is its file metadata/footer.

```text
┌──────────────────────────────┐
│         PARQUET FILE         │
├──────────────────────────────┤
│ Row Group 1                  │
│   column chunks + pages      │
├──────────────────────────────┤
│ Row Group 2                  │
│   column chunks + pages      │
├──────────────────────────────┤
│ Row Group 3                  │
│   column chunks + pages      │
├──────────────────────────────┤
│                              │
│ FILE METADATA / FOOTER       │
│                              │
│ • schema                     │
│ • row-group metadata         │
│ • column metadata            │
│ • statistics                 │
│ • offsets                    │
│ • encodings                  │
│ • compression information   │
└──────────────────────────────┘
```

Think of the footer as the reader's **map of the file**.

---

## 9. Column-Chunk Statistics

Parquet can store statistics describing a column chunk.

```text
age Column Chunk

values:
[18, 22, 25, 31, 40]

statistics:
min         = 18
max         = 40
null_count  = 0
value_count = 5
```

Now imagine:

```text
             AGE

RG1       min=18  max=40
RG2       min=41  max=65
RG3       min=66  max=90
```

Query:

```sql
WHERE age > 70
```

The reader can reason:

```text
RG1   18 ─── 40      SKIP

RG2   41 ─── 65      SKIP

RG3   66 ───── 90    MAY MATCH
                        │
                        ▼
                       READ
```

This allows **row-group pruning**.

### Important Statistics

| Statistic | Meaning |
|---|---|
| `min` | Lower bound / minimum |
| `max` | Upper bound / maximum |
| `null_count` | Number of nulls |
| `value_count` | Number of encoded values in the column chunk |

For nested/repeated schemas, don't automatically interpret `value_count` as simply the number of non-null SQL values.

---

## 10. Statistics Truncation

Numbers have small min/max representations:

```text
min = 18
max = 90
```

Strings or binary values can be enormous.

Instead of putting giant values into metadata, writers may use bounded/truncated statistics.

Conceptually:

```text
actual value:

"watermelon-some-extremely-long-value........"

                 ↓

metadata bound:

"water..."
```

The actual truncation rules must preserve safe comparison bounds. It is **not** simply “cut both strings after N characters.”

The goal is:

```text
keep metadata small
       +
retain useful skipping information
```

---

## 11. Bloom Filters

Min/max statistics have a limitation.

Suppose:

```text
user_id chunk

min = 100
max = 9,000,000
```

Query:

```sql
WHERE user_id = 472938
```

Because:

```text
100 ───────── 472938 ───────── 9,000,000
```

min/max cannot eliminate the chunk.

A **Bloom filter** can help.

```text
Column Chunk values
        │
        ▼
┌────────────────┐
│  Bloom Filter  │
└───────┬────────┘
        │
        ▼
     472938?
```

It can answer:

```text
Definitely NOT present
        │
        ▼
       SKIP
```

or:

```text
MAYBE present
        │
        ▼
continue checking
```

Bloom filters can produce **false positives**:

```text
Bloom says MAYBE
      │
      ├── actually exists
      │
      └── actually absent
```

But a correct Bloom filter does not produce false negatives:

```text
Bloom says NO
      │
      ▼
definitely absent
```

Memory trick:

```text
min/max
   ↓
range filtering


Bloom filter
   ↓
equality / point lookups
```

---

## 12. Page Index

Column-chunk statistics operate at a coarse level.

Suppose:

```text
age Column Chunk

overall:
min=10
max=50
```

Query:

```sql
WHERE age >= 45
```

The entire chunk cannot be skipped.

But its pages might look like:

```text
Page 1    10–20
Page 2    21–30
Page 3    31–40
Page 4    41–50
```

Only Page 4 could contain values ≥45.

This is where the **Page Index** helps:

```text
             PAGE INDEX
                 │
        ┌────────┴────────┐
        ▼                 ▼
  Column Index       Offset Index
```

Memory trick:

```text
Column Index
     ↓
WHAT is in the pages?


Offset Index
     ↓
WHERE are the pages?
```

---

## 13. Column Index

The Column Index provides page-level statistics/information.

```text
age COLUMN CHUNK

┌────────┬─────┬─────┐
│        │ min │ max │
├────────┼─────┼─────┤
│ Page 1 │ 10  │ 20  │
│ Page 2 │ 21  │ 30  │
│ Page 3 │ 31  │ 40  │
│ Page 4 │ 41  │ 50  │
└────────┴─────┴─────┘
```

For:

```sql
WHERE age >= 45
```

the reader gets:

```text
Page 1   10–20   SKIP
Page 2   21–30   SKIP
Page 3   31–40   SKIP
Page 4   41–50   MAY MATCH
```

So the Column Index answers:

> **Which pages might contain useful data?**

---

## 14. Offset Index

After deciding that Page 4 matters, the reader needs to locate it.

That's the Offset Index's job.

Conceptually:

```text
Page 1 → offset 10,000
Page 2 → offset 14,000
Page 3 → offset 18,000
Page 4 → offset 22,000
```

Page locations contain information such as:

```text
offset
compressed_page_size
first_row_index
```

The relationship:

```text
Column Index
      │
      ▼
"Page 4 might match"
      │
      ▼
Offset Index
      │
      ▼
"Page 4 is here"
      │
      ▼
READ PAGE 4
```

Therefore:

```text
Chunk statistics → skip large chunk/row group

Column Index      → choose pages

Offset Index      → locate pages
```

---

## 15. Physical Types vs Logical Types

So far we've discussed **where data lives**.

Now:

> What does the stored data mean?

Parquet has primitive/physical storage types such as:

```text
BOOLEAN
INT32
INT64
FLOAT
DOUBLE
BYTE_ARRAY
FIXED_LEN_BYTE_ARRAY
...
```

Applications use richer concepts:

```text
DATE
TIMESTAMP
STRING
DECIMAL
UUID
JSON
...
```

Parquet combines physical representation with logical annotations.

For example:

```text
birth_date
│
├── Physical type: INT32
│
└── Logical type: DATE
```

Conceptually:

```text
2026-09-16
     │
     ▼
    DATE
     │
     ▼
   INT32
     │
     ▼
days since 1970-01-01
```

The physical type answers:

> **How is this stored?**

The logical annotation answers:

> **How should I interpret it?**

---

## 16. Logical-Type Examples

### String

```text
name
│
├── Physical: BYTE_ARRAY
└── Logical:  STRING
```

```text
"Alice"
   ↓
STRING
   ↓
BYTE_ARRAY
```

### Decimal

```text
price
│
├── Physical: INT64
└── Logical:
      DECIMAL
      precision=10
      scale=2
```

A physical integer:

```text
12345
```

with scale `2` represents:

```text
123.45
```

Remember:

```text
Physical type
      ↓
storage


Logical type
      ↓
meaning
```

---

## 17. OriginalType vs LogicalTypeAnnotation

This distinction mostly comes from Parquet's evolution.

Older representations used converted types, commonly represented in APIs as `OriginalType`.

```text
OLDER

INT32
  │
  └── OriginalType.DATE
```

Modern Parquet uses richer logical-type annotations:

```text
NEWER

INT32
  │
  └── LogicalTypeAnnotation.DATE
```

The newer representation can express richer semantics.

For example:

```text
TIMESTAMP
│
├── unit
│    ├── MILLIS
│    ├── MICROS
│    └── NANOS
│
└── isAdjustedToUTC
```

There aren't separate physical and logical values.

There is one physical representation plus schema metadata describing its meaning:

```text
stored INT32
     │
     ▼
    20712
     │
     │ schema says DATE
     ▼
 interpreted as
 2026-09-16
```

---

## 18. Field IDs

Schema fields normally have names:

```text
id
name
age
```

But names can change.

Version 1:

```text
name
```

Version 2:

```text
full_name
```

How can a system know this was a rename rather than a completely different field?

A Parquet schema field can have a **Field ID**.

```text
name
├── Field ID = 2
└── STRING
```

After renaming:

```text
full_name
├── Field ID = 2
└── STRING
```

Therefore:

```text
OLD                         NEW

name                        full_name
Field ID = 2  ───────────►  Field ID = 2
```

The name changed.

The identity remained stable.

This is particularly useful for **schema evolution**.

### Nested Example

```text
user
│
├── name          [ID 1]
│
├── address       [ID 2]
│    ├── city     [ID 3]
│    └── zip      [ID 4]
│
└── age           [ID 5]
```

A nested field can be renamed while preserving its identity.

Field IDs are **schema metadata**. They are not repeated alongside every value.

---

# 19. Putting Everything Together

Consider:

```sql
SELECT name
FROM users
WHERE age = 35;
```

The reader can progressively eliminate unnecessary work:

```text
                    QUERY
                 age = 35
                     │
                     ▼
                FILE FOOTER
                     │
                     ▼
             inspect schema
                     │
                     ▼
             column pruning
              age + name only
                     │
                     ▼
          Column Chunk Statistics
                     │
              Which Row Groups?
                     │
           ┌─────────┼─────────┐
           ▼         ▼         ▼
          RG1       RG2       RG3
          SKIP      KEEP      SKIP
                     │
                     ▼
                Bloom Filter
                     │
                  MAYBE
                     │
                     ▼
                Column Index
                     │
              Which pages?
                     │
                     ▼
                  Page 3
                     │
                     ▼
                Offset Index
                     │
              Where is Page 3?
                     │
                     ▼
                 Read Page
                     │
                     ▼
          decompress + decode
                     │
                     ▼
                age = 35
                     │
                     ▼
            matching name values
```

---

# 20. The Optimization Ladder

```text
1. COLUMN PRUNING
        ↓
Don't read unnecessary columns

2. COLUMN-CHUNK STATISTICS
        ↓
Don't read unnecessary row groups

3. BLOOM FILTER
        ↓
Prove searched values are absent

4. COLUMN INDEX
        ↓
Don't read unnecessary pages

5. OFFSET INDEX
        ↓
Locate useful pages

6. ENCODING + COMPRESSION
        ↓
Store/read fewer bytes
```

Not every Parquet file contains Bloom filters or page indexes, and not every reader uses every optional optimization.

The fundamental structure remains:

```text
FILE
 │
 ├── ROW GROUP
 │      │
 │      └── COLUMN CHUNK
 │              │
 │              └── PAGE
 │
 └── FOOTER / METADATA
```

---

# Final Mental Model

If you remember only one diagram, use this:

```text
                       PARQUET FILE
                            │
              ┌─────────────┴─────────────┐
              │                           │
             DATA                       FOOTER
              │                           │
          ROW GROUP                       ├── Schema
              │                           │     │
         COLUMN CHUNK                     │     ├── name
              │                           │     ├── Field ID
       ┌──────┴──────┐                    │     ├── physical type
       │             │                    │     └── logical type
       ▼             ▼                    │
 Dictionary       Data Pages              └── Row Group /
    Page                                      Column metadata
                                                   │
                                                   ├── min
                                                   ├── max
                                                   ├── null_count
                                                   └── value_count

                    COLUMN CHUNK
                          │
               ┌──────────┼──────────┐
               ▼          ▼          ▼
            Statistics   Bloom    Page Index
                                    │
                             ┌──────┴──────┐
                             ▼             ▼
                       Column Index   Offset Index
                             │             │
                             ▼             ▼
                       WHAT pages?    WHERE pages?
```

---

# Cheat Sheet

| Concept | Remember it as |
|---|---|
| **Row Group** | Horizontal batch of rows |
| **Column Chunk** | One column within one row group |
| **Page** | Smaller storage unit inside a column chunk |
| **Data Page** | Encoded column data |
| **Dictionary Page** | Dictionary values |
| **Dictionary Encoding** | Replace repeated values with compact dictionary IDs |
| **PLAIN** | Direct representation; useful when dictionary encoding isn't appropriate |
| **Footer** | Map/schema/metadata describing the file |
| **min / max** | Range-based skipping |
| **null_count** | Number of nulls represented by statistics |
| **Bloom Filter** | “Definitely absent” or “maybe present” |
| **Column Index** | What is inside individual pages? |
| **Offset Index** | Where are those pages? |
| **Physical Type** | How bytes are represented |
| **Logical Type** | What those bytes mean |
| **OriginalType** | Older converted/logical type representation |
| **LogicalTypeAnnotation** | Newer, richer logical-type representation |
| **Field ID** | Stable schema-field identity |

---

# The One Idea Behind Parquet

Most of Parquet's design ultimately helps a query engine answer:

```text
"Can I avoid reading this?"
```

```text
        FILE
         │
         ▼
      COLUMN? ── no ──► SKIP
         │
        yes
         ▼
    ROW GROUP? ─ no ──► SKIP
         │
        yes
         ▼
       PAGE? ─── no ──► SKIP
         │
        yes
         ▼
       READ
         │
         ▼
      DECODE
         │
         ▼
      RESULT
```

That is the core mental model:

> **Parquet organizes data and metadata so analytical engines can read as little data as possible.**

NOTE: Written by AI, Prompted by Human
