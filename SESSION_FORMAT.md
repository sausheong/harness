# Session JSONL format, version 1

New sessions begin with one `session_header` record. Its nonempty `id` is the
session identity, `schemaVersion` is 1, and `timestamp` records creation time.
The header has no parent. It is independent of agent/key names and survives
renaming the file. Copying a file preserves identity; a product-level fork must
assign a new identity explicitly rather than pretending the copy is a new session.

```json
{"id":"ses_example","type":"session_header","schemaVersion":1,"timestamp":1788796800,"data":null}
{"id":"e_example","type":"message","role":"user","timestamp":1788796801,"data":{"text":"Hello"}}
```

The header must be the first nonblank record and cannot occur again. Unsupported
header versions, a version field on a conversation record, and graph links to
headers or selection controls are errors. No loader, recovery or migration path
silently accepts an unsupported version. Entry IDs are unique within the log;
parents precede children. Selection controls use `type: selection`, their target
in `parentId`, and `data: {"version":1}`. They select a conversation node without
becoming part of its history. Ordinary appends subsequently select the new node.

`Session.Entries` contains graph entries and selection controls, excluding the
format header. `History` and `View` exclude both kinds of control record. A raw
file export must retain the header; serialising `Entries` alone is not a complete
identity-preserving session export. Store.List exposes the stored session ID and
excludes the header from EntryCount.

Store.Create commits a complete synced header file without overwriting an
existing key. First append to a new/empty file writes the header before the entry.
Rewrite retains the same header. Use an exclusive session lease and Flush/Close
according to their documented producer-joining contract. A failed append may
leave an incomplete final record and must be treated as degraded persistence.

Legacy headerless files remain readable without implicit modification. Their
historical format has no persisted session ID, so ordinary Load still assigns
an ephemeral ID. Use MigrateLegacy before exposing durable session identity:
it adds the version-1 header, corrects supported old compaction duplicate IDs,
and retains a synced complete-original backup before atomic replacement. The
report distinguishes format upgrade from duplicate-ID remapping. Repeating the
migration on a valid version-1 file preserves its exact bytes and ID.

Recovery repairs only incomplete final JSON objects and retains its own backup.
A future-version or structurally invalid complete record is never a recoverable
tail. Truncated first-header recovery yields an empty log; without a complete
header there was no recoverable durable identity. Unsupported versions require
an explicit future migration implementation.

Payload-specific schemas, application session catalogues, attachment/export
policies and cross-platform release qualification remain separate work. Unknown
JSON fields are retained by the legacy converter; generic typed rewrites do not
promise lossless preservation of arbitrary unknown fields.

## Application annotations

`annotation` records contain `data: {"version":1,"kind":"app.name","payload":{...}}`.
They are durable application metadata, exposed by Entries and Annotations but
excluded from History and View. They do not move the selected leaf, even in an
empty session. Their optional parent attaches them to the current conversation
node; conversation nodes cannot use annotations as parents or branch targets.
Kinds contain ASCII letters, digits, dots, underscores or hyphens (1–128 bytes).
The encoded annotation data is limited to 64 KiB. Unsupported annotation versions
are rejected. Annotate flushes before committing the in-memory record; invalid
input returns an error without degrading persistence. Stored metadata is retained
by full graph rewrites and raw exports. Applications define payload schemas,
request deduplication and inherited-versus-new accounting semantics for forks.

This extension is part of the unpublished session format candidate. The older
5cc93a3 development snapshot does not implement annotation control semantics and
must not be used to load files created by this candidate.

## Attachment references (unpublished development format)

New persistent message/tool-result images store `attachment: {sha256, size}`
instead of base64 `data`. Blobs live in the agent session directory's private
`.attachments` directory. Legacy inline data remains readable. Image MIME type
and unknown payload fields remain in JSONL. AppendContext publishes and syncs
blobs before the referring record; ResolveImages verifies and hydrates them for
model replay. Missing/corrupt references are errors. In-memory sessions retain
inline data. This extension requires the corresponding runtime; older loaders
must not be used for image replay on these unpublished candidates.

CopyAttachments publishes only referenced blobs into a target store before its
session JSONL is published. Portable JSONL copies must travel together with their
sibling `.attachments` directory. Copying JSONL alone is insufficient when it
contains references. Blob quotas do not evict existing session data.

MigrateLegacy also externalizes legacy inline images, including image records
larger than the normal reader limit. Its migration-only record cap is 64 MiB,
within the existing 128 MiB total import cap. Each blob must fit the attachment
store's 32 MiB limit. Normalized output still passes the unchanged 10 MiB record
reader and strict graph/duplicate-ID validation before replacement. The backup
and reported source digest refer to the original unmodified bytes. The report's
ExternalizedImages count describes transformed image occurrences. Repeated
migration of reference-backed records leaves bytes and identity unchanged.
Invalid image data and oversized non-image records remain errors. A failed
conversion can leave unreferenced immutable blobs; it never rewrites the source.
Combined recovery is available through the explicit MigrateRecovering API.

## Combined recovery and migration

MigrateLegacy remains strict about unfinished tails. MigrateRecovering may
remove only a syntactically unfinished final JSON object, preserving a full
original backup. It converts and validates the complete retained prefix under
one lifetime writer lease, including oversized inline images and recognised
historical compaction clones. Interior corruption, terminated malformed records
and invalid retained graphs remain errors with no source replacement.
TailRecovered and DiscardedTailBytes are set only after replacement; Migrated
still indicates replacement when a later directory sync fails. The source hash
covers the full original including the discarded tail. Restart is idempotent.

### Listing metadata

`Store.List` scans records up to the legacy image import record limit and checks
scan/read errors. Each `SessionInfo.MetadataError` is empty only when that file's
metadata scan completed. On failure the key stays discoverable for explicit
recovery, but ID and entry count are cleared; timestamps use file modification
time when available. Non-regular files are reported without opening them.
Listing extracts syntactic metadata; it does not validate the session graph or
replace leased loading/migration. Callers must not interpret a metadata error as
an empty, valid session.
