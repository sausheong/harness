# Session attachment storage

`Open(directory, quotaBytes)` opens a private (0700) content store. Its parent directory must already exist;
the store directory and its parent publication are synced before use. Use
`DefaultStoreBytes` for a 1 GiB quota. `Put(ctx, data)` publishes a synced 0600
blob and returns a `Ref` containing its lowercase SHA-256 digest and exact byte
length. `Read(ctx, ref)` verifies regular-file identity, size and digest. Blobs
are limited to 32 MiB; MIME type belongs in the referring session record.

Publication uses a synced temporary file, exclusive hard link and directory
sync. A cross-process lock serialises deduplication, quota admission and
publication on Linux/macOS. Interrupted pending files are removed under that
lock on a later new-content put. Existing published attachments are never
evicted. A full store fails admission. Directory scanning is bounded to 8192
entries, including control files; oversized stores require explicit maintenance.

References contain no caller-chosen filesystem path. File operations use
`os.Root`; external symlinks cannot escape that root. The caller owns the
configured private root and must not replace it with an untrusted directory.
Do not close the store until its operations have joined. Cancellation interrupts
lock waits and chunked I/O; the caller still owns its input byte slice until Put
returns. Other platforms reject writer locking explicitly.

This package is the storage foundation. Session image references, runtime
hydration, legacy inline-image conversion, fork copying and portable export
packaging must be integrated before claiming attachment workflow completion.
