# Native qualification fixtures

Qualification requires a working local Docker daemon with a Unix socket and
permission for the runner user to use it. The workflow prepares fixtures before
running tests and fails if the daemon is unavailable. Skipped container tests
are never accepted. Linux uses the installed Docker service. Hosted macOS uses
macos-15-intel with Colima, following Colima's own integration runner choice:
https://github.com/abiosoft/colima/blob/main/.github/workflows/macos-integration.yml
The workflow starts an ephemeral daemon on that hosted runner; it does not
register or expose a developer's machine as a self-hosted runner.

The workflow pulls the named fixture base versions, resolves platform-specific
immutable image IDs and records them in evidence/fixtures.json before tests.
These IDs freeze one run; tags are not cross-run immutable references. To reproduce
an earlier run, load its recorded image IDs and invoke the helper directly with
those IDs instead of resolving current tags.

prepare_container_fixtures.py accepts only loaded immutable IDs, checks Linux and
matching image architectures, then creates a minimal implicit-volume negative
fixture by creating (never starting) a network-disabled container and committing
its VOLUME metadata. This avoids build-engine differences and registry lookups.
The temporary container is removed in a finally block. It uses a private Docker configuration
and explicit socket. The generated fixtures.env is suitable for GITHUB_ENV; it
contains image IDs and a local socket path, never provider credentials.

The helper retains images for the caller's test run and does not prune shared
Docker state. Native test cases own and remove their individual containers.
