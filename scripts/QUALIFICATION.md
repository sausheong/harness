# Native qualification fixtures

Qualification requires a working local Docker daemon with a Unix socket and
permission for the runner user to use it. The workflow prepares fixtures before
running tests and fails if the daemon is unavailable. It does not start or
provision a daemon, and does not accept skipped container tests.

Linux GitHub runners can use their installed Docker service. A macOS runner must
have a Docker-capable environment provisioned before this workflow can succeed.
Do not assume the standard hosted macOS image provides it. See GitHub's runner
limitations: https://docs.github.com/en/actions/reference/runners/larger-runners .
Local macOS qualification remains possible using the explicitly authorised local
daemon; retain exact commit, raw results and platform identity with that evidence.

The workflow pulls the named fixture base versions, resolves platform-specific
immutable image IDs and records them in evidence/fixtures.json before tests.
These IDs freeze one run; tags are not cross-run immutable references. To reproduce
an earlier run, load its recorded image IDs and invoke the helper directly with
those IDs instead of resolving current tags.

prepare_container_fixtures.py accepts only loaded immutable IDs, checks Linux and
matching image architectures, then creates a minimal implicit-volume negative
fixture with build networking disabled. It uses a private Docker configuration
and explicit socket. The generated fixtures.env is suitable for GITHUB_ENV; it
contains image IDs and a local socket path, never provider credentials.

The helper retains images for the caller's test run and does not prune shared
Docker state. Native test cases own and remove their individual containers.
