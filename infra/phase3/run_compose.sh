#!/usr/bin/env bash
# VM-only Compose entry point. Do not invoke docker compose directly: this
# validates the reviewed, digest-qualified Ollama image before startup.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="$script_dir/compose.env"

if [[ "${1:-}" == "--env-file" ]]; then
    if [[ $# -lt 2 ]]; then
        echo "--env-file requires a path" >&2
        exit 2
    fi
    env_file="$2"
    shift 2
fi

if [[ ! -r "$env_file" ]]; then
    echo "Compose environment file is not readable: $env_file" >&2
    exit 2
fi

# Do not try to reproduce Compose's permissive dotenv grammar. This file has a
# deliberately narrower contract: comments/blank lines plus one literal image
# assignment. In particular, reject colon delimiters, whitespace variants,
# quotes, interpolation, and any later override before Compose can parse them.
image=""
found_image=0
while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^[[:space:]]*$ || "$line" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    if [[ "$line" =~ ^CAUSALOPS_OLLAMA_IMAGE= ]]; then
        if [[ "$found_image" == "1" ]]; then
            echo "Compose environment must define CAUSALOPS_OLLAMA_IMAGE exactly once" >&2
            exit 2
        fi
        image="${line#CAUSALOPS_OLLAMA_IMAGE=}"
        found_image=1
        continue
    fi
    echo "Compose environment may contain only a literal CAUSALOPS_OLLAMA_IMAGE= assignment" >&2
    exit 2
done < "$env_file"

if [[ "$found_image" != "1" ]]; then
    echo "Compose environment must define CAUSALOPS_OLLAMA_IMAGE exactly once" >&2
    exit 2
fi

if ! LC_ALL=C grep -Eq '^[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[a-f0-9]{64}$' <<<"$image"; then
    echo "CAUSALOPS_OLLAMA_IMAGE must be an immutable @sha256: digest reference" >&2
    exit 2
fi

if [[ "${1:-}" == "--validate" ]]; then
    if [[ $# -ne 1 ]]; then
        echo "--validate does not accept Compose arguments" >&2
        exit 2
    fi
    exit 0
fi

if [[ "${CAUSALOPS_EXECUTION_ENV:-}" != "vm" ]]; then
    echo "Phase 3 Compose may run only with CAUSALOPS_EXECUTION_ENV=vm" >&2
    exit 2
fi

for argument in "$@"; do
    case "$argument" in
        -f | -f?* | --file | --file=* | --env-file | --env-file=* | \
            --project-directory | --project-directory=*)
            echo "Compose configuration-source override is not permitted: $argument" >&2
            exit 2
            ;;
    esac
done

# Shell variables take precedence over --env-file during Compose interpolation.
# Remove inherited configuration sources so the reviewed file and Compose file
# above are the only sources that can affect the image definition.
unset CAUSALOPS_OLLAMA_IMAGE COMPOSE_FILE COMPOSE_ENV_FILES
exec docker compose --env-file "$env_file" -f "$script_dir/docker-compose.yml" "$@"
