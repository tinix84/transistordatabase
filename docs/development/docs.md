# Documentation

How to build and contribute to the documentation.

## Pre-push checks

The pre-push hook checks the README version and builds the MkDocs documentation.
It does not update version text, documentation dates, or files in the checkout.
Update source documents before you commit.

The maintained hook is `scripts/pre-push`. To install it, run these commands from
the repository root. Preserve an existing hook before replacing it.

```bash
hook_path=$(git rev-parse --git-path hooks/pre-push)
if [ -f "$hook_path" ]; then
    hook_backup=$(mktemp /tmp/transistordatabase-pre-push.XXXXXXXX)
    cp "$hook_path" "$hook_backup"
    printf 'Previous hook saved to %s\n' "$hook_backup"
fi
install -m 755 scripts/pre-push "$hook_path"
```

Each build uses a fresh `/tmp/transistordatabase-docs.XXXXXXXX` directory.
The hook reports the output location and retains the build log and generated files.
An absent MkDocs command or a failed build remains a warning and does not block a push.
