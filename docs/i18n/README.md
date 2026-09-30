# Documentation i18n (internationalization)

This directory holds translated copies of the project documentation so we can
offer the README and key guides in more than one language. It is a **starter
scaffold**: the goal is a simple, low-friction convention for adding a locale,
not a fully automated translation pipeline.

## Layout

```
docs/i18n/
├── README.md            # this file — the convention & workflow
├── languages.yml        # registry of locales (single source of truth)
├── new_locale.py        # helper that scaffolds a new locale folder
└── <lang>/              # one folder per locale, ISO 639-1 code
    └── README.<lang>.md # translated README (partial is fine to start)
```

A locale folder is named with an
[ISO 639-1](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)
code (`es`, `fr`, `pt`, `zh`, …). Translated files mirror the source filename and
append the locale suffix before the extension, e.g. `README.md` →
`README.es.md`. Keeping the source name stable makes diffs against the English
original trivial.

## Adding a locale

1. Register it in [`languages.yml`](./languages.yml) (code, English name, native
   name, and the directory name to use).
2. Scaffold the folder + a README stub:

   ```bash
   # from the repository root
   python docs/i18n/new_locale.py fr
   ```

   This creates `docs/i18n/fr/` and copies the README template into
   `docs/i18n/fr/README.fr.md` with untranslated headings so a contributor can
   fill it in incrementally.
3. Translate what you can. A **partial** locale is explicitly acceptable — it is
   how a new language gets started. Leave untranslated sections in English; do
   not delete them.
4. Open a PR. Reviewers check that the code, folder name, and file name all
   match `languages.yml`.

## Proof-of-workflow locale

Spanish (`es`) ships as the first partial locale
([`docs/i18n/es/README.es.md`](./es/README.es.md)). It translates the README
title and the intro/feature sections and leaves the rest as a to-do, showing the
exact pattern a contributor should follow.

## Building / serving translated docs

The repo currently builds docs with **Sphinx** in CI (see
[`docs/Makefile`](../Makefile), `make html` → `docs/_build/html`) and has an
optional **MkDocs** config ([`docs/mkdocs.yml`](../mkdocs.yml)). Neither build is
wired for language switching yet, and this scaffold deliberately does **not**
change either build so nothing breaks. When you are ready to serve locales, pick
one path:

### Option A — MkDocs + mkdocs-material i18n plugin

Add to `docs/requirements.txt`:

```
mkdocs
mkdocs-material
mkdocs-static-i18n
```

Enable the plugin in `docs/mkdocs.yml`:

```yaml
plugins:
  - search
  - i18n:
      languages:
        - locale: en
          default: true
        - locale: es
          name: Español
          build: true
```

`mkdocs-static-i18n` discovers `README.es.md` alongside `README.md` automatically,
so the folder convention above is what it expects.

### Option B — Sphinx `sphinx-intl` / gettext

Add `sphinx-intl` and drive it with the gettext builder:

```bash
pip install sphinx-intl
make -C docs gettext          # extract translatable strings to docs/_build/gettext
sphinx-intl update -p docs/_build/gettext -l es   # create docs/locales/es/LC_MESSAGES
```

Sphinx's gettext flow and the `docs/i18n/` folder are complementary: keep the
free-form Markdown translations here for humans, and use gettext for
auto-generated API docs.

## Contributing notes

- Never machine-translate and merge without human review.
- Keep code blocks, shell commands, links, and identifiers **identical** to the
  English source; translate only prose.
- Update the `progress` field in `languages.yml` as a rough coverage signal
  (`stub` → `partial` → `complete`).
