#!/usr/bin/env sh
conda list --explicit > conda-spec/spec-file.txt
conda env export --from-history > conda-spec/export-history.yml
conda env export > conda-spec/export.yml
