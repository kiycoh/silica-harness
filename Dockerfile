# One image, three processes. ENTRYPOINT is the program, the argument picks the
# lane: no argument opens the REPL, `--gui` serves the web UI, `mcp` speaks the
# stdio protocol. Splitting this into three Dockerfiles would be three copies of
# the same pip install, drifting apart at the first dependency bump.

# Build stage exists for one reason: setuptools-scm shells out to git, and the
# runtime image has no git. Here the wheel gets a real version; there it does
# not need the 115 MB of history to install one.
FROM python:3.11-slim AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
# The escape hatch for a context without .git (see .dockerignore): declared
# so setuptools-scm can read it as an env var during the wheel build.
ARG SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SILICA_HARNESS
RUN pip wheel --no-deps --no-cache-dir . -w /wheels

FROM python:3.11-slim
# Path.home() resolves through os.path.expanduser, which reads $HOME — so this
# single line relocates all ~14 hardcoded `~/.silica` call sites (ledger.db,
# checkpoints.db, undo_journal.db, index/, runs/, cache/) onto one volume,
# without the codebase needing a SILICA_HOME of its own. Set explicitly and not
# inherited: under `--user 1000:1000` with no /etc/passwd entry, an unset HOME
# sends expanduser to the pwd module and it raises.
ENV HOME=/data \
    PYTHONUNBUFFERED=1
COPY --from=build /wheels /wheels
# --find-links without --no-index: the local wheel wins for silica-harness
# itself, every dependency still resolves from PyPI. Installing by name (rather
# than by wheel path) is what lets the extras be named here.
RUN pip install --no-cache-dir --find-links=/wheels "silica-harness[gui,mcp]" \
 && rm -rf /wheels \
 && mkdir -p /data /vault
# Not installed: git, ffmpeg, soffice, whisper-cli, yt-dlp, mineru. Every one is
# reached through shutil.which and its absence disables that converter lane
# instead of crashing — `silica doctor` reports which ones are missing. Add them
# to this image only for the lanes you actually run.
WORKDIR /vault
ENTRYPOINT ["silica"]
