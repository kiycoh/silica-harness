# Commercial Licensing

Silica is dual-licensed. It is available as open source under the
[GNU Affero General Public License v3.0 or later](LICENSE). Organizations that
cannot, or choose not to, comply with the AGPL terms may license the software
under commercial terms directly from the copyright holder.

This document outlines the commercial licensing policy. It does not constitute a
license agreement on its own: the commercial license is a separate written agreement
between your organization and the copyright holder. The AGPL applies until that
agreement is executed.

## When the AGPL is sufficient

You do not need a commercial license, and owe no licensing fees, if you:

- Run Silica on your own infrastructure (for individual or internal organizational
  use), modified or unmodified, without external distribution;
- Integrate it with an agent harness (Claude Code, Codex, or any MCP client) as a
  personal or internal workflow memory engine;
- Retain the notes it generates. The AGPL governs the software, not its output:
  your vault contents remain entirely your property;
- Contribute to the project under the contributor terms specified in
  [CONTRIBUTING.md](CONTRIBUTING.md#license).

## When you need a commercial license

A commercial license is required if you wish to:

1. Embed Silica (or a derivative work) within a proprietary product or closed-source
   application without distributing the corresponding source code under the AGPL;
2. Host or run a modified version of Silica as a network service for external users
   without providing access to the modified source code, as required by
   [AGPL Section 13](LICENSE);
3. Obtain enterprise terms not provided by the AGPL: commercial warranties,
   indemnification, service level agreements (SLAs), or dedicated support.

A commercial license grants exemptions from the copyleft obligations of the AGPL for
your specific use case. It does not diminish any rights you have under the AGPL.

## Scope of coverage

The commercial license covers the core Silica engine: the `silica-harness` package
and the contents of this repository. Third-party dependencies retain their respective
licenses. Companion repositories (such as the Obsidian plugin) are governed by their
own licensing terms.

## Terms and Pricing

Commercial licenses are granted per organization and priced based on deployment scope
(e.g., seats, active instances, or distribution tier).

To request a commercial license, contact **alessandrocarosia@proton.me** with an
overview of your use case, architectural integration, and projected scale. We will
provide a quote along with the standard commercial agreement.

## Dual-Licensing Rationale

The copyleft license guarantees that the open-source version remains free and accessible
to the community, while commercial licensing funds continuous maintenance and development.
Silica is maintained under single copyright ownership with contributor grants
([CONTRIBUTING.md](CONTRIBUTING.md#license)), enabling compliant dual licensing.
