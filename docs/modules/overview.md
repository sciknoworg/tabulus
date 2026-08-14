# Component Overview

Components are processing modules with explicit file-based contracts.

A component should:

- Run independently from the command line.
- Accept a small set of input paths.
- Write a documented output contract.
- Preserve raw output where useful.
- Avoid assuming a specific upstream library.
- Report errors in a machine-readable way.

The current code is service-oriented. The documentation structure intentionally reframes it as module-oriented so the pipeline can run on a GPU server without Docker and later be containerized from stable modules.
