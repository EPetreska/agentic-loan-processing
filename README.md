# Agentic BPM: loan applications

## Known limitations

- The bank's workflow (which holds the database password) and the agent (which holds only its case's token) run in one program, `run.py`, so they are kept apart by the code rather than by separate services; in deployment each becomes its own service with its own database user and only the rights it needs.
