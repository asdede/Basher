# Basher
Fedora's Terminal AI helper.

## Käynnistys

Juurikansiossa `docker compose up --build`

## Riippuvuudet yms.

- Docker 
- Docker compose
- Linux

## Käyttö

### CLI/Terminaali

- Aseta Scripts kansion basher.sh:n polku .bashrc tiedostoon. Lisää sille haluamasi alias.

esim
```bash
#nano ~/.bashrc

alias basher='/Basher/Scripts/basher.sh'

# lisää polku exporttiin

export PATH=/Basher/Scripts:$PATH
```

Sen jälkeen tätä voidaan käyttää suoraa terminaalista komennolla `basher hello worrrld`

#### Komennot
*Kopioitu suoraan basher --help*
```
SYNOPSIS
  basher.sh [OPTIONS] [QUESTION...]
  basher.sh [OPTIONS]
  echo "QUESTION" | basher.sh [OPTIONS]

DESCRIPTION
  Sends your question to the Basher API and prints the assistant's response.
  You can pass the question as arguments (no quotes needed) or via stdin.

OPTIONS (planned; currently only --help is implemented)
  --help    Show this help and exit.
  --start   Starts the basher containers.
  --stop    Stops the basher containers.
  --down    Destroys the basher containers.
  --status  Show whether the API is running and healthy.

ENVIRONMENT
  BASHER_API_URL            Base URL of the API (default: http://localhost:8000).
  BASHER_DEBUG              If set, show context and retrieval info (e.g. BASHER_DEBUG=1).
  BASHER_USE_GLOW           If set and glow is installed, render markdown with glow.
  BASHER_CONNECT_TIMEOUT    Seconds to wait for connection (default: 5).
  BASHER_HEALTH_MAX_TIME    Max seconds for health check (default: 10).
  BASHER_ASK_MAX_TIME       Max seconds for /ask request (default: 300).

EXAMPLES
  basher.sh how do I list files
  basher.sh "one quoted question"
  echo "how do I use grep?" | basher.sh
  BASHER_DEBUG=1 basher.sh explain bash arrays

EXIT CODES
  0   Success.
  1   Usage error, API error, no response, or invalid JSON.
  2   API unreachable or not healthy (connection failed or health check failed).
  130 Script interrupted (e.g. Ctrl+C).


Usage: basher.sh [OPTIONS] your question here
   or: echo "Your question" | basher.sh [OPTIONS]

Options:
  --help   Show this help.
  --start  Starts the basher containers.
  --stop   Stops the basher containers.
  --down   Destroys the basher containers.
  --status Show API status.
```

### UI (electronjs)

Tulossa