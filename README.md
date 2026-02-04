# SINGER: An Onboard Generalist Vision-Language Navigation Policy for Drones

## Installation

SINGER requires [FiGS-Standalone](https://github.com/StanfordMSL/FiGS-Standalone) as its base environment.

### Prerequisites

- Docker with NVIDIA Container Toolkit
- FiGS-Standalone repository (for the base Docker image)

### 1) Clone repositories

```bash
git clone https://github.com/StanfordMSL/FiGS-Standalone.git
git clone https://github.com/StanfordMSL/SINGER.git
```

### 2) Build the FiGS base image (one-time setup)

```bash
cd FiGS-Standalone
docker-compose build
```

### 3) Run SINGER development environment

```bash
cd SINGER
docker-compose run singer
```

This will:
- Use the `figs:latest` base image
- Mount SINGER and FiGS-Standalone for editable development
- Auto-install packages on first run
- Persist installed packages across container restarts

### Configuration

You can customize paths via environment variables or a `.env` file:

```bash
# Default values
FIGS_PATH=../FiGS-Standalone
DATA_PATH=/media/admin/data/StanfordMSL/nerf_data
```

### Rebuilding

After changes to FiGS-Standalone's Dockerfile:

```bash
cd FiGS-Standalone
docker-compose build

# Clear SINGER's cached packages
cd ../SINGER
docker-compose down -v
docker-compose run singer
```
