# =============================================================
# Dockerfile — packages our FastAPI app into a container
#
# WHAT IS DOCKER?
#   Your laptop has Python 3.11, Anaconda, specific versions
#   of every library. Another machine has Python 3.9, different
#   versions. Code breaks.
#
#   Docker fixes this: package your app + exact Python version
#   + exact library versions into ONE image.
#   That image runs identically everywhere.
#
# ANALOGY:
#   Without Docker: give someone a recipe and hope they have
#   the right ingredients and oven.
#   With Docker: give them a ready-made meal in a sealed box.
#   They just heat and eat — guaranteed same result.
#
# LAYERS:
#   Each line in a Dockerfile = one layer
#   Docker caches layers — if requirements.txt didn't change,
#   Docker reuses the cached layer → faster builds
# =============================================================

# Base image — start from official Python 3.11 slim image
# "slim" = minimal OS, no unnecessary tools = smaller image size
# Smaller image = faster to push/pull = cheaper storage
FROM python:3.11-slim

# Set working directory inside the container
# All subsequent commands run from this directory
WORKDIR /app

# Copy requirements first (before code)
# WHY? Docker layer caching:
# If code changes but requirements don't → Docker reuses
# the pip install layer → much faster rebuilds
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir = don't cache pip downloads inside image
#                  keeps image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Expose port 8000 to the outside world
# This is documentation — tells Docker the app uses port 8000
EXPOSE 8000

# Health check — Docker checks this every 30 seconds
# If it fails 3 times → container marked unhealthy → restarted
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Command to run when container starts
# uvicorn = ASGI server (runs FastAPI)
# --host 0.0.0.0 = accept connections from any IP
#   (not just localhost — needed inside containers)
# --port 8000 = listen on port 8000
# --workers 2 = 2 parallel worker processes
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]