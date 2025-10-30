# Base Ollama image
FROM ollama/ollama:latest

# (Optional) Pre-pull nomic-embed-text model at build time
# This ensures the model is ready as soon as the container starts
RUN (ollama serve & sleep 5) && ollama pull nomic-embed-text && pkill ollama || true

# Expose the default Ollama port (Render replaces it with $PORT automatically)
EXPOSE 11434

# Serve Ollama
CMD ["ollama", "serve"]
