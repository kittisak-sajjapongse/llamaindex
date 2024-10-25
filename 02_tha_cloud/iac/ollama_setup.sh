#!/usr/bin/env bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:70b

# Allow Ollama to serve request from anywhere
OLLAMA_SVR_FILE="/etc/systemd/system/ollama.service"
#sed -i '/^Environment=/ s/"$/;OLLAMA_HOST=0.0.0.0"/' "$OLLAMA_SVR_FILE"
sed -i '/^Environment=/c\Environment="OLLAMA_HOST=0.0.0.0"' "$OLLAMA_SVR_FILE"

sudo systemctl enable ollama.service
sudo systemctl start ollama.service
sudo systemctl daemon-reload
sudo systemctl restart ollama
