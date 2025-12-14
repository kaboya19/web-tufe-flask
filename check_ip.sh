#!/bin/bash
# Sunucu IP adresini kontrol et

echo "Sunucu IP Adresleri:"
echo "==================="
echo ""

# Public IP (internet üzerinden görünen IP)
echo "Public IP:"
curl -s ifconfig.me
echo ""
echo ""

# Local IP (sunucu içindeki IP)
echo "Local IP:"
hostname -I | awk '{print $1}'
echo ""

# Alternatif public IP kontrolü
echo "Alternatif Public IP kontrolü:"
curl -s ipinfo.io/ip
echo ""


