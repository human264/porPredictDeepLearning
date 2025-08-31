# scripts/curl-examples.sh
#!/usr/bin/env bash
set -e

# 1) 규칙 업서트
curl -X POST http://localhost:8000/bundle_rules \
  -H "Content-Type: application/json" \
  -d '{"pattern":"(?i)(?=.*pump)(?=.*motor)","target_label":"B5","priority":10,"enabled":true}'

# 2) 학습
curl -X POST http://localhost:8000/bundle/train

# 3) 예측 (한 묶음씩)
curl -s -X POST http://localhost:8000/bundle/predict \
  -H "Content-Type: application/json" \
  -d '{"por_id": 20250001, "items": [
        "Pump ABC 40mm flange",
        "Motor AC 5kW 220V",
        "Bolt M12 30mm",
        "Gasket 40mm fiber"
      ]}' | jq

# 4) 통계
curl http://localhost:8000/stats | jq
