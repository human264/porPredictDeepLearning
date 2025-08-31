-- setup_with_dummy.sql
-- PostgreSQL + pgvector: 테이블 생성 + 더미데이터 + 학습 뷰
-- 실행은 docker-compose가 자동 수행
BEGIN;

-- 확장
CREATE EXTENSION IF NOT EXISTS vector;

-- 안전 재실행을 위한 정리
DROP VIEW IF EXISTS vw_mr_training CASCADE;

DROP TABLE IF EXISTS tb_mr_proposed CASCADE;
DROP TABLE IF EXISTS bundle_predictions CASCADE;
DROP TABLE IF EXISTS model_versions CASCADE;
DROP TABLE IF EXISTS bundle_rules CASCADE;
DROP TABLE IF EXISTS bundle_items CASCADE;
DROP TABLE IF EXISTS bundle_sets CASCADE;
DROP TABLE IF EXISTS category_alias_b CASCADE;
DROP TABLE IF EXISTS categories_b CASCADE;

DROP TABLE IF EXISTS tb_por_detail CASCADE;
DROP TABLE IF EXISTS tb_mr CASCADE;

-- 카테고리 B 마스터
CREATE TABLE categories_b (
  code   VARCHAR PRIMARY KEY,
  name   TEXT NOT NULL,
  active BOOLEAN DEFAULT TRUE
);

-- (선택) 별칭
CREATE TABLE category_alias_b (
  alias TEXT PRIMARY KEY,
  category_code VARCHAR NOT NULL REFERENCES categories_b(code)
);

-- 과거 정답/원천 (사용자 스키마 힌트 기반)
CREATE TABLE tb_mr (
  id            BIGSERIAL PRIMARY KEY,
  por_id        BIGINT UNIQUE NOT NULL,
  category_code VARCHAR NOT NULL REFERENCES categories_b(code),
  category_name TEXT
);

CREATE TABLE tb_por_detail (
  por_id    BIGINT NOT NULL,
  line_no   INT    NOT NULL,
  item_name TEXT,
  spec_text TEXT,
  mr_id     BIGINT REFERENCES tb_mr(id) ON UPDATE CASCADE ON DELETE SET NULL,
  PRIMARY KEY (por_id, line_no)
);

CREATE INDEX idx_tb_por_detail_por ON tb_por_detail(por_id);
CREATE INDEX idx_tb_mr_por ON tb_mr(por_id);

-- 내부 학습/예측 저장
CREATE TABLE bundle_sets (
  id         BIGSERIAL PRIMARY KEY,
  label      VARCHAR NOT NULL REFERENCES categories_b(code),
  set_embed  VECTOR(512),
  created_at TIMESTAMP DEFAULT NOW()
);

DROP INDEX IF EXISTS idx_bundle_sets_set_embed;
CREATE INDEX idx_bundle_sets_set_embed
  ON bundle_sets
  USING ivfflat (set_embed vector_cosine_ops)
  WITH (lists = 100);

CREATE TABLE bundle_items (
  id         BIGSERIAL PRIMARY KEY,
  set_id     BIGINT NOT NULL REFERENCES bundle_sets(id) ON DELETE CASCADE,
  raw_text   TEXT NOT NULL,
  clean_text TEXT NOT NULL,
  item_embed VECTOR(512)
);

-- 규칙
CREATE TABLE bundle_rules (
  pattern      TEXT PRIMARY KEY,
  target_label VARCHAR NOT NULL REFERENCES categories_b(code),
  priority     INT NOT NULL DEFAULT 0,
  enabled      BOOLEAN DEFAULT TRUE
);

-- 모델 버전/예측 로그
CREATE TABLE model_versions (
  version_id BIGSERIAL PRIMARY KEY,
  trained_at TIMESTAMP DEFAULT NOW(),
  details    JSONB
);

CREATE TABLE bundle_predictions (
  id              BIGSERIAL PRIMARY KEY,
  set_id          BIGINT REFERENCES bundle_sets(id),
  predicted_label VARCHAR NOT NULL,
  confidence      REAL,
  top3            JSONB,
  model_version   BIGINT REFERENCES model_versions(version_id),
  predicted_at    TIMESTAMP DEFAULT NOW()
);

-- (선택) 운영 반영 전 제안 테이블
CREATE TABLE tb_mr_proposed (
  por_id         BIGINT PRIMARY KEY,
  category_code  VARCHAR NOT NULL REFERENCES categories_b(code),
  category_name  TEXT,
  confidence     REAL,
  top3           JSONB,
  model_version  BIGINT,
  predicted_at   TIMESTAMP DEFAULT NOW()
);

-- 시드 카테고리/별칭
INSERT INTO categories_b(code, name, active) VALUES
  ('B1','PIPING',TRUE),
  ('B2','ELECTRIC',TRUE),
  ('B3','STEEL',TRUE),
  ('B4','INSTRUMENT',TRUE),
  ('B5','MECHANICAL',TRUE)
ON CONFLICT (code) DO NOTHING;

INSERT INTO category_alias_b(alias, category_code) VALUES
  ('PIPE','B1'), ('VALVE','B1'),
  ('CABLE','B2'),
  ('PLATE','B3'), ('BEAM','B3'),
  ('GAUGE','B4'),
  ('PUMP','B5')
ON CONFLICT (alias) DO NOTHING;

-- 더미 정답
INSERT INTO tb_mr(por_id, category_code, category_name) VALUES
  (1001, 'B5', 'MECHANICAL'),
  (1002, 'B1', 'PIPING'),
  (1003, 'B2', 'ELECTRIC'),
  (1004, 'B4', 'INSTRUMENT'),
  (1005, 'B3', 'STEEL')
ON CONFLICT (por_id) DO NOTHING;

-- 더미 라인
INSERT INTO tb_por_detail(por_id, line_no, item_name, spec_text) VALUES
 (1001,1,'Pump ABC','40mm flange, Ø40mm, gasket included'),
 (1001,2,'Motor AC','5kW 220V, PN16'),
 (1001,3,'Coupling','DN50, steel'),
 (1001,4,'Gasket','fiber 40mm'),

 (1002,1,'Pipe SCH40','DN50 6m length'),
 (1002,2,'Flange WN','PN16, 50mm'),
 (1002,3,'Valve Gate','DN50, SS'),
 (1002,4,'Gasket Spiral','Ø50mm'),

 (1003,1,'Power Cable','3C x 2.5mm'),
 (1003,2,'Breaker MCCB','AC 220V 40A'),
 (1003,3,'Plug','industrial, 3P'),
 (1003,4,'Panel','IP54 enclosure'),

 (1004,1,'Pressure Gauge','0~10 bar, 1/2 inch'),
 (1004,2,'Thermometer','-20~120°C, DN20'),
 (1004,3,'Transmitter','4-20mA, PN16'),
 (1004,4,'Manifold','3-way, SS316'),

 (1005,1,'Plate','SS400, 10mm'),
 (1005,2,'Angle','L50x50x5'),
 (1005,3,'Channel','C100x50x5'),
 (1005,4,'Beam','H200x100x5.5x8');

-- mr_id 연결(패턴 B 대비)
UPDATE tb_por_detail d
SET mr_id = m.id
FROM tb_mr m
WHERE d.por_id = m.por_id
  AND (d.mr_id IS DISTINCT FROM m.id);

-- 규칙 시드
INSERT INTO bundle_rules(pattern, target_label, priority, enabled) VALUES
  ('(?i)(?=.*pump)(?=.*motor)', 'B5', 10, TRUE),
  ('(?i)(pipe|valve|flange|gasket)', 'B1', 8, TRUE),
  ('(?i)(cable|breaker|panel|plug)', 'B2', 8, TRUE),
  ('(?i)(gauge|thermometer|transmitter)', 'B4', 8, TRUE),
  ('(?i)(plate|angle|channel|beam)', 'B3', 8, TRUE)
ON CONFLICT (pattern) DO NOTHING;

-- 학습 뷰 (패턴 A: tb_mr.por_id 조인)
CREATE OR REPLACE VIEW vw_mr_training AS
SELECT
  m.category_code AS label,
  ARRAY_AGG(CONCAT_WS(' ', d.item_name, d.spec_text) ORDER BY d.line_no) AS items
FROM tb_por_detail d
JOIN tb_mr m ON m.por_id = d.por_id
WHERE m.category_code IS NOT NULL
GROUP BY m.por_id, m.category_code;


CREATE TABLE IF NOT EXISTS tb_mr_proposed_hist (
  id BIGSERIAL PRIMARY KEY,
  por_id BIGINT NOT NULL,
  category_code VARCHAR NOT NULL,
  category_name TEXT,
  confidence REAL,
  top3 JSONB,
  model_version BIGINT,
  predicted_at TIMESTAMP DEFAULT NOW()
);


COMMIT;
