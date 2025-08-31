BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

-- 정리
DROP VIEW IF EXISTS vw_training_activity CASCADE;
DROP TABLE IF EXISTS tb_mr_proposed_hist CASCADE;
DROP TABLE IF EXISTS tb_mr_proposed CASCADE;
DROP TABLE IF EXISTS bundle_predictions CASCADE;
DROP TABLE IF EXISTS model_versions CASCADE;
DROP TABLE IF EXISTS bundle_rules CASCADE;
DROP TABLE IF EXISTS bundle_items CASCADE;
DROP TABLE IF EXISTS bundle_sets CASCADE;
DROP TABLE IF EXISTS activity_codes CASCADE;
DROP TABLE IF EXISTS tb_por_detail CASCADE;
DROP TABLE IF EXISTS tb_mr CASCADE;

-- 활동 코드 마스터 (정답 label 사전)
CREATE TABLE activity_codes (
  actocode   VARCHAR NOT NULL,
  actno      VARCHAR NOT NULL,
  name       TEXT,
  active     BOOLEAN DEFAULT TRUE,
  PRIMARY KEY (actocode, actno)
);

-- 과거 정답 (1:1 도 있고 1:n 도 허용)
-- 핵심 복합키 pjtno/porser/porseq/revno + 활동 코드
CREATE TABLE tb_mr (
  mr_id      BIGSERIAL PRIMARY KEY,
  pjtno      VARCHAR NOT NULL,
  porser     VARCHAR NOT NULL,
  porseq     VARCHAR NOT NULL,
  revno      VARCHAR NOT NULL,
  actocode   VARCHAR NOT NULL,
  actno      VARCHAR NOT NULL,
  category_name TEXT,
  UNIQUE (pjtno, porser, porseq, revno, actocode, actno),
  FOREIGN KEY (actocode, actno) REFERENCES activity_codes (actocode, actno)
);

-- 원천 POR 라인 (복합키 + 라인)
CREATE TABLE tb_por_detail (
  pjtno     VARCHAR NOT NULL,
  porser    VARCHAR NOT NULL,
  porseq    VARCHAR NOT NULL,
  revno     VARCHAR NOT NULL,
  line_no   INT     NOT NULL,
  item_name TEXT,
  spec_text TEXT,
  -- 분류 규칙에 영향을 주는 속성
  mccsno    TEXT,
  block     TEXT,
  event     TEXT,
  sign      TEXT,          -- '+' 또는 '-'
  duration  INT,
  PRIMARY KEY (pjtno, porser, porseq, revno, line_no)
);
CREATE INDEX idx_por_hdr ON tb_por_detail (pjtno, porser, porseq, revno);

-- 학습/예측 번들 저장 (정답/예측 셋)
CREATE TABLE bundle_sets (
  id            BIGSERIAL PRIMARY KEY,
  pjtno         VARCHAR NOT NULL,
  porser        VARCHAR NOT NULL,
  porseq        VARCHAR NOT NULL,
  revno         VARCHAR NOT NULL,
  -- 정답 라벨(학습셋) 또는 예측 라벨(옵션 persist 시)
  label_actocode VARCHAR,
  label_actno    VARCHAR,
  set_embed     VECTOR(512),
  created_at    TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_bundle_sets_hdr ON bundle_sets (pjtno, porser, porseq, revno);
CREATE INDEX idx_bundle_sets_label ON bundle_sets (label_actocode, label_actno);
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

-- 규칙: 룰로 특정 활동 코드 분류(헤더 속성까지 반영 가능)
CREATE TABLE bundle_rules (
  rule_id      BIGSERIAL PRIMARY KEY,
  pattern      TEXT NOT NULL,            -- 텍스트 정규식 (아이템 집합 문자열)
  target_actocode VARCHAR NOT NULL,
  target_actno    VARCHAR NOT NULL,
  priority     INT NOT NULL DEFAULT 0,
  enabled      BOOLEAN DEFAULT TRUE,
  -- 헤더 속성 조건(선택): null 이면 무시
  where_mccsno TEXT,
  where_block  TEXT,
  where_event  TEXT,
  where_sign   TEXT,
  where_duration_min INT,
  where_duration_max INT,
  FOREIGN KEY (target_actocode, target_actno) REFERENCES activity_codes(actocode, actno)
);

-- 모델 버전 (학습 메타)
CREATE TABLE model_versions (
  version_id BIGSERIAL PRIMARY KEY,
  trained_at TIMESTAMP DEFAULT NOW(),
  details    JSONB
);

-- 예측 로그 (항상 append)
CREATE TABLE bundle_predictions (
  id              BIGSERIAL PRIMARY KEY,
  pjtno           VARCHAR NOT NULL,
  porser          VARCHAR NOT NULL,
  porseq          VARCHAR NOT NULL,
  revno           VARCHAR NOT NULL,
  predicted_actocode VARCHAR NOT NULL,
  predicted_actno    VARCHAR NOT NULL,
  confidence      REAL,
  top3            JSONB,
  model_version   BIGINT REFERENCES model_versions(version_id),
  explain         JSONB,      -- rule_hit / knn / model 비중 등
  predicted_at    TIMESTAMP DEFAULT NOW()
);

-- 제안 결과(헤더 단위 + 활동 단위로 보관: 복합 PK)
CREATE TABLE tb_mr_proposed (
  pjtno       VARCHAR NOT NULL,
  porser      VARCHAR NOT NULL,
  porseq      VARCHAR NOT NULL,
  revno       VARCHAR NOT NULL,
  actocode    VARCHAR NOT NULL,
  actno       VARCHAR NOT NULL,
  category_name TEXT,
  confidence  REAL,
  top3        JSONB,
  model_version BIGINT,
  predicted_at TIMESTAMP DEFAULT NOW(),
  PRIMARY KEY (pjtno, porser, porseq, revno, actocode, actno)
);

-- (옵션) 제안 이력
CREATE TABLE tb_mr_proposed_hist (
  id BIGSERIAL PRIMARY KEY,
  pjtno       VARCHAR NOT NULL,
  porser      VARCHAR NOT NULL,
  porseq      VARCHAR NOT NULL,
  revno       VARCHAR NOT NULL,
  actocode    VARCHAR NOT NULL,
  actno       VARCHAR NOT NULL,
  category_name TEXT,
  confidence  REAL,
  top3        JSONB,
  model_version BIGINT,
  predicted_at TIMESTAMP DEFAULT NOW()
);

-- 학습 뷰: 과거 tb_mr(정답) + tb_por_detail 묶음
-- 기본 패턴(헤더 키로 조인, 모든 라인을 묶음)
CREATE OR REPLACE VIEW vw_training_activity AS
SELECT
  m.pjtno, m.porser, m.porseq, m.revno,
  m.actocode, m.actno,
  ARRAY_AGG(CONCAT_WS(' ', d.item_name, d.spec_text) ORDER BY d.line_no) AS items
FROM tb_mr m
JOIN tb_por_detail d
  ON d.pjtno=m.pjtno AND d.porser=m.porser AND d.porseq=m.porseq AND d.revno=m.revno
GROUP BY m.pjtno, m.porser, m.porseq, m.revno, m.actocode, m.actno;

-- 시드 (예시)
INSERT INTO activity_codes(actocode, actno, name, active) VALUES
  ('ACT','001','MECHANICAL',TRUE),
  ('ACT','002','PIPING',TRUE),
  ('ACT','003','ELECTRIC',TRUE),
  ('ACT','004','INSTRUMENT',TRUE),
  ('ACT','005','STEEL',TRUE)
ON CONFLICT DO NOTHING;

-- 더미 정답/원천
INSERT INTO tb_mr(pjtno,porser,porseq,revno,actocode,actno,category_name) VALUES
('P1','S1','Q1','R1','ACT','001','MECHANICAL'),
('P2','S1','Q1','R1','ACT','002','PIPING'),
('P3','S1','Q1','R1','ACT','003','ELECTRIC'),
('P4','S1','Q1','R1','ACT','004','INSTRUMENT'),
('P5','S1','Q1','R1','ACT','005','STEEL')
ON CONFLICT DO NOTHING;

INSERT INTO tb_por_detail(pjtno,porser,porseq,revno,line_no,item_name,spec_text,mccsno,block,event,sign,duration) VALUES
('P1','S1','Q1','R1',1,'Pump ABC','40mm flange, Ø40mm, gasket included','M1','B1','E1','+',10),
('P1','S1','Q1','R1',2,'Motor AC','5kW 220V, PN16','M1','B1','E1','+',10),
('P1','S1','Q1','R1',3,'Coupling','DN50, steel','M1','B1','E1','+',10),
('P1','S1','Q1','R1',4,'Gasket','fiber 40mm','M1','B1','E1','+',10),

('P2','S1','Q1','R1',1,'Pipe SCH40','DN50 6m length','M2','B1','E2','+',20),
('P2','S1','Q1','R1',2,'Flange WN','PN16, 50mm','M2','B1','E2','+',20),
('P2','S1','Q1','R1',3,'Valve Gate','DN50, SS','M2','B1','E2','+',20),
('P2','S1','Q1','R1',4,'Gasket Spiral','Ø50mm','M2','B1','E2','+',20),

('P3','S1','Q1','R1',1,'Power Cable','3C x 2.5mm','M3','B2','E3','-',30),
('P3','S1','Q1','R1',2,'Breaker MCCB','AC 220V 40A','M3','B2','E3','-',30),
('P3','S1','Q1','R1',3,'Plug','industrial, 3P','M3','B2','E3','-',30),
('P3','S1','Q1','R1',4,'Panel','IP54 enclosure','M3','B2','E3','-',30),

('P4','S1','Q1','R1',1,'Pressure Gauge','0~10 bar, 1/2 inch','M4','B3','E4','+',25),
('P4','S1','Q1','R1',2,'Thermometer','-20~120°C, DN20','M4','B3','E4','+',25),
('P4','S1','Q1','R1',3,'Transmitter','4-20mA, PN16','M4','B3','E4','+',25),
('P4','S1','Q1','R1',4,'Manifold','3-way, SS316','M4','B3','E4','+',25),

('P5','S1','Q1','R1',1,'Plate','SS400, 10mm','M5','B3','E5','+',15),
('P5','S1','Q1','R1',2,'Angle','L50x50x5','M5','B3','E5','+',15),
('P5','S1','Q1','R1',3,'Channel','C100x50x5','M5','B3','E5','+',15),
('P5','S1','Q1','R1',4,'Beam','H200x100x5.5x8','M5','B3','E5','+',15);

-- 룰 예시 (헤더 조건 포함)
INSERT INTO bundle_rules(pattern,target_actocode,target_actno,priority,enabled,where_mccsno,where_event)
VALUES
('(?i)(?=.*pump)(?=.*motor)','ACT','001',10,TRUE,'M1',NULL),
('(?i)(pipe|valve|flange|gasket)','ACT','002',8,TRUE,'M2',NULL),
('(?i)(cable|breaker|panel|plug)','ACT','003',8,TRUE,'M3',NULL),
('(?i)(gauge|thermometer|transmitter)','ACT','004',8,TRUE,'M4',NULL),
('(?i)(plate|angle|channel|beam)','ACT','005',8,TRUE,'M5',NULL)
ON CONFLICT DO NOTHING;

COMMIT;
