BEGIN;

-- 깨끗이 정리
DROP VIEW  IF EXISTS vw_training_item_act CASCADE;
DROP VIEW  IF EXISTS vw_training_item_mr  CASCADE;
DROP TABLE IF EXISTS item_predictions     CASCADE;
DROP TABLE IF EXISTS model_versions       CASCADE;
DROP TABLE IF EXISTS tb_por_detail        CASCADE;
DROP TABLE IF EXISTS tb_mr                CASCADE;
DROP TABLE IF EXISTS activity_codes       CASCADE;

-- 활동 코드 마스터
CREATE TABLE activity_codes (
  actocode   VARCHAR NOT NULL,
  actno      VARCHAR NOT NULL,
  name       TEXT,
  active     BOOLEAN DEFAULT TRUE,
  PRIMARY KEY (actocode, actno)
);

-- MR (헤더 단위의 "정답" 집합, 1:1 또는 1:n 허용)
CREATE TABLE tb_mr (
  mr_id        BIGSERIAL PRIMARY KEY,
  mrno         TEXT,                   -- MR 번호(없으면 NULL 가능)
  pjtno        VARCHAR NOT NULL,
  porser       VARCHAR NOT NULL,
  porseq       VARCHAR NOT NULL,
  revno        VARCHAR NOT NULL,
  actocode     VARCHAR NOT NULL,
  actno        VARCHAR NOT NULL,
  category_name TEXT,
  UNIQUE (pjtno, porser, porseq, revno, actocode, actno),
  FOREIGN KEY (actocode, actno) REFERENCES activity_codes(actocode, actno)
);

-- POR 상세(아이템 라인) - 아이템 "한 줄"이 모델의 입력 단위
CREATE TABLE tb_por_detail (
  pjtno     VARCHAR NOT NULL,
  porser    VARCHAR NOT NULL,
  porseq    VARCHAR NOT NULL,
  revno     VARCHAR NOT NULL,
  line_no   INT     NOT NULL,
  item_name TEXT,
  spec_text TEXT,
  -- 구조화 피처(카테고리/수치)
  mccsno    TEXT,
  block     TEXT,
  event     TEXT,
  sign      TEXT,       -- '+' / '-'
  duration  INT,
  deptcode  TEXT,
  shiptype  TEXT,
  PRIMARY KEY (pjtno, porser, porseq, revno, line_no)
);
CREATE INDEX idx_por_hdr ON tb_por_detail (pjtno, porser, porseq, revno);

-- 모델 메타/버전
CREATE TABLE model_versions (
  version_id BIGSERIAL PRIMARY KEY,
  trained_at TIMESTAMP DEFAULT NOW(),
  details    JSONB
);

-- (선택) 아이템 단위 예측 로그
CREATE TABLE item_predictions (
  id           BIGSERIAL PRIMARY KEY,
  task         TEXT NOT NULL,        -- 'item_act' | 'item_mr'
  pjtno        VARCHAR NOT NULL,
  porser       VARCHAR NOT NULL,
  porseq       VARCHAR NOT NULL,
  revno        VARCHAR NOT NULL,
  line_no      INT NOT NULL,
  label        TEXT NOT NULL,        -- 예: 'ACT:001' 또는 'MR-0001'
  confidence   REAL,
  topk         JSONB,
  model_version BIGINT REFERENCES model_versions(version_id),
  predicted_at TIMESTAMP DEFAULT NOW()
);

-- -------------------------
-- 학습용 뷰 (아이템 단위)
-- -------------------------

-- 1) 아이템 -> ACT 라벨 (actocode:actno)
CREATE OR REPLACE VIEW vw_training_item_act AS
SELECT
  d.pjtno, d.porser, d.porseq, d.revno, d.line_no,
  m.actocode, m.actno,
  d.item_name,
  d.spec_text,
  d.mccsno, d.block, d.event, d.sign, d.duration, d.deptcode, d.shiptype
FROM tb_por_detail d
JOIN tb_mr m
  ON m.pjtno=d.pjtno AND m.porser=d.porser AND m.porseq=d.porseq AND m.revno=d.revno
WHERE (m.actocode, m.actno) IN (SELECT actocode, actno FROM activity_codes WHERE active=TRUE);

-- 2) 아이템 -> MR 라벨 (mrno 또는 mr_id)
CREATE OR REPLACE VIEW vw_training_item_mr AS
SELECT
  d.pjtno, d.porser, d.porseq, d.revno, d.line_no,
  COALESCE(m.mrno, 'MR-'||LPAD(m.mr_id::text, 6, '0')) AS mr_label,
  d.item_name,
  d.spec_text,
  d.mccsno, d.block, d.event, d.sign, d.duration, d.deptcode, d.shiptype
FROM tb_por_detail d
JOIN tb_mr m
  ON m.pjtno=d.pjtno AND m.porser=d.porser AND m.porseq=d.porseq AND m.revno=d.revno;

-- -------------------------
-- 시드 데이터
-- -------------------------

INSERT INTO activity_codes(actocode, actno, name, active) VALUES
  ('ACT','001','MECHANICAL',TRUE),
  ('ACT','002','PIPING',TRUE),
  ('ACT','003','ELECTRIC',TRUE),
  ('ACT','004','INSTRUMENT',TRUE),
  ('ACT','005','STEEL',TRUE)
ON CONFLICT DO NOTHING;

-- 헤더별 정답 (한 헤더에 하나의 actcode/actno를 가정; 실제로는 1:n 가능)
INSERT INTO tb_mr(mrno, pjtno, porser, porseq, revno, actocode, actno, category_name) VALUES
('MR-000001','P1','S1','Q1','R1','ACT','001','MECHANICAL'),
('MR-000002','P2','S1','Q1','R1','ACT','002','PIPING'),
('MR-000003','P3','S1','Q1','R1','ACT','003','ELECTRIC'),
('MR-000004','P4','S1','Q1','R1','ACT','004','INSTRUMENT'),
('MR-000005','P5','S1','Q1','R1','ACT','005','STEEL')
ON CONFLICT DO NOTHING;

-- POR 라인(아이템) + 구조화 피처
-- P1: MECHANICAL
INSERT INTO tb_por_detail(pjtno,porser,porseq,revno,line_no,item_name,spec_text,
  mccsno,block,event,sign,duration,deptcode,shiptype) VALUES
('P1','S1','Q1','R1',1,'Pump ABC','40mm flange, Ø40mm, gasket included','M1','B1','E1','+',10,'D12A','CONTAINER'),
('P1','S1','Q1','R1',2,'Motor AC','5kW 220V, PN16','M1','B1','E1','+',10,'D12A','CONTAINER'),
('P1','S1','Q1','R1',3,'Coupling','DN50, steel','M1','B1','E1','+',10,'D12A','CONTAINER'),
('P1','S1','Q1','R1',4,'Gasket','fiber 40mm','M1','B1','E1','+',10,'D12A','CONTAINER');

-- P2: PIPING
INSERT INTO tb_por_detail VALUES
('P2','S1','Q1','R1',1,'Pipe SCH40','DN50 6m length','M2','B1','E2','+',20,'D20B','TANKER'),
('P2','S1','Q1','R1',2,'Flange WN','PN16, 50mm','M2','B1','E2','+',20,'D20B','TANKER'),
('P2','S1','Q1','R1',3,'Valve Gate','DN50, SS','M2','B1','E2','+',20,'D20B','TANKER'),
('P2','S1','Q1','R1',4,'Gasket Spiral','Ø50mm','M2','B1','E2','+',20,'D20B','TANKER');

-- P3: ELECTRIC
INSERT INTO tb_por_detail VALUES
('P3','S1','Q1','R1',1,'Power Cable','3C x 2.5mm','M3','B2','E3','-',30,'D30C','LNGC'),
('P3','S1','Q1','R1',2,'Breaker MCCB','AC 220V 40A','M3','B2','E3','-',30,'D30C','LNGC'),
('P3','S1','Q1','R1',3,'Plug','industrial, 3P','M3','B2','E3','-',30,'D30C','LNGC'),
('P3','S1','Q1','R1',4,'Panel','IP54 enclosure','M3','B2','E3','-',30,'D30C','LNGC');

-- P4: INSTRUMENT
INSERT INTO tb_por_detail VALUES
('P4','S1','Q1','R1',1,'Pressure Gauge','0~10 bar, 1/2 inch','M4','B3','E4','+',25,'D40D','CONTAINER'),
('P4','S1','Q1','R1',2,'Thermometer','-20~120°C, DN20','M4','B3','E4','+',25,'D40D','CONTAINER'),
('P4','S1','Q1','R1',3,'Transmitter','4-20mA, PN16','M4','B3','E4','+',25,'D40D','CONTAINER'),
('P4','S1','Q1','R1',4,'Manifold','3-way, SS316','M4','B3','E4','+',25,'D40D','CONTAINER');

-- P5: STEEL
INSERT INTO tb_por_detail VALUES
('P5','S1','Q1','R1',1,'Plate','SS400, 10mm','M5','B3','E5','+',15,'D50E','BULKER'),
('P5','S1','Q1','R1',2,'Angle','L50x50x5','M5','B3','E5','+',15,'D50E','BULKER'),
('P5','S1','Q1','R1',3,'Channel','C100x50x5','M5','B3','E5','+',15,'D50E','BULKER'),
('P5','S1','Q1','R1',4,'Beam','H200x100x5.5x8','M5','B3','E5','+',15,'D50E','BULKER');

COMMIT;
