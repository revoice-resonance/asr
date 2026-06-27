--
-- ASR Database Schema — Initial Migration
-- Extracted and fixed from the original PostgreSQL cluster dump.
-- This file ONLY contains the ASR database objects (corpus + asr_tasks).
-- Roles and database creation belong in the bootstrap/init-databases.sql script.
--
-- Fixes applied:
--   - validate_corpus_audio_fields(): renamed origin_audio_status → status,
--     origin_audio_path → file_path, origin_audio_filename → file_name.
--     Removed invalid status values (QUEUING/PENDING/SUCCESS are on asr_tasks,
--     not corpus). Validate: UPLOADING requires file_path NOT NULL;
--     UPLOADED requires file_name NOT NULL.
--

-- ==========================================================================
-- Functions
-- ==========================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION validate_corpus_audio_fields()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    -- UPLOADING 状态：必须有 file_path（临时文件路径）
    IF NEW.status = 'UPLOADING' THEN
        IF NEW.file_path IS NULL THEN
            RAISE EXCEPTION '状态 % 时，file_path 不能为空', NEW.status;
        END IF;
    END IF;

    -- UPLOADED 状态：必须有 file_name（已持久化的文件名）
    IF NEW.status = 'UPLOADED' THEN
        IF NEW.file_name IS NULL THEN
            RAISE EXCEPTION '状态 UPLOADED 时，file_name 不能为空';
        END IF;
    END IF;

    RETURN NEW;
END;
$$;

-- ==========================================================================
-- Sequences
-- ==========================================================================

CREATE SEQUENCE corpus_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

CREATE SEQUENCE asr_tasks_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;

-- ==========================================================================
-- Tables
-- ==========================================================================

-- Corpus (语料统一管理表)
CREATE TABLE corpus (
    id          bigint NOT NULL DEFAULT nextval('corpus_id_seq'::regclass),
    created_at  timestamp with time zone NOT NULL DEFAULT now(),
    updated_at  timestamp with time zone NOT NULL DEFAULT now(),

    -- File identity
    file_md5    character varying(32) NOT NULL,      -- 文件MD5（唯一标识，防重复）
    file_name   character varying(255) NOT NULL,     -- 原始文件名
    file_path   text,                                 -- 存储路径
    file_size   bigint,                               -- 文件大小（字节）
    duration    integer,                               -- 音频时长（毫秒）
    sample_rate integer,                               -- 采样率（Hz）
    channels    smallint DEFAULT 1,                   -- 声道数: 1=单声道, 2=立体声

    -- Metadata
    language     character varying(20) DEFAULT 'zh-CN'::character varying,  -- 语言代码
    status       character varying(20) DEFAULT 'UPLOADING'::character varying NOT NULL,
                  -- 上传状态: UPLOADING, UPLOADED, FAILED
    text_content text,                                 -- 识别结果文本（冗余字段，方便查询）

    -- Business context
    business_id   character varying(50),              -- 业务ID（如患者ID）
    business_type character varying(20),               -- 业务类型（如 PATIENT_VOICE, SAMPLE_AUDIO）
    tags          jsonb DEFAULT '[]'::jsonb NOT NULL, -- 标签（JSON数组）

    -- Lifecycle
    is_deleted boolean DEFAULT false NOT NULL,
    remark     text,

    -- Constraints
    CONSTRAINT corpus_pkey PRIMARY KEY (id),
    CONSTRAINT corpus_file_md5_key UNIQUE (file_md5),
    CONSTRAINT ck_corpus_status CHECK (
        status = ANY (ARRAY['UPLOADING', 'UPLOADED', 'FAILED'])
    )
);

COMMENT ON TABLE  corpus IS '语料统一管理表';
COMMENT ON COLUMN corpus.file_md5       IS '文件MD5（唯一标识，防重复）';
COMMENT ON COLUMN corpus.file_name      IS '原始文件名';
COMMENT ON COLUMN corpus.file_path      IS '存储路径';
COMMENT ON COLUMN corpus.file_size      IS '文件大小（字节）';
COMMENT ON COLUMN corpus.duration       IS '音频时长（毫秒）';
COMMENT ON COLUMN corpus.sample_rate    IS '采样率（Hz）';
COMMENT ON COLUMN corpus.channels       IS '声道数: 1=单声道, 2=立体声';
COMMENT ON COLUMN corpus.language       IS '语言代码，如 zh-CN, en-US';
COMMENT ON COLUMN corpus.status         IS '上传状态: UPLOADING, UPLOADED, FAILED';
COMMENT ON COLUMN corpus.text_content   IS '识别结果文本（冗余字段，方便查询）';
COMMENT ON COLUMN corpus.business_id    IS '业务ID（如患者ID）';
COMMENT ON COLUMN corpus.business_type  IS '业务类型（如 PATIENT_VOICE, SAMPLE_AUDIO）';
COMMENT ON COLUMN corpus.tags           IS '标签（JSON数组）';

-- ASR Tasks (ASR任务统一管理表)
CREATE TABLE asr_tasks (
    id          bigint NOT NULL DEFAULT nextval('asr_tasks_id_seq'::regclass),
    created_at  timestamp with time zone NOT NULL DEFAULT now(),
    updated_at  timestamp with time zone NOT NULL DEFAULT now(),

    corpus_id    bigint NOT NULL,                       -- 关联语料ID

    -- Task state
    status       character varying(20) DEFAULT 'PENDING'::character varying NOT NULL,
                  -- 状态: PENDING, PROCESSING, SUCCESS, FAILED
    asr_engine   character varying(30) DEFAULT 'WHISPER'::character varying NOT NULL,
                  -- ASR引擎: WHISPER, AZURE, ALIYUN, TENCENT, HUAWEI
    engine_config jsonb DEFAULT '{}'::jsonb NOT NULL,   -- 引擎配置参数

    -- Results
    result_text    text,                                 -- 识别结果文本
    confidence     numeric(5,4),                         -- 整体置信度 0~1
    result_detail  jsonb,                                -- 详细结果（时间戳、词级置信度等）
    processing_time integer,                             -- 处理耗时（毫秒）
    error_message  text,

    -- Timestamps
    started_at   timestamp with time zone,               -- 开始处理时间
    completed_at timestamp with time zone,               -- 完成时间

    -- Constraints
    CONSTRAINT asr_tasks_pkey PRIMARY KEY (id),
    CONSTRAINT ck_asr_task_status CHECK (
        status = ANY (ARRAY['PENDING', 'PROCESSING', 'SUCCESS', 'FAILED'])
    ),
    CONSTRAINT ck_asr_engine CHECK (
        asr_engine = ANY (ARRAY['WHISPER', 'AZURE', 'ALIYUN', 'TENCENT', 'HUAWEI'])
    )
);

COMMENT ON TABLE  asr_tasks IS 'ASR任务统一管理表';
COMMENT ON COLUMN asr_tasks.corpus_id       IS '关联语料ID';
COMMENT ON COLUMN asr_tasks.status          IS '状态: PENDING, PROCESSING, SUCCESS, FAILED';
COMMENT ON COLUMN asr_tasks.asr_engine      IS 'ASR引擎: WHISPER, AZURE, ALIYUN, TENCENT, HUAWEI';
COMMENT ON COLUMN asr_tasks.engine_config   IS '引擎配置参数';
COMMENT ON COLUMN asr_tasks.result_text     IS '识别结果文本';
COMMENT ON COLUMN asr_tasks.confidence      IS '整体置信度 0~1';
COMMENT ON COLUMN asr_tasks.result_detail   IS '详细结果（时间戳、词级置信度等）';
COMMENT ON COLUMN asr_tasks.processing_time IS '处理耗时（毫秒）';
COMMENT ON COLUMN asr_tasks.started_at      IS '开始处理时间';
COMMENT ON COLUMN asr_tasks.completed_at    IS '完成时间';

-- ==========================================================================
-- Indexes
-- ==========================================================================

-- Corpus indexes
CREATE INDEX idx_corpus_file_md5      ON corpus USING btree (file_md5);
CREATE INDEX idx_corpus_status        ON corpus USING btree (status);
CREATE INDEX idx_corpus_business_id   ON corpus USING btree (business_id);
CREATE INDEX idx_corpus_business_type ON corpus USING btree (business_type);
CREATE INDEX idx_corpus_created_at    ON corpus USING btree (created_at);
CREATE INDEX idx_corpus_is_deleted    ON corpus USING btree (is_deleted);

-- ASR task indexes
CREATE INDEX idx_asr_tasks_status       ON asr_tasks USING btree (status);
CREATE INDEX idx_asr_tasks_engine       ON asr_tasks USING btree (asr_engine);
CREATE INDEX idx_asr_tasks_corpus_id    ON asr_tasks USING btree (corpus_id);
CREATE INDEX idx_asr_tasks_created_at   ON asr_tasks USING btree (created_at);
CREATE INDEX idx_asr_tasks_completed_at ON asr_tasks USING btree (completed_at) WHERE (completed_at IS NOT NULL);
CREATE INDEX idx_asr_tasks_status_engine ON asr_tasks USING btree (status, asr_engine);

-- ==========================================================================
-- Triggers
-- ==========================================================================

CREATE TRIGGER trg_corpus_updated_at
    BEFORE UPDATE ON corpus
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trg_asr_tasks_updated_at
    BEFORE UPDATE ON asr_tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
