-- ═══════════════════════════════════════════════════════════════════
-- SUPABASE SCHEMA — auto-cash precios de mercado
-- Ejecutar en el SQL Editor de Supabase
-- ═══════════════════════════════════════════════════════════════════

-- Tabla principal de precios de mercado (scrapeados de Patiotuerca)
CREATE TABLE IF NOT EXISTS precios_mercado (
    id              BIGSERIAL PRIMARY KEY,
    marca           TEXT NOT NULL,
    modelo_slug     TEXT NOT NULL,
    modelo_nombre   TEXT,
    año             INTEGER NOT NULL CHECK (año BETWEEN 1990 AND 2030),
    precio_ideal    NUMERIC(10,2) NOT NULL,
    precio_min      NUMERIC(10,2),
    precio_max      NUMERIC(10,2),
    url             TEXT,
    fecha_scrape    DATE NOT NULL DEFAULT CURRENT_DATE,
    enlace_listado  TEXT,
    fuente          TEXT DEFAULT 'patiotuerca',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Constraint único para el upsert (no duplicar mismo modelo/año/fecha)
ALTER TABLE precios_mercado
    ADD CONSTRAINT precios_mercado_unique
    UNIQUE (marca, modelo_slug, año, fecha_scrape);

-- Índices para consultas rápidas del modelo de precio
CREATE INDEX idx_precios_marca_modelo_año
    ON precios_mercado (marca, modelo_slug, año);

CREATE INDEX idx_precios_fecha
    ON precios_mercado (fecha_scrape DESC);

CREATE INDEX idx_precios_marca
    ON precios_mercado (marca);

-- Trigger para updated_at automático
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_precios_mercado_updated_at
    BEFORE UPDATE ON precios_mercado
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- Vista: precio más reciente por marca/modelo/año
CREATE OR REPLACE VIEW precios_mercado_latest AS
SELECT DISTINCT ON (marca, modelo_slug, año)
    id,
    marca,
    modelo_slug,
    modelo_nombre,
    año,
    precio_ideal,
    precio_min,
    precio_max,
    enlace_listado,
    fecha_scrape,
    fuente
FROM precios_mercado
ORDER BY marca, modelo_slug, año, fecha_scrape DESC;

-- Vista: histórico de precios por modelo (para detectar tendencia)
CREATE OR REPLACE VIEW precios_mercado_historico AS
SELECT
    marca,
    modelo_slug,
    modelo_nombre,
    año,
    fecha_scrape,
    precio_ideal,
    precio_min,
    precio_max,
    precio_ideal - LAG(precio_ideal) OVER (
        PARTITION BY marca, modelo_slug, año
        ORDER BY fecha_scrape
    ) AS variacion_precio,
    ROUND(
        (precio_ideal - LAG(precio_ideal) OVER (
            PARTITION BY marca, modelo_slug, año
            ORDER BY fecha_scrape
        )) / NULLIF(LAG(precio_ideal) OVER (
            PARTITION BY marca, modelo_slug, año
            ORDER BY fecha_scrape
        ), 0) * 100, 2
    ) AS variacion_pct
FROM precios_mercado
ORDER BY marca, modelo_slug, año, fecha_scrape;

-- ── Tabla de anuncios activos (para calcular tiempo de venta) ────────────────

CREATE TABLE IF NOT EXISTS anuncios_patiotuerca (
    id              TEXT PRIMARY KEY,   -- ID numérico del anuncio en Patiotuerca
    marca           TEXT,
    modelo          TEXT,
    año             INTEGER,
    ciudad          TEXT,
    precio_publicado NUMERIC(10,2),
    km              INTEGER,
    tipo_carroceria TEXT,
    link            TEXT,
    primera_vez_visto DATE NOT NULL,    -- fecha del primer snapshot donde apareció
    ultima_vez_visto  DATE NOT NULL,    -- fecha del último snapshot donde estaba
    vendido         BOOLEAN DEFAULT FALSE,
    dias_en_mercado INTEGER,            -- calculado al marcar como vendido
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_anuncios_marca_modelo ON anuncios_patiotuerca (marca, modelo);
CREATE INDEX idx_anuncios_vendido ON anuncios_patiotuerca (vendido);
CREATE INDEX idx_anuncios_año ON anuncios_patiotuerca (año);

-- Vista: tiempo promedio de venta por modelo
CREATE OR REPLACE VIEW tiempo_venta_por_modelo AS
SELECT
    marca,
    modelo,
    año,
    COUNT(*) FILTER (WHERE vendido = TRUE)  AS unidades_vendidas,
    ROUND(AVG(dias_en_mercado) FILTER (WHERE vendido = TRUE), 1) AS dias_promedio_venta,
    ROUND(AVG(precio_publicado) FILTER (WHERE vendido = TRUE), 2) AS precio_promedio_venta,
    MIN(precio_publicado) FILTER (WHERE vendido = TRUE) AS precio_min_venta,
    MAX(precio_publicado) FILTER (WHERE vendido = TRUE) AS precio_max_venta
FROM anuncios_patiotuerca
GROUP BY marca, modelo, año
HAVING COUNT(*) FILTER (WHERE vendido = TRUE) >= 3
ORDER BY unidades_vendidas DESC;

-- ── Row Level Security (RLS) — activar en producción ────────────────────────
-- ALTER TABLE precios_mercado ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE anuncios_patiotuerca ENABLE ROW LEVEL SECURITY;
-- Crear políticas según tu auth de Supabase

-- ── Verificación ─────────────────────────────────────────────────────────────
SELECT 'Schema creado correctamente' AS status;
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public'
AND table_name IN ('precios_mercado', 'anuncios_patiotuerca');
