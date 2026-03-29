-- Init.sql
CREATE TABLE IF NOT EXISTS users (
    id         SERIAL PRIMARY KEY,
    username   VARCHAR(100) UNIQUE NOT NULL,
    password   VARCHAR(64)  NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create table 
CREATE TABLE IF NOT EXISTS predictions (
    id                      SERIAL PRIMARY KEY,
    timestamp               TIMESTAMP DEFAULT NOW(),
    utilisateur             VARCHAR(100) NOT NULL,
    radius_worst            FLOAT,
    texture_worst           FLOAT,
    perimeter_worst         FLOAT,
    area_worst              FLOAT,
    smoothness_worst        FLOAT,
    compactness_worst       FLOAT,
    concavity_worst         FLOAT,
    concave_points_worst    FLOAT,
    symmetry_worst          FLOAT,
    fractal_dimension_worst FLOAT,
    prediction              VARCHAR(1),
    probability_pct         FLOAT
);