-- Creating the project catalog.
CREATE CATALOG IF NOT EXISTS finance_ml;

-- Selecting the project catalog.
USE CATALOG finance_ml;

-- Creating Medallion Architecture schemas.
CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS gold;
