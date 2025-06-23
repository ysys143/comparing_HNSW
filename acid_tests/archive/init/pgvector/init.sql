-- Initialize pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create test table
CREATE TABLE IF NOT EXISTS test_vectors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vector vector(384),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS idx_vectors_ivfflat ON test_vectors 
USING ivfflat (vector vector_cosine_ops)
WITH (lists = 100);

-- Create function to update timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for updated_at
CREATE TRIGGER update_test_vectors_updated_at 
BEFORE UPDATE ON test_vectors 
FOR EACH ROW 
EXECUTE FUNCTION update_updated_at_column();

-- Enable row-level security (for testing isolation)
ALTER TABLE test_vectors ENABLE ROW LEVEL SECURITY;

-- Create a test user with limited permissions
CREATE USER test_user WITH PASSWORD 'test_password';
GRANT ALL PRIVILEGES ON DATABASE vectordb TO test_user;
GRANT ALL PRIVILEGES ON TABLE test_vectors TO test_user;