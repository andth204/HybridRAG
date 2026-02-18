import psycopg2
from minio import Minio
from src.config.settings import settings


def check_postgres():
    print("\n[1] PostgreSQL:")
    try:
        conn = psycopg2.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            dbname=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
        )
        cursor = conn.cursor()
        cursor.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' ORDER BY table_name;
        """)
        tables = [row[0] for row in cursor.fetchall()]

        if tables:
            print(f"   ✅ Connected | {settings.POSTGRES_USER}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}")
            print(f"   Tables ({len(tables)}):")
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table};")
                print(f"      • {table}: {cursor.fetchone()[0]} rows")
        else:
            print("   ⚠️  Connected but no tables found!")
            print(f"   → Run: Get-Content build/postgres/schema/init.sql | docker exec -i smartchat-postgres psql -U {settings.POSTGRES_USER} -d {settings.POSTGRES_DB}")
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"   ❌ Error: {e}")

def check_minio():
    print("\n[2] MinIO:")
    try:
        client = Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        bucket_names = [b.name for b in client.list_buckets()]
        print(f"   ✅ Connected | {settings.MINIO_ACCESS_KEY}@{settings.MINIO_ENDPOINT}")
        print(f"   Buckets: {bucket_names or 'None'}")
        if settings.MINIO_BUCKET_NAME not in bucket_names:
            print(f"   ⚠️  Default bucket '{settings.MINIO_BUCKET_NAME}' not found → will be auto-created on first use")
    except Exception as e:
        print(f"   ❌ Error: {e}")

check_postgres()
check_minio()