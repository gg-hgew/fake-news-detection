from pymongo import MongoClient

MONGO_URI = "mongodb+srv://gayatrig2322_db_user:MyMongo123@cluster0.pqpjqza.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(MONGO_URI)

try:
    client.admin.command('ping')
    print("✅ Successfully connected to MongoDB!")
except Exception as e:
    print("❌ Connection failed:", e)
